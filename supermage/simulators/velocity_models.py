import torch
from torch import pi, sqrt
from caskade import Module, forward, Param
from caustics.light.base import Source
import numpy as np
from numpy.polynomial.legendre import leggauss
from pykeops.torch import LazyTensor
from torch.nn.functional import conv2d, avg_pool2d


def leggauss_interval(n, t_low, t_high, device, dtype):
    """
    Return Gauss–Legendre nodes & weights mapped from [-1,1] -> [t_low, t_high].
    """
    x, w = leggauss(n)   # x in [-1,1], w for [-1,1]
    
    half_width = 0.5*(t_high - t_low)
    mid        = 0.5*(t_high + t_low)
    
    x_mapped = half_width*(x) + mid       # in [t_low, t_high]
    w_mapped = half_width*(w)            # scaled by the interval length
    
    x_mapped_t = torch.tensor(x_mapped, dtype=dtype, device=device)
    w_mapped_t = torch.tensor(w_mapped, dtype=dtype, device=device)
    return x_mapped_t, w_mapped_t


def transform_DE(t):
    """
    Double-exponential transform:
      u = exp((π/2) * sinh(t)),
      du/dt = (π/2)*cosh(t)*u.
    """
    u = torch.exp((np.pi/2.0)*torch.sinh(t))
    du_dt = (np.pi/2.0)*torch.cosh(t)*u
    return u, du_dt


class MGEVelocity(Module):
    def __init__(self, N_components: int, variable_M_to_L = False):
        super().__init__("MGEVelocity")
        self.N_components = N_components
        
        # Same parameter definitions
        self.surf   = Param("surf",   shape=(N_components,))
        self.sigma  = Param("sigma",  shape=(N_components,))
        self.qobs   = Param("qobs",   shape=(N_components,))
        if variable_M_to_L:
            self.M_to_L = Param("M_to_L", shape=(N_components,))
        else:
            self.M_to_L = Param("M_to_L", shape=())
        
        self.inc   = Param("inc",   shape=())
        self.m_bh  = Param("m_bh",  shape=())

    @forward
    def velocity(self, x, y, z,
                 surf=None, sigma=None, qobs=None, M_to_L=None,
                 inc=None, m_bh=None,
                 G=0.004301,
                 soft=0.0,
                 quad_points=128):
        """
        Compute the rotational velocity at points (x, y, z), but use a
        double-exponential transform from [0,1] -> (0,∞).
        """
        device = x.device
        dtype  = x.dtype
        
        # --- geometry & mass density the same as your original ---
        cos_inc = torch.cos(inc)
        sin_inc = torch.sin(inc)
        q_arg = qobs**2 - cos_inc**2
        if torch.any(q_arg <= 0.0):
            raise ValueError("Inclination too low for deprojection")
        q_intr = torch.sqrt(q_arg) / sin_inc
        
        sqrt_2pi = np.sqrt(2.0*np.pi)
        mass_density = surf * M_to_L * qobs / (q_intr * sigma * sqrt_2pi)
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        R_flat = torch.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
        N_points = R_flat.shape[0]
        
        # Scale by median sigma
        scale = sigma.quantile(q = 0.5)
        sigma_sc = sigma / scale
        R_sc = R_flat / scale
        soft_sc = soft / scale

        mds = sigma_sc.quantile(q = 0.5)
        mxs = torch.max(sigma_sc)
        xlim = torch.tensor([
            torch.arcsinh(torch.log(1e-7 * mds)*2/np.pi),
            torch.arcsinh(torch.log(1e3  * mxs)*2/np.pi)
        ])
        
        # --- Gauss–Legendre on [0,1] ---
        t_1d, w_1d = leggauss_interval(quad_points, xlim[0].item(), xlim[1].item(), device, dtype)
        
        # --- Double-exponential transform t->u in (0,∞) ---
        u_1d, du_1d = transform_DE(t_1d)
        
        # KeOps LazyTensors for efficient computation
        # Reshape for KeOps: [N,1] and [1,Q] formats
        R_i = LazyTensor(R_sc.reshape(N_points, 1, 1))  # [N,1,1]
        u_j = LazyTensor(u_1d.reshape(1, quad_points, 1))  # [1,Q,1]
        w_j = LazyTensor(w_1d.reshape(1, quad_points, 1))  # [1,Q,1]
        du_j = LazyTensor(du_1d.reshape(1, quad_points, 1))  # [1,Q,1]
        # (1) pack all Gaussian parameters in one vector of shape (C,)
        sigma_vec     = LazyTensor(sigma_sc.view(1, 1, -1))      # (1,1,C)
        q_intr_vec    = LazyTensor(q_intr.view(1, 1, -1))
        mass_den_vec  = LazyTensor(mass_density.view(1, 1, -1))


        one_plus     = 1 + u_j
        exponent     = -0.5 * R_i**2 / (sigma_vec**2 * one_plus)
        exp_val      = exponent.exp()
        denom        = one_plus**2 * (q_intr_vec**2 + u_j).sqrt()
        
        term         = (q_intr_vec * mass_den_vec * exp_val) / denom
        weighted     = term * du_j * w_j
        
        # First reduce on the quadrature index j, then on the component index m
        integral_val = weighted.sum(axis=1).sum(axis=1).squeeze()
            
        
        # Multiply by the factor 2 pi G scale^2
        vc2_mge_factor = 2.0 * np.pi * G * (scale**2)
        vc2_mge = vc2_mge_factor * integral_val
        
        # Black hole contribution
        vc2_bh = G * 10**m_bh / scale * (R_sc**2 + soft_sc**2).pow(-1.5)
        
        # Final velocity
        v_rot_flat = R_sc * torch.sqrt(vc2_mge + vc2_bh)
        v_rot = v_rot_flat.reshape_as(x)
        
        return v_rot

class Nuker_MGE(Module):
    def __init__(self, N_MGE_components: int, Nuker_NN, r_min, r_max, device):
        super().__init__("NukerMGE")
        self.N_components = N_MGE_components
        self.MGE = MGEVelocity(self.N_components)
        self.MGE.surf = torch.ones((self.N_components), device = device).to(dtype = torch.float64)
        self.MGE.sigma = torch.ones((self.N_components), device = device).to(dtype = torch.float64)
        self.MGE.qobs = torch.ones((self.N_components), device = device).to(dtype = torch.float64)
        self.MGE.M_to_L = 1
        self.NN = Nuker_NN

        inner_slope=torch.tensor([3], device = device)
        outer_slope=torch.tensor([3], device = device)
        low_Gauss=torch.log10(r_min/torch.sqrt(inner_slope))
        high_Gauss=torch.log10(r_max/torch.sqrt(outer_slope))
        dx=(high_Gauss-low_Gauss)/self.N_components
        self.sigma = 10**(low_Gauss+(0.5+torch.arange(self.N_components, device = device))*dx).to(dtype = torch.float64)
        
        self.inc   = Param("inc",   shape=())
        self.m_bh  = Param("m_bh",  shape=())
        self.MGE.inc = self.inc
        self.MGE.m_bh = self.m_bh

        self.q   = Param("qobs",   shape=())

        self.alpha = Param("alpha", shape=(1, ))
        self.beta = Param("beta", shape=(1, ))
        self.gamma = Param("gamma", shape=(1, ))
        self.r_b = Param("break_r", shape = ())
        self.I_b = Param("intensity_r_b", shape = ())

    @forward
    def velocity(self, x, y, z,
                 inc=None, m_bh=None, q = None,
                 alpha = None, beta = None, gamma = None, r_b = None, I_b = None,
                 G=0.004301,
                 soft=0.0,
                 quad_points=128):
        device = x.device
        dtype  = x.dtype

        qobs = q*torch.ones(self.N_components, device = device).to(dtype = torch.float64)

        NN_input = torch.cat([alpha, gamma, beta]).to(torch.float32)
        NN_output = self.NN.forward(NN_input).to(torch.float64)
        
        surf = NN_output*I_b
        MGE_sigma = self.sigma*r_b
        v_rot = self.MGE.velocity(x, y, z, surf = surf, sigma = MGE_sigma, qobs = qobs, G = G, soft = soft, quad_points = quad_points)
        return v_rot






