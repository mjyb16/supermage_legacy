import torch
from torch import pi, sqrt
from caskade import Module, forward, Param
from caustics.light.base import Source
import numpy as np
from numpy.polynomial.legendre import leggauss
from pykeops.torch import LazyTensor
from torch.nn.functional import conv2d, avg_pool2d
from functools import lru_cache
import math


@lru_cache(maxsize=None)
def _leggauss_const(n, dtype, device):
    x_np, w_np = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(x_np, dtype=dtype, device = device),
            torch.as_tensor(w_np, dtype=dtype, device = device))

# 2.  Pure-Torch mapping keeps autograd alive and avoids graph breaks.
def leggauss_interval(n, t_low, t_high, device=None, dtype=None):
    x0, w0 = _leggauss_const(n, dtype, device)

    half = 0.5 * (t_high - t_low)        # tensor ops → differentiable
    mid  = 0.5 * (t_high + t_low)

    # allow t_low / t_high to be batched – add a dim for broadcasting
    x = half.unsqueeze(-1) * x0 + mid.unsqueeze(-1)
    w = half.unsqueeze(-1) * w0
    return x, w


def transform_DE(t):
    """
    Double-exponential transform:
      u = exp((π/2) * sinh(t)),
      du/dt = (π/2)*cosh(t)*u.
    """
    u = torch.exp((np.pi/2.0)*torch.sinh(t))
    du_dt = (np.pi/2.0)*torch.cosh(t)*u
    return u, du_dt
    

def interpolate_velocity(R_grid: torch.Tensor,
                         R_map : torch.Tensor,
                         v_grid: torch.Tensor) -> torch.Tensor:
    """
    1-D linear interpolation on an arbitrary monotonic grid.
    Any value outside [R_grid[0], R_grid[-1]] is clamped to the edges.
    Works on CUDA tensors, keeps gradients, avoids out-of-bounds.
    """
    # 1. Clamp the query points to the grid range
    R_clamp = R_map.clamp(min=R_grid[0], max=R_grid[-1])

    # 2. Locate the interval: first index such that R_grid[idx_hi] ≥ R_clamp
    idx_hi = torch.searchsorted(R_grid, R_clamp, right=False)

    #   For values equal to R_grid[-1] we still get idx_hi == len(R_grid)
    idx_hi = idx_hi.clamp(max=R_grid.numel() - 1)

    # 3. Lower neighbour
    idx_lo = (idx_hi - 1).clamp(min=0)

    # 4. Gather the two bracketing points
    R_lo, R_hi = R_grid[idx_lo], R_grid[idx_hi]
    v_lo, v_hi = v_grid[idx_lo], v_grid[idx_hi]

    # 5. Linear weight (when R_lo == R_hi, weight → 0)
    w = torch.where(
        R_hi == R_lo,
        torch.zeros_like(R_lo),
        (R_clamp - R_lo) / (R_hi - R_lo)
    )

    return v_lo + w * (v_hi - v_lo)


class MGEVelocityIntr(Module):
    """
    Identical setup to `ThinMGEVelocity`, but now we use the intrinsic q directly.
    """
    def __init__(self, N_components: int, device, dtype, quad_points=128, radius_res = 4096, variable_M_to_L = False, soft = 0.0, G=0.004301):
        """
        Soft: softening length in parsecs
        """
        super().__init__("MGEVelocityIntr")
        self.device = device
        self.dtype  = dtype
        
        self.N_components = N_components
        
        # Same parameter definitions
        self.surf   = Param("surf",   shape=(N_components,))
        self.sigma  = Param("sigma",  shape=(N_components,))
        self.qintr   = Param("qintr",   shape=(N_components,))
        if variable_M_to_L:
            self.M_to_L = Param("M_to_L", shape=(N_components,))
        else:
            self.M_to_L = Param("M_to_L", shape=())
        
        self.m_bh  = Param("m_bh",  shape=())
        self.quad_points = quad_points
        self.radius_res = radius_res

        self.soft = soft
        self.G = G
        self.inc = Param("inc",   shape=())

    def radial_velocity(self, R_flat,
                 surf, sigma, qintr, M_to_L,
                 inc, m_bh):
        """
        Compute the rotational velocity at radii R_flat, but use a
        double-exponential transform from [0,1] -> (0,∞).
        """        
        sqrt_2pi = np.sqrt(2.0*np.pi)
        qobs = torch.sqrt(qintr**2 * (torch.sin(inc))**2 + (torch.cos(inc))**2)
        mass_density = surf * M_to_L * qobs / (qintr * sigma * sqrt_2pi)
        
        N_points = R_flat.shape[0]
        
        # Scale by median sigma
        scale = sigma.quantile(q = 0.5)
        sigma_sc = sigma / scale
        R_sc = R_flat / scale
        soft_sc = self.soft / scale

        mds = sigma_sc.quantile(q = 0.5)
        mxs = torch.max(sigma_sc)
        xlim = (torch.arcsinh(torch.log(1e-7 * mds)*2/np.pi),
            torch.arcsinh(torch.log(1e3  * mxs)*2/np.pi))
        
        # --- Gauss–Legendre on [0,1] ---
        lo, hi = xlim
        t_1d, w_1d = leggauss_interval(self.quad_points, lo, hi, device = self.device, dtype = self.dtype)
        
        # --- Double-exponential transform t->u in (0,∞) ---
        u_1d, du_1d = transform_DE(t_1d)
        
        R_i  = R_sc.view(-1, 1, 1)                     # (N,1,1)
        u_j  = u_1d.view(1,  -1, 1)                    # (1,Q,1)torch.linspace(-1, 1, nx, device=device, dtype=dtype) * pixelscale * (nx - 1) / 2
        w_j  = w_1d.view(1,  -1, 1)                    # (1,Q,1)
        du_j = du_1d.view(1, -1, 1)                    # (1,Q,1)
        
        sigma_mat    = sigma_sc.view(1, 1, -1)         # (1,1,C)
        qintr_mat   = qintr.view(1, 1, -1)           # (1,1,C)
        mass_den_mat = mass_density.view(1, 1, -1)     # (1,1,C)
        
        # ---- kernel -----------------------------------------------------------------
        one_plus = 1.0 + u_j                           # (1,Q,1)
        exp_val  = torch.exp(-0.5 * R_i.pow(2) /
                             (sigma_mat.pow(2) * one_plus))          # (N,Q,C)
        
        denom    = one_plus.pow(2) * torch.sqrt(qintr_mat.pow(2) + u_j)
        
        term      = (qintr_mat * mass_den_mat * exp_val) / denom    # (N,Q,C)
        weighted  = term * du_j * w_j                                # (N,Q,C)
        
        # ---- quadrature & component sums -------------------------------------------
        integral_val = weighted.sum(dim=1).sum(dim=1)   # (N,)
        
        # ---- finish exactly as before ----------------------------------------------
        vc2_mge_factor = 2.0 * np.pi * self.G * (scale**2)
        vc2_mge = vc2_mge_factor * integral_val
        
        vc2_bh = self.G * 10**m_bh / scale * (R_sc**2 + soft_sc**2).pow(-1.5)
        
        v_rot_flat = R_sc * torch.sqrt(vc2_mge + vc2_bh)   # (N,)
        
        return v_rot_flat
        
    @forward
    def velocity(
        self,
        R_map,                           # 2-D tensor [H,W]  (pc)
        surf=None, sigma=None, qintr=None, M_to_L=None, inc = None, m_bh=None
    ):
        """
        Returns v_rot(R) for every pixel in the sky plane.
        """

        Rmin = torch.as_tensor(self.soft, dtype=self.dtype, device=self.device)
        Rmax = R_map.max()

        # 1-D lookup table (same as before)
        R_grid = torch.logspace(
            torch.log10(Rmin),
            torch.log10(Rmax),
            self.radius_res,
            device=self.device,
            dtype=self.dtype,
        )
        v_grid = self.radial_velocity(
            R_grid, surf, sigma, qintr, M_to_L, inc, m_bh
        )

        # interpolate onto the pixel-by-pixel radii
        v_abs = interpolate_velocity(R_grid, R_map, v_grid)      # (H,W)

        return v_abs

class ThinMGEVelocity(Module):
    """
    Identical parameters to `MGEVelocity`, but the `forward` method now
    takes *only* a radius map R_map instead of (x,y,z) grids.
    """
    def __init__(self, N_components: int, device, dtype, quad_points=128, radius_res = 4096, variable_M_to_L = False, soft = 0.0, G=0.004301):
        """
        Soft: softening length in parsecs
        """
        super().__init__("MGEVelocity")
        self.device = device
        self.dtype  = dtype
        
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
        self.quad_points = quad_points
        self.radius_res = radius_res

        self.soft = soft
        self.G = G

    def radial_velocity(self, R_flat,
                 surf, sigma, qobs, M_to_L,
                 inc, m_bh):
        """
        Compute the rotational velocity at radii R_flat, but use a
        double-exponential transform from [0,1] -> (0,∞).
        """
        
        # --- geometry & mass density the same as your original ---
        cos_inc = torch.cos(inc)
        sin_inc = torch.sin(inc)
        q_arg = qobs**2 - cos_inc**2
        if torch.any(q_arg <= 0.0):
            raise ValueError("Inclination too low for deprojection")
        q_intr = torch.sqrt(q_arg) / sin_inc
        
        sqrt_2pi = np.sqrt(2.0*np.pi)
        mass_density = surf * M_to_L * qobs / (q_intr * sigma * sqrt_2pi)
        
        N_points = R_flat.shape[0]
        
        # Scale by median sigma
        scale = sigma.quantile(q = 0.5)
        sigma_sc = sigma / scale
        R_sc = R_flat / scale
        soft_sc = self.soft / scale

        mds = sigma_sc.quantile(q = 0.5)
        mxs = torch.max(sigma_sc)
        xlim = (torch.arcsinh(torch.log(1e-7 * mds)*2/np.pi),
            torch.arcsinh(torch.log(1e3  * mxs)*2/np.pi))
        
        # --- Gauss–Legendre on [0,1] ---
        lo, hi = xlim
        t_1d, w_1d = leggauss_interval(self.quad_points, lo, hi, device = self.device, dtype = self.dtype)

        
        # --- Double-exponential transform t->u in (0,∞) ---
        u_1d, du_1d = transform_DE(t_1d)
        
        R_i  = R_sc.view(-1, 1, 1)                     # (N,1,1)
        u_j  = u_1d.view(1,  -1, 1)                    # (1,Q,1)torch.linspace(-1, 1, nx, device=device, dtype=dtype) * pixelscale * (nx - 1) / 2
        w_j  = w_1d.view(1,  -1, 1)                    # (1,Q,1)
        du_j = du_1d.view(1, -1, 1)                    # (1,Q,1)
        
        sigma_mat    = sigma_sc.view(1, 1, -1)         # (1,1,C)
        q_intr_mat   = q_intr.view(1, 1, -1)           # (1,1,C)
        mass_den_mat = mass_density.view(1, 1, -1)     # (1,1,C)
        
        # ---- kernel -----------------------------------------------------------------
        one_plus = 1.0 + u_j                           # (1,Q,1)
        exp_val  = torch.exp(-0.5 * R_i.pow(2) /
                             (sigma_mat.pow(2) * one_plus))          # (N,Q,C)
        
        denom    = one_plus.pow(2) * torch.sqrt(q_intr_mat.pow(2) + u_j)
        
        term      = (q_intr_mat * mass_den_mat * exp_val) / denom    # (N,Q,C)
        weighted  = term * du_j * w_j                                # (N,Q,C)
        
        # ---- quadrature & component sums -------------------------------------------
        integral_val = weighted.sum(dim=1).sum(dim=1)   # (N,)
        
        # ---- finish exactly as before ----------------------------------------------
        vc2_mge_factor = 2.0 * np.pi * self.G * (scale**2)
        vc2_mge = vc2_mge_factor * integral_val
        
        vc2_bh = self.G * 10**m_bh / scale * (R_sc**2 + soft_sc**2).pow(-1.5)
        
        v_rot_flat = R_sc * torch.sqrt(vc2_mge + vc2_bh)   # (N,)
        
        return v_rot_flat
        
    @forward
    def velocity(
        self,
        R_map,                           # 2-D tensor [H,W]  (pc)
        surf=None, sigma=None, qobs=None, M_to_L=None,
        inc=None, m_bh=None,
    ):
        """
        Returns v_rot(R) for every pixel in the sky plane.
        """

        Rmin = torch.as_tensor(self.soft, dtype=self.dtype, device=self.device)
        Rmax = R_map.max()

        # 1-D lookup table (same as before)
        R_grid = torch.logspace(
            torch.log10(Rmin),
            torch.log10(Rmax),
            self.radius_res,
            device=self.device,
            dtype=self.dtype,
        )
        v_grid = self.radial_velocity(
            R_grid, surf, sigma, qobs, M_to_L, inc, m_bh
        )

        # interpolate onto the pixel-by-pixel radii
        v_abs = interpolate_velocity(R_grid, R_map, v_grid)      # (H,W)

        return v_abs


class MGEVelocity(Module):
    def __init__(self, N_components: int, device, dtype, quad_points=128, radius_res = 4096, variable_M_to_L = False, soft = 0.0, G=0.004301):
        """
        Soft: softening length in parsecs
        """
        super().__init__("MGEVelocity")
        self.device = device
        self.dtype  = dtype
        
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
        self.quad_points = quad_points
        self.radius_res = radius_res

        self.soft = soft
        self.G = G

    def radial_velocity(self, R_flat,
                 surf, sigma, qobs, M_to_L,
                 inc, m_bh):
        """
        Compute the rotational velocity at radii R_flat, but use a
        double-exponential transform from [0,1] -> (0,∞).
        """
        
        # --- geometry & mass density the same as your original ---
        cos_inc = torch.cos(inc)
        sin_inc = torch.sin(inc)
        q_arg = qobs**2 - cos_inc**2
        if torch.any(q_arg <= 0.0):
            raise ValueError("Inclination too low for deprojection")
        q_intr = torch.sqrt(q_arg) / sin_inc
        
        sqrt_2pi = np.sqrt(2.0*np.pi)
        mass_density = surf * M_to_L * qobs / (q_intr * sigma * sqrt_2pi)
        
        N_points = R_flat.shape[0]
        
        # Scale by median sigma
        scale = sigma.quantile(q = 0.5)
        sigma_sc = sigma / scale
        R_sc = R_flat / scale
        soft_sc = self.soft / scale

        mds = sigma_sc.quantile(q = 0.5)
        mxs = torch.max(sigma_sc)
        xlim = torch.tensor([
            torch.arcsinh(torch.log(1e-7 * mds)*2/np.pi),
            torch.arcsinh(torch.log(1e3  * mxs)*2/np.pi)
        ])
        
        # --- Gauss–Legendre on [0,1] ---
        t_1d, w_1d = leggauss_interval(self.quad_points, xlim[0].item(), xlim[1].item(), self.device, self.dtype)
        
        # --- Double-exponential transform t->u in (0,∞) ---
        u_1d, du_1d = transform_DE(t_1d)
        
        R_i  = R_sc.view(-1, 1, 1)                     # (N,1,1)
        u_j  = u_1d.view(1,  -1, 1)                    # (1,Q,1)torch.linspace(-1, 1, nx, device=device, dtype=dtype) * pixelscale * (nx - 1) / 2
        w_j  = w_1d.view(1,  -1, 1)                    # (1,Q,1)
        du_j = du_1d.view(1, -1, 1)                    # (1,Q,1)
        
        sigma_mat    = sigma_sc.view(1, 1, -1)         # (1,1,C)
        q_intr_mat   = q_intr.view(1, 1, -1)           # (1,1,C)
        mass_den_mat = mass_density.view(1, 1, -1)     # (1,1,C)
        
        # ---- kernel -----------------------------------------------------------------
        one_plus = 1.0 + u_j                           # (1,Q,1)
        exp_val  = torch.exp(-0.5 * R_i.pow(2) /
                             (sigma_mat.pow(2) * one_plus))          # (N,Q,C)
        
        denom    = one_plus.pow(2) * torch.sqrt(q_intr_mat.pow(2) + u_j)
        
        term      = (q_intr_mat * mass_den_mat * exp_val) / denom    # (N,Q,C)
        weighted  = term * du_j * w_j                                # (N,Q,C)
        
        # ---- quadrature & component sums -------------------------------------------
        integral_val = weighted.sum(dim=1).sum(dim=1)   # (N,)
        
        # ---- finish exactly as before ----------------------------------------------
        vc2_mge_factor = 2.0 * np.pi * self.G * (scale**2)
        vc2_mge = vc2_mge_factor * integral_val
        
        vc2_bh = self.G * 10**m_bh / scale * (R_sc**2 + soft_sc**2).pow(-1.5)
        
        v_rot_flat = R_sc * torch.sqrt(vc2_mge + vc2_bh)   # (N,)
        
        return v_rot_flat

    @forward
    def velocity(self, rot_x, rot_y, rot_z,
                 surf=None, sigma=None, qobs=None, M_to_L=None,
                 inc=None, m_bh=None):
        """
        Compute the rotational velocity at points (x, y, z), but use a
        double-exponential transform from [0,1] -> (0,∞).
        """
        # cylindrical radii in the **galaxy** frame
        R_map = torch.sqrt(rot_x**2 + rot_y**2)
        Rmin = self.soft
        Rmax = R_map.max()

        R_grid = torch.logspace(np.log10(Rmin), np.log10(Rmax), self.radius_res,
                        device=self.device, dtype=self.dtype)
        
        v_grid = self.radial_velocity(R_grid, surf, sigma, qobs, M_to_L, inc, m_bh)

        v_abs = interpolate_velocity(R_grid, R_map, v_grid)

        return v_abs
        
       

class Nuker_MGE(Module):
    def __init__(self, N_MGE_components: int, Nuker_NN, r_min, r_max, device, dtype, quad_points=128):
        super().__init__("NukerMGE")
        self.N_components = N_MGE_components
        self.MGE = MGEVelocity(self.N_components, quad_points = quad_points, dtype = dtype, device = device)
        self.MGE.surf = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.sigma = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.qobs = torch.ones((self.N_components), device = device).to(dtype = dtype)
        self.MGE.M_to_L = 1
        self.NN = Nuker_NN

        inner_slope=torch.tensor([3], device = device, dtype = dtype)
        outer_slope=torch.tensor([3], device = device, dtype = dtype)
        low_Gauss=torch.log10(r_min/torch.sqrt(inner_slope))
        high_Gauss=torch.log10(r_max/torch.sqrt(outer_slope))
        dx=(high_Gauss-low_Gauss)/self.N_components
        self.sigma = 10**(low_Gauss+(0.5+torch.arange(self.N_components, device = device))*dx).to(dtype = dtype)
        
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
    def velocity(self, rot_x, rot_y, rot_z,
                 inc=None, m_bh=None, q = None,
                 alpha = None, beta = None, gamma = None, r_b = None, I_b = None,
                 G=0.004301,
                 soft=0.0):
        device = R_flat.device
        dtype  = R_flat.dtype

        qintr = q*torch.ones(self.N_components, device = device).to(dtype = dtype)

        NN_input = torch.cat([alpha, gamma, beta]).to(torch.float32)
        NN_output = self.NN.forward(NN_input).to(torch.float64)
        
        surf = NN_output*I_b
        MGE_sigma = self.sigma*r_b
        v_rot = self.MGE.velocity(rot_x, rot_y, rot_z, surf = surf, sigma = MGE_sigma, qintr = qintr, G = G, soft = soft)
        return v_rot






