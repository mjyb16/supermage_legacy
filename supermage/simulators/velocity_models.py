import torch
from torch import pi, sqrt
from caskade import Module, forward, Param
from caustics.light.base import Source
import numpy as np
from pykeops.torch import LazyTensor
from torch.nn.functional import conv2d, avg_pool2d


class MGEVelocity(Module):
    """
    This class handles MGE parameters
    and computes rotational velocities. It uses Param for all parameters.

    Parameters
    ----------
    N_components : int
        Number of Gaussian components in the MGE.
    """

    def __init__(self, N_components: int):
        super().__init__()
        self.N_components = N_components
        
        # Stellar dynamical parameters
        self.surf = Param("surf", shape=(N_components,))
        self.sigma = Param("sigma", shape=(N_components,))
        self.qobs = Param("qobs", shape=(N_components,))
        self.M_to_L = Param("M_to_L", shape=(N_components,))
        
        # Scalar parameters
        self.inc = Param("inc", None)   # inclination in radians
        self.m_bh = Param("m_bh", None) # black hole mass
        # If needed, you can also add x0, y0, pa, etc. as in MGE brightness if relevant.

    @forward
    def velocity(self, x, y, z, surf, sigma, qobs, M_to_L, inc, m_bh):
        """
        Compute the rotational velocity at points (x, y, z).
        Parameters are automatically taken from the Param attributes.

        * x, y, z: Coordinates (pc)
        * surf, sigma, qobs: MGE parameters for each Gaussian
        * M_to_L: Mass-to-light ratio for each Gaussian
        * inc: Inclination angle
        * m_bh: Black hole mass
        """
        G = 4.517103e-30  # pc^3 / (M_star * s^2)
        
        device = x.device
        # Compute q_j intrinsic axial ratios from qobs and inc
        cos_inc = torch.cos(inc)
        sin_inc = torch.sin(inc)
        q_j = torch.sqrt((qobs**2 - cos_inc**2) / sin_inc**2)
        q_j = q_j.clamp(min=1e-3)
        
        # Compute total mass for each Gaussian
        # L_j = 2*pi*surf*sigma^2*qobs
        L_j = 2 * pi * surf * sigma**2 * qobs
        M_j = M_to_L * L_j
        
        # Flatten coordinates
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        R_flat = torch.sqrt(x_flat**2 + y_flat**2)
        
        # Reshape for broadcasting
        N = x_flat.shape[0]
        R_flat = R_flat[:, None]  # [N,1]
        sigma_j = sigma[None, :]  # [1,N_gauss]
        q_j = q_j[None, :]        # [1,N_gauss]
        M_j = M_j[None, :]        # [1,N_gauss]
        
        # Compute exponential terms
        exp_term = torch.exp(-R_flat**2 / (2 * sigma_j**2))
        
        # Compute radial force for each Gaussian
        G_const = G / (sqrt(2*torch.tensor(pi, device = "cuda"))*sigma_j**3 * q_j)
        F_R_j = G_const * M_j * R_flat * exp_term   # [N, N_gauss]
        
        F_R_total = F_R_j.sum(dim=1)  # [N]
        
        # Add black hole force
        epsilon = 1e-10
        R_safe = R_flat.squeeze().clamp(min=epsilon)
        F_R_BH = G * m_bh / R_safe**2
        F_R_total += F_R_BH
        
        v_rot_flat = torch.sqrt(R_safe * F_R_total)
        v_rot = v_rot_flat.reshape(x.shape)
        
        return v_rot