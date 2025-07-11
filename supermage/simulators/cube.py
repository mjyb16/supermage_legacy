import torch
from torch import vmap
from math import pi
from caskade import Module, forward, Param
from pykeops.torch import LazyTensor
from supermage.utils.math_utils import DoRotation, DoRotationT
from supermage.utils.cube_tools import freq_to_vel_systemic_torch, freq_to_vel_absolute_torch
import torch.nn.functional as F
import caustics
from caustics.light import Pixelated
from torch.nn.functional import avg_pool2d, conv2d
import numpy as np
import math

def make_spatial_axis(
    fov_half: float,
    n_out: int,
    upscale: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
):
    """
    Returns
    -------
    x_hi : (n_out*upscale,) - fine-grid pixel *centres*
    dx_hi : width of one fine pixel
    """
    # total number of fine pixels
    n_hi  = n_out * upscale

    # coarse-pixel size and fine-pixel size
    dx_coarse = 2 * fov_half / n_out
    dx_hi     = dx_coarse / upscale          # = 2*fov_half / (n_out*upscale)

    # coordinate of the very first fine-pixel centre
    x0 = -fov_half + 0.5 * dx_hi             # centres, not edges!

    idx = torch.arange(n_hi, device=device, dtype=dtype)
    x_hi = x0 + idx * dx_hi                  # [-fov,+fov] exclusive of edges

    return x_hi, dx_hi


def make_frequency_axis(freqs_coarse: torch.Tensor,
                        upscale: int,
                        device="cuda",
                        dtype=torch.float64):
    """
    Build the high-resolution frequency axis whose simple block average
    (or mean pooling) collapses back to `freqs_coarse`.
    """
    Δf_coarse = freqs_coarse[1] - freqs_coarse[0]          # coarse step
    Δf_fine   = Δf_coarse / upscale                        # fine step
    n_fine    = freqs_coarse.numel() * upscale             # total fine samples

    # first fine-pixel centre sits *upscale/2* fine steps below the
    # first coarse-pixel centre
    f0_fine = freqs_coarse[0] - (upscale - 1) * Δf_fine / 2

    freqs_fine = f0_fine + Δf_fine * torch.arange(n_fine,
                                                  device=device,
                                                  dtype=dtype)
    return freqs_fine, Δf_fine






def rotate_rect_phys(cube, angle_rad, U, U_y):
    """
    Rotate (D, H, W) without distortion, accounting for
    - pixel aspect     (dx/dy = U / U_y)
    - normalised-grid  anisotropy (W_p ≠ H_p)
    """
    D, H, W = cube.shape
    dev, dt = cube.device, cube.dtype

    # 1. symmetric padding
    pad = int(0.5*(math.sqrt(2)-1)*max(H, W)) + 2
    cube_p = F.pad(cube, (pad, pad, pad, pad))     # (D, H_p, W_p)
    H_p, W_p = H + 2*pad, W + 2*pad

    # 2. scaling factor for off-diagonals in normalised coords
    kappa = (U_y / U) * (H_p - 1) / (W_p - 1)      # ≥ 1
    inv_k = 1.0 / kappa

    a = -angle_rad
    theta = cube.new_tensor([[ math.cos(a), -math.sin(a)*kappa , 0.0],
                             [ math.sin(a)*inv_k ,  math.cos(a) , 0.0]])

    grid = F.affine_grid(theta.unsqueeze(0),
                         size=(1, D, H_p, W_p),
                         align_corners=True)
    cube_big = F.grid_sample(cube_p.unsqueeze(0), grid,
                             mode='bilinear',  # or 'nearest'
                             padding_mode='border',
                             align_corners=True).squeeze(0)

    # 3. centre-crop back to original rectangle
    y0 = (H_p - H)//2
    x0 = (W_p - W)//2
    return cube_big[:, y0:y0+H, x0:x0+W]            # (D, H, W)

class MinMajThinCubeSimulatorKeOps(Module):
    def __init__(
        self,
        velocity_model,
        intensity_model,
        freqs,
        systemic_or_redshift,
        frequency_upscale,
        cube_fov_half,
        image_res_out,
        major_upscale,
        minor_upscale,
        line="co21",
        device="cuda",
        dtype=torch.float64,
    ):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        # free parameters that can be fitted
        self.inclination     = Param("inclination", None)   # rad
        self.sky_rot         = Param("sky_rot", None)       # rad
        self.line_broadening = Param("line_broadening", None)
        self.velocity_shift  = Param("velocity_shift", None)

        # bookkeeping
        self.systemic_or_redshift = systemic_or_redshift
        self.line = line

        # -------------------------------------------------------
        # INTERNAL RESOLUTIONS
        # -------------------------------------------------------
        self.frequency_upscale = frequency_upscale
        self.image_upscale     = major_upscale
        self.N_out   = image_res_out        # coarse pixels along one axis
        self.side_hi = self.N_out * minor_upscale  # fine-grid side length after rotation

        # -------------------------------------------------------
        # 2-D IMAGE GRID  (x_img, y_img)
        # -------------------------------------------------------
        self.U   = major_upscale                 # e.g. 2 or 3
        self.U_y = minor_upscale   # Nyquist along minor axis
        
        dx_coarse = 2 * cube_fov_half / image_res_out
        x_hi, _   = make_spatial_axis(
            cube_fov_half,
            image_res_out,
            self.U,
            device=self.device,
            dtype=self.dtype,
        )
        y_hi, _   = make_spatial_axis(
            cube_fov_half,
            image_res_out,
            self.U_y,
            device=self.device,
            dtype=self.dtype,
        )
        self.H_gal = x_hi.numel()        # = N_out * U
        self.W_gal = y_hi.numel()        # = N_out * U_y
        self.x_gal, self.y_gal = torch.meshgrid(x_hi, y_hi, indexing="ij")

        # -------------------------------------------------------
        # FREQUENCY GRID  (no change)
        # -------------------------------------------------------
        self.freqs = freqs                                              # coarse axis (1-D)
        self.freqs_upsampled, _ = make_frequency_axis(                  # fine axis (1-D)
            self.freqs,
            self.frequency_upscale,
            device=self.device,
            dtype=self.dtype,
        )
        self.frequency_res = self.freqs_upsampled.numel()               # D_fine

        # -------------------------------------------------------
        # KeOps LazyTensor setup for memory-efficient operations
        # -------------------------------------------------------
        # Create LazyTensors for spatial coordinates
        # Flatten spatial coordinates for KeOps operations
        x_flat = self.x_gal.flatten()  # (H*W,)
        y_flat = self.y_gal.flatten()  # (H*W,)
        
        # Create LazyTensors with proper dimensions and axis specifications
        # axis=0 means these are "i" variables (spatial pixels)
        self.x_keops = LazyTensor(x_flat.view(-1, 1), axis=0)
        self.y_keops = LazyTensor(y_flat.view(-1, 1), axis=0)
        
        # For frequency/velocity labels, we'll create them in the forward pass
        # since they depend on velocity_shift parameter

        # -------------------------------------------------------
        # CONSTANTS
        # -------------------------------------------------------
        self.pi = torch.tensor(np.pi, device=self.device, dtype=self.dtype)

        # Make sure the downstream models can "see" the inclination
        self.velocity_model.inc = self.inclination

    @forward
    def forward(
        self,
        inclination=None,
        sky_rot=None,
        line_broadening=None,
        velocity_shift=None,
    ):
        """
        Thin-disk forward model using KeOps for memory-efficient operations
        """
        
        # ---------------------------------------------------------------
        # 1.  SKY  →  GALAXY radius (using KeOps for memory efficiency)
        # ---------------------------------------------------------------
        cos_i, sin_i = torch.cos(inclination), torch.sin(inclination)
        
        # Create LazyTensors for transformed coordinates
        y_gal_keops = self.y_keops / cos_i
        x_gal_keops = self.x_keops
        
        # Compute R_map using KeOps
        R_map_keops = (x_gal_keops**2 + y_gal_keops**2 + 1e-12).sqrt()  # (H*W, 1)
        cos_theta_keops = x_gal_keops / R_map_keops
        
        # ---------------------------------------------------------------
        # 2.  INTENSITY  I(R) - evaluate and flatten
        # ---------------------------------------------------------------
        # Materialize R_map using sum reduction (forces evaluation)
        R_map_flat = R_map_keops.sum(dim=1)  # This forces materialization: (H*W,)
        R_map_dense = R_map_flat.view(self.H_gal, self.W_gal)  # reshape to (H,W)
        I_pix = self.intensity_model.brightness(R_map_dense)  # (H,W)
        I_pix_flat = I_pix.flatten()  # (H*W,)
        I_pix_keops = LazyTensor(I_pix_flat.view(-1, 1), axis=0)  # (H*W, 1)
        
        # ---------------------------------------------------------------
        # 3.  VELOCITY  v_rot(R)  and line-of-sight projection
        # ---------------------------------------------------------------
        # Similarly, materialize for velocity model evaluation
        v_rot_dense = self.velocity_model.velocity(R_map_dense)  # (H,W)
        v_rot_flat = v_rot_dense.flatten()  # (H*W,)
        v_rot_keops = LazyTensor(v_rot_flat.view(-1, 1), axis=0)  # (H*W, 1)
        
        # Materialize cos_theta for line-of-sight velocity computation
        cos_theta_flat = cos_theta_keops.sum(dim=1)  # materialize: (H*W,)
        cos_theta_keops_mat = LazyTensor(cos_theta_flat.view(-1, 1), axis=0)  # (H*W, 1)
        
        # Line-of-sight velocity using KeOps
        v_los_keops = v_rot_keops * sin_i * cos_theta_keops_mat  # (H*W, 1)
        
        # ---------------------------------------------------------------
        # 4.  FREQUENCY axis  → velocity labels (1-D)
        # ---------------------------------------------------------------
        v_labels_1d, _ = freq_to_vel_absolute_torch(
            self.freqs_upsampled, self.line,
            device=self.device, dtype=self.dtype
        )                         # shape (Dν,)
        
        v_labels_1d = v_labels_1d - velocity_shift            # systemic shift
        # axis=1 means this is a "j" variable (frequency/velocity channels)
        v_labels_keops = LazyTensor(v_labels_1d.view(1, -1), axis=1)  # (1, Dν)
        
        # ---------------------------------------------------------------
        # 5.  GAUSSIAN broadening using KeOps (THIS IS THE KEY IMPROVEMENT)
        # ---------------------------------------------------------------
        sig2 = line_broadening ** 2
        norm = 1.0 / torch.sqrt(2 * self.pi * sig2)
        
        # Memory-efficient Gaussian computation using KeOps
        # v_labels_keops: (1, Dν) with axis=1, v_los_keops: (H*W, 1) with axis=0
        # The subtraction broadcasts to (H*W, Dν)
        diff_keops = v_labels_keops - v_los_keops  # (H*W, Dν)
        
        # Gaussian PDF computation
        pdf_keops = (-0.5 * diff_keops**2 / sig2).exp()  # (H*W, Dν)
        
        # Multiply by intensity and normalization
        cube_keops = pdf_keops * I_pix_keops * norm  # (H*W, Dν)
        
        # ---------------------------------------------------------------
        # 6.  Materialize the result and reshape
        # ---------------------------------------------------------------
        # Materialize the LazyTensor by using a reduction operation
        # We'll use a dummy variable to force evaluation
        dummy_var = LazyTensor(torch.ones(1, 1, device=self.device, dtype=self.dtype), axis=1)
        cube_materialized = (cube_keops * dummy_var).sum(dim=1)  # (H*W, Dν)
        
        # Reshape to the expected cube format
        cube_hi = cube_materialized.view(self.H_gal, self.W_gal, -1)  # (H, W, Dν)
        
        # ---------------------------------------------------------------
        # 7.  Re-order axes & downsample
        # ---------------------------------------------------------------
        cube_hi = cube_hi.permute(2, 0, 1)        # (Dν, H, W)
        
        # (i) rotate directly with physical aspect ratio
        cube_rot = rotate_rect_phys(cube_hi, sky_rot, self.U, self.U_y)
        
        # (ii) final 3-D pooling to coarse cube
        cube_5d  = cube_rot.unsqueeze(0).unsqueeze(0)
        cube_ds  = F.avg_pool3d(
                      cube_5d,
                      kernel_size=(self.frequency_upscale, self.U, self.U_y),
                      stride     =(self.frequency_upscale, self.U, self.U_y)
                  ).squeeze(0).squeeze(0)          # (N_freq_out, N_out, N_out)
        
        return cube_ds



class MinMajThinCubeSimulator(Module):
    def __init__(
        self,
        velocity_model,
        intensity_model,
        freqs,
        systemic_or_redshift,
        frequency_upscale,
        cube_fov_half,
        image_res_out,
        major_upscale,
        minor_upscale,
        line="co21",
        device="cuda",
        dtype=torch.float64,
    ):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        # free parameters that can be fitted
        self.inclination     = Param("inclination", None)   # rad
        self.sky_rot         = Param("sky_rot", None)       # rad
        self.line_broadening = Param("line_broadening", None)
        self.velocity_shift  = Param("velocity_shift", None)

        # bookkeeping
        self.systemic_or_redshift = systemic_or_redshift
        self.line = line

        # -------------------------------------------------------
        # INTERNAL RESOLUTIONS
        # -------------------------------------------------------
        self.frequency_upscale = frequency_upscale
        self.image_upscale     = major_upscale#image_upscale
        self.N_out   = image_res_out        # coarse pixels along one axis
        self.side_hi = self.N_out * minor_upscale  # fine-grid side length after rotation

        # -------------------------------------------------------
        # 2-D IMAGE GRID  (x_img, y_img)
        # -------------------------------------------------------
        # NB:  Thin disk ⇒ no need to carry a z-axis in memory
        #self.pixelscale_pc = 2 * cube_fov_half / self.image_res             # pc / fine-pixel
        #x_hi, _ = make_spatial_axis(
        #    cube_fov_half,
        #    image_res_out,
        #    image_upscale,
        #    device=self.device,
        #    dtype=self.dtype,
        #)
        # sky-plane meshgrid (shape: H×W)
        #self.img_x, self.img_y = torch.meshgrid(x_hi, x_hi, indexing="ij")  # (H, W)

        self.U   = major_upscale                 # e.g. 2 or 3
        self.U_y = minor_upscale   # Nyquist along minor axis
        
        dx_coarse = 2 * cube_fov_half / image_res_out
        x_hi, _   = make_spatial_axis(
            cube_fov_half,
            image_res_out,
            self.U,
            device=self.device,
            dtype=self.dtype,
        )
        y_hi, _   = make_spatial_axis(
            cube_fov_half,
            image_res_out,
            self.U_y,
            device=self.device,
            dtype=self.dtype,
        )
        self.H_gal = x_hi.numel()        # = N_out * U
        self.W_gal = y_hi.numel()        # = N_out * U_y
        self.x_gal, self.y_gal = torch.meshgrid(x_hi, y_hi, indexing="ij")

        # -------------------------------------------------------
        # FREQUENCY GRID  (no change)
        # -------------------------------------------------------
        self.freqs = freqs                                              # coarse axis (1-D)
        self.freqs_upsampled, _ = make_frequency_axis(                  # fine axis (1-D)
            self.freqs,
            self.frequency_upscale,
            device=self.device,
            dtype=self.dtype,
        )
        self.frequency_res = self.freqs_upsampled.numel()               # D_fine

        # -------------------------------------------------------
        # KeOps LazyTensor holding ν labels, broadcast over pixels
        # -------------------------------------------------------
        # We need a 3-D tensor of shape (H, W, D_fine, 1) for KeOps
        cube_z_labels = self.freqs_upsampled.view(1, 1, -1, 1)           # (1,1,D,1)
        cube_z_labels = self.freqs_upsampled.view(1, 1, -1, 1).expand(
            self.H_gal, self.W_gal, -1, 1)          # use rectangular H×W here
        Dν = self.frequency_res
        self.cube_nu_keops = LazyTensor(          # shape (1,1,Dν,1)
            self.freqs_upsampled.view(1, 1, Dν, 1)
        )

        # -------------------------------------------------------
        # CONSTANTS
        # -------------------------------------------------------
        self.pi = torch.tensor(np.pi, device=self.device, dtype=self.dtype)

        # Make sure the downstream models can “see” the inclination
        self.velocity_model.inc = self.inclination

    @forward
    def forward(
        self,
        inclination=None,
        sky_rot=None,
        line_broadening=None,
        velocity_shift=None,
    ):
        """
        Thin-disk forward model – *dense* PyTorch version
        (no KeOps, so easier to debug).
        Returns
        -------
        cube_downsampled : Tensor  (N_freq_out, H_out, W_out)
        """
    
        # ---------------------------------------------------------------
        # 1.  SKY  →  GALAXY radius
        # ---------------------------------------------------------------
        #x_sky, y_sky = self.img_x, self.img_y
        #cos_pa, sin_pa = torch.cos(sky_rot), torch.sin(sky_rot)
        cos_i,  sin_i  = torch.cos(inclination), torch.sin(inclination)
    
        # CCW rotation by PA  (φ measured from +y (North) through +x (East))
        #x_rot =  cos_pa * x_sky - sin_pa * y_sky
        #y_rot =  sin_pa * x_sky + cos_pa * y_sky
        y_gal = self.y_gal / cos_i
        x_gal = self.x_gal
    
        R_map     = torch.sqrt(x_gal**2 + y_gal**2 + 1e-12)   # (H,W)
        cos_theta = x_gal / R_map
    
        # ---------------------------------------------------------------
        # 2.  INTENSITY  I(R)
        # ---------------------------------------------------------------
        I_pix = self.intensity_model.brightness(R_map)        # (H,W)
    
        # ---------------------------------------------------------------
        # 3.  VELOCITY  v_rot(R)  and line-of-sight projection
        # ---------------------------------------------------------------
        v_rot = self.velocity_model.velocity(R_map)           # (H,W)
        v_los = v_rot * sin_i * cos_theta                     # (H,W)
    
        # ---------------------------------------------------------------
        # 4.  FREQUENCY axis  → velocity labels (1-D)
        # ---------------------------------------------------------------
        v_labels_1d, _ = freq_to_vel_absolute_torch(
            self.freqs_upsampled, self.line,
            device=self.device, dtype=self.dtype
        )                         # shape (Dν,)
    
        v_labels_1d = v_labels_1d - velocity_shift            # systemic shift
    
        # broadcast to (H,W,Dν)
        v_labels = v_labels_1d.view(1, 1, -1).expand(self.H_gal, self.W_gal, -1)
        v_los_b  = v_los.unsqueeze(-1)                        # (H,W,1)
    
        # ---------------------------------------------------------------
        # 5.  GAUSSIAN broadening
        # ---------------------------------------------------------------
        sig2 = line_broadening ** 2
        norm = 1.0 / torch.sqrt(2 * self.pi * sig2)
    
        pdf = torch.exp(-0.5 * (v_labels - v_los_b) ** 2 / sig2)  # (H,W,Dν)
        cube_hi = pdf * I_pix.unsqueeze(-1) * norm                # (H,W,Dν)
    
        # ---------------------------------------------------------------
        # 6.  Re-order axes & downsample
        # ---------------------------------------------------------------
        cube_hi = cube_hi.permute(2, 0, 1)        # (Dν, H, W)
        
        # (i) rotate directly with physical aspect ratio
        cube_rot = rotate_rect_phys(cube_hi, sky_rot, self.U, self.U_y)
        
        # (ii) final 3-D pooling to coarse cube
        cube_5d  = cube_rot.unsqueeze(0).unsqueeze(0)
        cube_ds  = F.avg_pool3d(
                      cube_5d,
                      kernel_size=(self.frequency_upscale, self.U, self.U_y),
                      stride     =(self.frequency_upscale, self.U, self.U_y)
                  ).squeeze(0).squeeze(0)          # (N_freq_out, N_out, N_out)
        
        return cube_ds

class ThinCubeSimulator(Module):
    def __init__(
        self,
        velocity_model,
        intensity_model,
        freqs,
        systemic_or_redshift,
        frequency_upscale,
        cube_fov_half,
        image_res_out,
        image_upscale,
        line="co21",
        device="cuda",
        dtype=torch.float64,
    ):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        # free parameters that can be fitted
        self.inclination     = Param("inclination", None)   # rad
        self.sky_rot         = Param("sky_rot", None)       # rad
        self.line_broadening = Param("line_broadening", None)
        self.velocity_shift  = Param("velocity_shift", None)

        # bookkeeping
        self.systemic_or_redshift = systemic_or_redshift
        self.line = line

        # -------------------------------------------------------
        # INTERNAL RESOLUTIONS
        # -------------------------------------------------------
        self.image_res        = image_res_out * image_upscale
        self.frequency_upscale = frequency_upscale
        self.image_upscale     = image_upscale

        # -------------------------------------------------------
        # 2-D IMAGE GRID  (x_img, y_img)
        # -------------------------------------------------------
        # NB:  Thin disk ⇒ no need to carry a z-axis in memory
        self.pixelscale_pc = 2 * cube_fov_half / self.image_res             # pc / fine-pixel
        x_hi, _ = make_spatial_axis(
            cube_fov_half,
            image_res_out,
            image_upscale,
            device=self.device,
            dtype=self.dtype,
        )
        # sky-plane meshgrid (shape: H×W)
        self.img_x, self.img_y = torch.meshgrid(x_hi, x_hi, indexing="ij")  # (H, W)

        # -------------------------------------------------------
        # FREQUENCY GRID  (no change)
        # -------------------------------------------------------
        self.freqs = freqs                                              # coarse axis (1-D)
        self.freqs_upsampled, _ = make_frequency_axis(                  # fine axis (1-D)
            self.freqs,
            self.frequency_upscale,
            device=self.device,
            dtype=self.dtype,
        )
        self.frequency_res = self.freqs_upsampled.numel()               # D_fine

        # -------------------------------------------------------
        # KeOps LazyTensor holding ν labels, broadcast over pixels
        # -------------------------------------------------------
        # We need a 3-D tensor of shape (H, W, D_fine, 1) for KeOps
        cube_z_labels = self.freqs_upsampled.view(1, 1, -1, 1)           # (1,1,D,1)
        cube_z_labels = cube_z_labels.expand(
            self.image_res, self.image_res, -1, 1                        # (H,W,D,1)
        )
        Dν = self.frequency_res
        self.cube_nu_keops = LazyTensor(          # shape (1,1,Dν,1)
            self.freqs_upsampled.view(1, 1, Dν, 1)
        )

        # -------------------------------------------------------
        # CONSTANTS
        # -------------------------------------------------------
        self.pi = torch.tensor(np.pi, device=self.device, dtype=self.dtype)

        # Make sure the downstream models can “see” the inclination
        self.velocity_model.inc = self.inclination

    @forward
    def forward(
        self,
        inclination=None,
        sky_rot=None,
        line_broadening=None,
        velocity_shift=None,
    ):
        """
        Thin-disk forward model – *dense* PyTorch version
        (no KeOps, so easier to debug).
        Returns
        -------
        cube_downsampled : Tensor  (N_freq_out, H_out, W_out)
        """
    
        # ---------------------------------------------------------------
        # 1.  SKY  →  GALAXY radius
        # ---------------------------------------------------------------
        x_sky, y_sky = self.img_x, self.img_y
        cos_pa, sin_pa = torch.cos(sky_rot), torch.sin(sky_rot)
        cos_i,  sin_i  = torch.cos(inclination), torch.sin(inclination)
    
        # CCW rotation by PA  (φ measured from +y (North) through +x (East))
        x_rot =  cos_pa * x_sky - sin_pa * y_sky
        y_rot =  sin_pa * x_sky + cos_pa * y_sky
        y_gal = y_rot / cos_i
        x_gal = x_rot
    
        R_map     = torch.sqrt(x_gal**2 + y_gal**2 + 1e-12)   # (H,W)
        cos_theta = x_gal / R_map
    
        # ---------------------------------------------------------------
        # 2.  INTENSITY  I(R)
        # ---------------------------------------------------------------
        I_pix = self.intensity_model.brightness(R_map)        # (H,W)
    
        # ---------------------------------------------------------------
        # 3.  VELOCITY  v_rot(R)  and line-of-sight projection
        # ---------------------------------------------------------------
        v_rot = self.velocity_model.velocity(R_map)           # (H,W)
        v_los = v_rot * sin_i * cos_theta                     # (H,W)
    
        # ---------------------------------------------------------------
        # 4.  FREQUENCY axis  → velocity labels (1-D)
        # ---------------------------------------------------------------
        v_labels_1d, _ = freq_to_vel_absolute_torch(
            self.freqs_upsampled, self.line,
            device=self.device, dtype=self.dtype
        )                         # shape (Dν,)
    
        v_labels_1d = v_labels_1d - velocity_shift            # systemic shift
    
        # broadcast to (H,W,Dν)
        v_labels = v_labels_1d.view(1, 1, -1).expand(
            self.image_res, self.image_res, -1
        )
        v_los_b  = v_los.unsqueeze(-1)                        # (H,W,1)
    
        # ---------------------------------------------------------------
        # 5.  GAUSSIAN broadening
        # ---------------------------------------------------------------
        sig2 = line_broadening ** 2
        norm = 1.0 / torch.sqrt(2 * self.pi * sig2)
    
        pdf = torch.exp(-0.5 * (v_labels - v_los_b) ** 2 / sig2)  # (H,W,Dν)
        cube_hi = pdf * I_pix.unsqueeze(-1) * norm                # (H,W,Dν)
    
        # ---------------------------------------------------------------
        # 6.  Re-order axes & downsample
        # ---------------------------------------------------------------
        cube_hi = cube_hi.permute(2, 0, 1)          # (Dν, H, W)
        cube_5d = cube_hi.unsqueeze(0).unsqueeze(0) # (1,1,D,H,W)
    
        cube_ds = F.avg_pool3d(
            cube_5d,
            kernel_size=(self.frequency_upscale,
                         self.image_upscale,
                         self.image_upscale),
            stride=(self.frequency_upscale,
                    self.image_upscale,
                    self.image_upscale),
        ).squeeze(0).squeeze(0)                     # (D_out, H_out, W_out)
    
        return cube_ds


class ThickCubeSimulator(Module):
    """
    Parameters for init
    ----------
    velocity_model : Module
        SuperMAGE velocity field model
    intensity_model : Module
        SuperMAGE intensity/brightness field model
    freqs : Tensor (1D)
        Frequencies at which to evaluate the cube.
    velocity_upscale : int
        Factor by which velocity_res_out is multiplied. High internal resolution needed to prevent aliasing (rec. 5x).
    velocity_min, velocity_max : float
        Velocity range (in km/s).
    cube_fov_half : float
        Spatial extent of the cube (pc). Gives half length of one side.
    image_res_out : int
        Final (downsampled) number of image pixels (2D) returned for the cube's spatial dimensions.
    image_upscale : int
        Factor by which image_res_out is multiplied. High internal resolution can be needed to prevent aliasing.
    """
    def __init__(self, velocity_model, intensity_model, freqs, systemic_or_redshift, frequency_upscale, cube_fov_half, image_res_out, image_upscale, line="co21", device = "cuda", dtype = torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        self.inclination = Param("inclination", None)
        self.velocity_model.inc = self.inclination
        self.sky_rot = Param("sky_rot", None)
        self.line_broadening = Param("line_broadening", None)
        self.velocity_shift = Param("velocity_shift", None)

        # Determine whether we want systemic velocity or redshift
        self.systemic_or_redshift = systemic_or_redshift
        self.line = line
        
        # Internal resolutions
        self.image_res    = image_res_out * image_upscale
        self.frequency_upscale = frequency_upscale
        self.image_upscale    = image_upscale

        # Image grid        
        self.pixelscale_pc = cube_fov_half*2/(self.image_res)
        x_hi, dx_hi = make_spatial_axis(cube_fov_half, image_res_out, image_upscale,
                                device=self.device, dtype=self.dtype)
        coords = torch.meshgrid(x_hi, x_hi, x_hi, indexing="ij")
        self.img_x, self.img_y, self.img_z = coords

        # Frequency grid
        self.freqs = freqs
        self.v_z = torch.zeros_like(self.img_z, device = self.device)
        self.freqs_upsampled, _ = make_frequency_axis(self.freqs, self.frequency_upscale, device=self.device, dtype=self.dtype)
        self.frequency_res = self.freqs_upsampled.numel()

        # Keops version of frequency grid
        cube_z_labels = self.freqs_upsampled * torch.ones((self.image_res, self.image_res, self.frequency_res), device = self.device, dtype = self.dtype)
        self.cube_z_l_keops = LazyTensor(cube_z_labels.unsqueeze(-1).expand(self.image_res, self.image_res, self.frequency_res, 1)[:, :, :, None, :])
        
        # Constants
        self.pi = torch.tensor(pi, device = self.device)

    @forward
    def forward(
        self,
        inclination=None, sky_rot=None, line_broadening=None, velocity_shift = None, 
    ):
        rot_x, rot_y, rot_z = DoRotation(self.img_x, self.img_y, self.img_z, inclination, sky_rot, device = self.device)

        source_intensity_cube = self.intensity_model.brightness(rot_x, rot_y, rot_z)
        intensity_cube = source_intensity_cube.unsqueeze(-1)
        #del source_intensity_cube
        torch.cuda.empty_cache()

        v_abs = self.velocity_model.velocity(rot_x, rot_y, rot_z)
        theta_rot = torch.atan2(rot_y, rot_x)
        #del rot_x, rot_y, rot_z
        torch.cuda.empty_cache()
        
        v_x = -v_abs * torch.sin(theta_rot)
        v_y = v_abs * torch.cos(theta_rot)
        #del v_abs, theta_rot
        torch.cuda.empty_cache()

        v_x_r, v_y_r, v_los_r = DoRotationT(v_x, v_y, self.v_z, inclination, sky_rot, device = self.device)
        v_los_keops = LazyTensor(v_los_r.unsqueeze(-1).expand(self.image_res, self.image_res, self.image_res, 1)[:, :, None, :, :])
        #del v_x, v_y, v_x_r, v_y_r, v_los_r
        torch.cuda.empty_cache()
        
        if self.systemic_or_redshift == "systemic":
            velocity_labels_unshifted, _ = freq_to_vel_absolute_torch(self.cube_z_l_keops, self.line, device = self.device, dtype = self.dtype) 
            velocity_labels = velocity_labels_unshifted - velocity_shift
        elif self.systemic_or_redshift == "redshift":
            print("Need to implement redshift")
            return
        else:
            print("Please specify 'redshift' or 'systemic'")
            return
        
        sig_sq = line_broadening**2
        prob_density_matrix = (-0.5*(velocity_labels - v_los_keops)**2 / sig_sq).exp() # 2D probability density, axis 1 is output velocity grid, axis 2 is LOS position
        cube = (prob_density_matrix @ intensity_cube) * (1/torch.sqrt(2*self.pi*sig_sq)) # Matrix inner product which results in a summation along axis 2 (LOS position)
        #del prob_density_matrix, intensity_cube, v_los_keops
        torch.cuda.empty_cache()
        
        cube_final = torch.squeeze(cube)
        #del cube
        torch.cuda.empty_cache()
        
        cube_final_3D = cube_final.permute(2, 0, 1)  # (frequency_res, image_res, image_res)
        #del cube_final
        torch.cuda.empty_cache()

        # Expand to (N=1, C=1, D, H, W)
        cube_5d = cube_final_3D.unsqueeze(0).unsqueeze(0)
        #del cube_final_3D
        torch.cuda.empty_cache()

        # Compute integer pooling sizes (assumes integral ratios)
        kernel_d = self.frequency_upscale
        kernel_h = self.image_upscale
        kernel_w = self.image_upscale

        # Use average pooling to downsample
        cube_downsampled = F.avg_pool3d(
            cube_5d,
            kernel_size=(kernel_d, kernel_h, kernel_w),
            stride=(kernel_d, kernel_h, kernel_w)
        )
        #del cube_5d
        torch.cuda.empty_cache()
        # Shape => (1, 1, frequency_res_out, image_res_out, image_res_out)
        cube_downsampled = cube_downsampled.squeeze(0).squeeze(0)
        torch.cuda.empty_cache()
        return cube_downsampled


class CubePosition(Module):
    """
    Generates an off-center (x, y position offset in arcsec) cube with the correct padding to match the FOV of the data. Note that the x offset parameter is in negative RA so that it increases from left to right.
    Parameters
    ----------
    source_cube (Module): The source 3D cube to be lensed.
    pixelscale_source (float): The pixel scale for the source cube.
    pixelscale_lens (float): The pixel scale for the output grid.
    pixels_x_source (int): The number of pixels in the source cube in the x-direction.
    pixels_x_lens (int): The number of pixels in the output grid in the x-direction.
    upsample_factor (int): The factor by which to upsample the image for lensing.
    name (str, optional): The name of the module. Default is "sim".
    """
    def __init__(
        self,
        source_cube,
        pixelscale_source,
        pixelscale_lens,
        pixels_x_source,
        pixels_x_lens,
        upsample_factor,
        name: str = "sim",
    ):
        super().__init__(name)
        
        self.source_cube = source_cube
        self.device = source_cube.device
        self.upsample_factor = upsample_factor
        self.src = Pixelated(name="source", shape=(pixels_x_source, pixels_x_source), pixelscale=pixelscale_source, image = torch.zeros((pixels_x_source, pixels_x_source)))

        # Create the high-resolution grid
        thx, thy = caustics.utils.meshgrid(
            pixelscale_lens / upsample_factor,
            upsample_factor * pixels_x_lens,
            dtype=source_cube.dtype, device = source_cube.device
        )

        self.thx = thx
        self.thy = thy

    @forward
    def forward(self):
        cube = self.source_cube.forward()

        def lens_channel(image):
            return self.src.brightness(self.thx, self.thy, image = image)
        
        # Ray-trace to get the lensed positions
        lensed_cube = vmap(lens_channel)(cube)
        del cube

        # Downsample to the desired resolution
        lensed_cube = avg_pool2d(lensed_cube[:, None], self.upsample_factor)[:, 0]
        torch.cuda.empty_cache()
        return lensed_cube