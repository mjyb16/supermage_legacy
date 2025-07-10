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


def gaussian_blur_2d(cube_spatial,       # (D, H, W)   fine-grid cube
                     sigma_px: float):
    """
    Depth-wise separable Gaussian blur across (H, W).
    cube_spatial : tensor of shape (D, H, W)
    sigma_px     : σ in *fine* pixels (float, > 0)

    Returns a blurred tensor of the same shape.
    """
    if sigma_px <= 0:
        return cube_spatial                # safety early-exit

    D, H, W          = cube_spatial.shape
    dev, dt          = cube_spatial.device, cube_spatial.dtype

    # --- build 1-D Gaussian --------------------------------------------------
    # kernel half-width:  ⌈3σ⌉ is plenty (covers 99.7 %)
    half = int(math.ceil(3 * sigma_px))
    k    = 2 * half + 1                     # full width  (odd)
    x    = torch.arange(-half, half + 1, device=dev, dtype=dt)
    g1d  = torch.exp(-0.5 * (x / sigma_px) ** 2)
    g1d /= g1d.sum()                       # normalise to unit DC gain

    # make depth-wise conv weights: shape (D,1,k,1) then (D,1,1,k)
    gH = g1d.view(1, 1, k, 1).expand(D, 1, k, 1)
    gW = g1d.view(1, 1, 1, k).expand(D, 1, 1, k)

    # depth-wise separable convolution = two 1-D passes
    cube = cube_spatial.unsqueeze(0)       # (1, D, H, W)

    cube = F.conv2d(cube, gH, padding=(half, 0), groups=D)
    cube = F.conv2d(cube, gW, padding=(0, half), groups=D)

    return cube.squeeze(0)                 # (D, H, W)

def lanczos_blur_2d(cube_spatial,
                    sigma_px: float,
                    a: int = 3):            # Lanczos order: 2 or 3 usual
    """
    cube_spatial : (D, H, W)   fine-grid cube
    sigma_px     : target σ in FINE pixels  (float > 0)
    a            : size of the Lanczos window (integer ≥ 2)

    Returns tensor of same shape (D, H, W)
    """
    if sigma_px <= 0:
        return cube_spatial                     # early exit (no blur)

    D, H, W   = cube_spatial.shape
    dev, dt   = cube_spatial.device, cube_spatial.dtype

    # --- build 1-D Lanczos kernel (length k) ------------------------------
    g1d = lanczos_kernel_1d(a=a,
                            sigma_px=sigma_px,
                            device=dev,
                            dtype=dt)
    k   = g1d.numel()                           # *actual* kernel length

    # depth-wise weights: (out_ch = D, in_ch = 1, k, 1) and (D,1,1,k)
    gH = g1d.view(1, 1, k, 1).expand(D, 1, k, 1).contiguous()
    gW = g1d.view(1, 1, 1, k).expand(D, 1, 1, k).contiguous()

    cube = cube_spatial.unsqueeze(0)            # (1, D, H, W)

    # first horizontal, then vertical pass
    half = (k - 1) // 2
    cube = F.conv2d(cube, gH, padding=(half, 0), groups=D)
    cube = F.conv2d(cube, gW, padding=(0, half), groups=D)

    return cube.squeeze(0)

def lanczos_kernel_1d(a: int, sigma_px: float, device, dtype):
    """
    Build a 1-D Lanczos window of radius `a` (a = 2 or 3 recommended),
    whose main-lobe standard deviation matches `sigma_px`.

    Returns tensor of shape (2*a+1,)
    """
    # Radius in *fine* pixels that corresponds to the requested σ
    R = sigma_px * math.sqrt(2 * math.pi)
    # Use at most 'a' lobes to keep the kernel compact
    half = int(min(a * R, 3 * a))          # cap width if σ too large

    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    # sinc(x) = sin(pi x)/(pi x)  ;  avoid div-by-zero at x=0
    sinc = torch.where(x == 0,
                       torch.ones_like(x),
                       torch.sin(math.pi * x / R) / (math.pi * x / R))
    lanc = torch.where(x == 0,
                       torch.ones_like(x),
                       torch.sin(math.pi * x / a) / (math.pi * x / a))

    k = sinc * lanc
    k /= k.sum()                           # DC-gain = 1
    return k
    

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
        sigma_px = 0.5 * self.image_upscale       # ≃ 0.5 coarse pixel
        cube_hi  = lanczos_blur_2d(cube_hi, sigma_px)
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