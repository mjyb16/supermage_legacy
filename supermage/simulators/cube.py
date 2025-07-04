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


def generate_meshgrid(grid_extent, gal_res, dtype=torch.float64, device="cuda"):
    """
    Generates a 3D centered meshgrid for simulation.

    Parameters
    ----------
    grid_extent : float
        Half the physical size of the box in each dimension (e.g., r_galaxy).
    gal_res : int
        Number of grid points along each dimension.
    dtype : torch.dtype, optional
        Desired tensor dtype. Defaults to torch.float64.
    device : torch.device or str, optional
        Device for tensor allocation. Defaults to 'cuda'.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        3D meshgrid tensors (X, Y, Z) in physical units, centered on 0.
    """
    # Coordinate range: [-1, 1], then scaled by extent and adjusted for pixel centering
    coords = torch.linspace(-1, 1, gal_res, dtype=dtype, device=device) * 2*grid_extent * (gal_res - 1) / (2*gal_res)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")
    return X, Y, Z

def make_spatial_axis(fov_half, n_out, upscale, device="cuda", dtype=torch.float64):
    """
    Return the 1-D coordinate array for one spatial axis of the *fine* cube.

    Parameters
    ----------
    fov_half   : half–width of the field of view (same units you want back)
    n_out      : number of pixels in the *coarse* image (after pooling)
    upscale    : image_upscale (number of fine pixels per coarse pixel)
    """
    n_hi = n_out * upscale                     # length of fine axis
    dx_hi = 2 * fov_half / (n_hi-1)     # fine-pixel size
    # first coarse-pixel centre is -fov_half + dx_lo/2 = -fov_half + dx_hi*upscale/2
    x_hi = (-fov_half + dx_hi * upscale / 2) + dx_hi * (torch.arange(n_hi,
                               device=device, dtype=dtype) - 0.5 * upscale)
    return x_hi, dx_hi

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

class CubeSimulator(Module):
    """
    Parameters for init
    ----------
    velocity_model : Module
        SuperMAGE velocity field model
    intensity_model : Module
        SuperMAGE intensity/brightness field model
    freqs : int
        Final (downsampled) number of pixels along velocity dimension.
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
    def __init__(self, velocity_model, intensity_model, freqs, systemic_or_redshift, frequency_upscale, cube_fov_half, image_res_out, image_upscale, radius_res, line="co21", device = "cuda", dtype = torch.float64):
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
        #self.flux = Param("flux", None)

        # Determine whether we want systemic velocity or redshift
        self.systemic_or_redshift = systemic_or_redshift
        self.line = line
        
        # Internal resolutions
        self.freq_res_out = len(freqs)
        self.frequency_res = self.freq_res_out * frequency_upscale
        self.image_res    = image_res_out * image_upscale
        self.frequency_upscale = frequency_upscale
        self.image_upscale    = image_upscale

        # Image grid        
        self.pixelscale_pc = cube_fov_half*2/(image_res_out*image_upscale)
        x_hi, dx_hi = make_spatial_axis(cube_fov_half, image_res_out, image_upscale,
                                device=self.device, dtype=self.dtype)
        coords = torch.meshgrid(x_hi, x_hi, x_hi, indexing="ij")
        self.img_x, self.img_y, self.img_z = coords

        # Frequency grid
        self.freqs = freqs
        self.v_z = torch.zeros_like(self.img_z, device = self.device)
        #freq_first = freqs[0]
        #freq_last = freqs[-1]
        
        #self.freqs_upsampled = torch.linspace(self.freqs[0], self.freqs[-1], self.frequency_res, device = self.device, dtype = self.dtype)
        df = (self.freqs[-1] - self.freqs[0]) / (self.frequency_res - 1)
        #self.freqs_upsampled = self.freqs[0] + df * (torch.arange(self.frequency_res, device=self.device, dtype=self.dtype) - (0.5*self.frequency_upscale))
        self.freqs_upsampled = torch.linspace(self.freqs[0], self.freqs[-1], self.frequency_res, device=self.device, dtype=self.dtype)
        
        cube_z_labels = self.freqs_upsampled * torch.ones((self.image_res, self.image_res, self.frequency_res), device = self.device, dtype = self.dtype)
        self.cube_z_l_keops = LazyTensor(cube_z_labels.unsqueeze(-1).expand(self.image_res, self.image_res, self.frequency_res, 1)[:, :, :, None, :])

        # Output resolutions
        # Freq_res_out defined in internal resolutions
        self.image_res_out = image_res_out
        #self.dv = (velocity_max_pc - velocity_min_pc) / self.velocity_res_out
        #Calculate dv in the forward pass of visibility_cube model
        self.dx = self.dy = 2*cube_fov_half/image_res_out
        
        # Constants
        self.pi = torch.tensor(pi, device = self.device)

        self.radius_res = radius_res

    @forward
    def forward(
        self,
        inclination=None, sky_rot=None, line_broadening=None, velocity_shift = None, 
    ):
        rot_x, rot_y, rot_z = DoRotation(self.img_x, self.img_y, self.img_z, inclination, sky_rot, device = self.device)

        # cylindrical radii in the **galaxy** frame
        R_map = torch.sqrt(rot_x**2 + rot_y**2)
        Rmin = 0.1*self.pixelscale_pc
        Rmax = R_map.max()

        R_grid = torch.logspace(np.log10(Rmin), np.log10(Rmax), self.radius_res,
                        device=self.device, dtype=self.dtype)
        
        # one call to the heavy integrator
        v_grid = self.velocity_model.velocity(R_grid, soft=Rmin)   # (NR,)

        v_abs = interpolate_velocity(R_grid, R_map, v_grid)
        # --------------------------------------------
        
        theta_rot = torch.atan2(rot_y, rot_x)
        v_x = -v_abs * torch.sin(theta_rot)
        v_y = v_abs * torch.cos(theta_rot)

        v_x_r, v_y_r, v_z_r = DoRotationT(v_x, v_y, self.v_z, inclination, sky_rot, device = self.device)
        v_los_keops = LazyTensor(v_z_r.unsqueeze(-1).expand(self.image_res, self.image_res, self.image_res, 1)[:, :, None, :, :])

        source_img_cube = self.intensity_model.brightness(rot_x, rot_y, rot_z)

        intensity_cube = source_img_cube.unsqueeze(-1)
        sig_sq = line_broadening**2
        if self.systemic_or_redshift == "systemic":
            velocity_labels_unshifted, _ = freq_to_vel_absolute_torch(self.freqs_upsampled, self.line, device = self.device, dtype = self.dtype)
            upsampled_vs = velocity_labels_unshifted - velocity_shift
            upsampled_vs_shaped = upsampled_vs.unsqueeze(0).unsqueeze(0)
            
            # Use average pooling to downsample
            downsampled_vs = F.avg_pool1d(
                upsampled_vs_shaped,
                kernel_size=self.frequency_upscale,
                stride=self.frequency_upscale
            )
            
            # Shape => (frequency_res_out,)
            downsampled_vs = downsampled_vs.squeeze(0).squeeze(0)
            print(downsampled_vs)
            velocity_labels_unshifted, _ = freq_to_vel_absolute_torch(self.cube_z_l_keops, self.line, device = self.device, dtype = self.dtype) 
            velocity_labels = velocity_labels_unshifted - velocity_shift
        elif self.systemic_or_redshift == "redshift":
            print("Need to implement redshift")
            return
        else:
            print("Please specify 'redshift' or 'systemic'")
            return
        kde_dist = (-0.5*(velocity_labels - v_los_keops)**2 / sig_sq).exp()
        cube = (kde_dist @ intensity_cube) * (1/torch.sqrt(2*self.pi*sig_sq))
        cube_final = torch.squeeze(cube)

        cube_final_3D = cube_final.permute(2, 0, 1)  # (frequency_res, image_res, image_res)

        # Expand to (N=1, C=1, D, H, W)
        cube_5d = cube_final_3D.unsqueeze(0).unsqueeze(0)

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
        # Shape => (1, 1, frequency_res_out, image_res_out, image_res_out)
        cube_downsampled = cube_downsampled.squeeze(0).squeeze(0)

        #raw_sum = cube_downsampled.sum().item()  # just the sum of array entries
        #voxel_area = self.dx * self.dy * self.dv  # for line emission integrated in 3D
        #flux_measured = raw_sum * voxel_area
        #scale_factor = flux / flux_measured
        #cube_downsampled *= scale_factor
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