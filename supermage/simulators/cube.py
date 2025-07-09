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

def make_spatial_axis(fov_half, n_out, upscale, device="cuda", dtype=torch.float64):
    n_hi = n_out * upscale
    dx_hi = 2 * fov_half / (n_hi - 1)
    center_index = (n_hi - 1) / 2
    x_hi = dx_hi * (torch.arange(n_hi, device=device, dtype=dtype) - center_index)
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


class CubeSimulator(Module):
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

class CubeSimulatorSlabbed(Module):
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
        self.x_hi, dx_hi = make_spatial_axis(cube_fov_half, image_res_out, image_upscale,
                                device=self.device, dtype=self.dtype)

        # Frequency grid
        self.freqs = freqs
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
        slab_size = 16  # Can be tuned (e.g., 16, 32, or 64)
    
        image_res = self.image_res
        N = image_res
        x_hi = self.freqs.new_tensor(self.x_hi)
    
        # Precompute 2D spatial grid (img_x, img_y), shared across slabs
        x = x_hi.view(-1, 1, 1)
        y = x_hi.view(1, -1, 1)
    
        # Frequency grid: keep 1D form and broadcast via KeOps later
        velocity_labels_unshifted, _ = freq_to_vel_absolute_torch(
            self.freqs_upsampled, self.line, device=self.device, dtype=self.dtype
        )
        velocity_labels = velocity_labels_unshifted - velocity_shift
        velocity_labels = velocity_labels.view(1, 1, -1, 1)  # [1, 1, F, 1]
    
        # Precompute normalization constant
        sig_sq = line_broadening ** 2
        norm_factor = 1 / torch.sqrt(2 * math.pi * sig_sq)
    
        cube_accumulator = []
    
        for z0 in range(0, N, slab_size):
            z1 = min(z0 + slab_size, N)
            z = x_hi[z0:z1].view(1, 1, -1)  # Only current z slab
    
            # Create slab grids: shape [N, N, slab_size]
            img_x, img_y, img_z = torch.broadcast_tensors(x, y, z)
    
            # Rotate coordinates
            rot_x, rot_y, rot_z = DoRotation(img_x, img_y, img_z, inclination, sky_rot, device=self.device)
    
            # Intensity field
            source_intensity_cube = self.intensity_model.brightness(rot_x, rot_y, rot_z)
            intensity_slab = source_intensity_cube.unsqueeze(-1)  # [N, N, slab, 1]
            del source_intensity_cube, img_x, img_y, img_z, rot_z
    
            # Velocity field
            v_abs = self.velocity_model.velocity(rot_x, rot_y, None)  # [N, N, slab]
            theta_rot = torch.atan2(rot_y, rot_x)
            del rot_x, rot_y
    
            v_x = -v_abs * torch.sin(theta_rot)
            v_y =  v_abs * torch.cos(theta_rot)
            del v_abs, theta_rot
    
            v_z = torch.zeros_like(v_x)  # shape [N, N, slab]
            v_x_r, v_y_r, v_los_r = DoRotationT(v_x, v_y, v_z, inclination, sky_rot, device=self.device)
            del v_x, v_y, v_z
    

            # Reshape so slab axis is nj (KeOps "j" dimension)
            # Reshape frequency axis as j-dimension (summation over slab)
            velocity_labels = velocity_labels.view(1, 1, 1, F, 1)  # [1,1,1,F,1]
            v_los_keops = LazyTensor(v_los_r[:, :, :, None, None])  # [N,N,slab,1,1]
            
            # Do not create pdm! Instead, directly form the kernel-weighted contraction:
            intensity_slab = intensity_slab.requires_grad_()  # if needed
            intensity_kt = LazyTensor(intensity_slab[:, :, :, None, :])  # [N,N,slab,1,1]
            
            # Now apply kernel contraction manually using KeOps
            cube_slab = (
                (-0.5 * (v_los_keops - velocity_labels) ** 2 / sig_sq).exp()
                * intensity_kt
            ).sum(dim=2) * norm_factor  # sum over LOS axis (slab)
            del v_los_r, v_los_keops, intensity_slab, pdm
    
            cube_accumulator.append(cube_slab)
    
        # Concatenate slabs along LOS axis
        cube_full = torch.cat(cube_accumulator, dim=2)  # [N, N, frequency_res]
    
        # Rearrange axes to match expected shape: (F, H, W)
        cube_final_3D = cube_full.permute(2, 0, 1)
    
        # Downsample via average pooling
        cube_5d = cube_final_3D.unsqueeze(0).unsqueeze(0)
        cube_downsampled = F.avg_pool3d(
            cube_5d,
            kernel_size=(self.frequency_upscale, self.image_upscale, self.image_upscale),
            stride=(self.frequency_upscale, self.image_upscale, self.image_upscale)
        ).squeeze(0).squeeze(0)  # Final shape: [F_out, H_out, W_out]
    
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