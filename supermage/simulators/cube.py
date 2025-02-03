import torch
from math import pi
from caskade import Module, forward, Param
from pykeops.torch import LazyTensor
from supermage.utils.math_utils import DoRotation, DoRotationT
import torch.nn.functional as F

def generate_meshgrid(grid_extent, gal_res, device = "cuda"):
    """
    Generates grid for simulation
    grid_extent: 2*r_galaxy usually
    """
    return torch.meshgrid(torch.linspace(-grid_extent, grid_extent, gal_res, device = device), torch.linspace(-grid_extent, grid_extent, gal_res, device = device), torch.linspace(-grid_extent, grid_extent, gal_res, device = device), indexing = "ij")

class CubeSimulator(Module):
    """
    Parameters for init
    ----------
    velocity_model : Module
        SuperMAGE velocity field model
    intensity_model : Module
        SuperMAGE intensity/brightness field model
    velocity_res_out : int
        Final (downsampled) number of pixels along velocity dimension.
    velocity_upscale : int
        Factor by which velocity_res_out is multiplied. High internal resolution needed to prevent aliasing (rec. 5x).
    velocity_min, velocity_max : float
        Velocity range (in m/s).
    cube_fov : float
        Spatial extent of the cube (pc).
    image_res_out : int
        Final (downsampled) number of image pixels (2D) returned for the cube's spatial dimensions.
    image_upscale : int
        Factor by which image_res_out is multiplied. High internal resolution can be needed to prevent aliasing.
    """
    def __init__(self, velocity_model, intensity_model, velocity_res_out, velocity_upscale, velocity_min, velocity_max, cube_fov, image_res_out, image_upscale):
        super().__init__()
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        self.inclination = Param("inclination", None)
        self.sky_rot = Param("sky_rot", None)
        self.line_broadening = Param("line_broadening", None)

        # Internal resolutions
        self.velocity_res = velocity_res_out * velocity_upscale
        self.image_res    = image_res_out * image_upscale
        self.velocity_upscale = velocity_upscale
        self.image_upscale    = image_upscale

        # Image grid        
        meshgrid = generate_meshgrid(cube_fov, self.image_res, device="cuda")
        self.img_x = meshgrid[0]
        self.img_y = meshgrid[1]
        self.img_z = meshgrid[2]

        # Velocity grid
        self.v_z = torch.zeros_like(meshgrid[0], device = "cuda")
        velocity_min_pc = velocity_min / 3.086e16
        velocity_max_pc = velocity_max / 3.086e16
        cube_z_labels = torch.linspace(velocity_min_pc, velocity_max_pc, self.velocity_res, device = "cuda")
        cube_z_labels = cube_z_labels * torch.ones((self.image_res, self.image_res, self.velocity_res), device = "cuda")
        self.cube_z_l_keops = LazyTensor(cube_z_labels.unsqueeze(-1).expand(self.image_res, self.image_res, self.velocity_res, 1)[:, :, :, None, :])

        # Output resolutions
        self.velocity_res_out = velocity_res_out
        self.image_res_out = image_res_out

        # Constants
        self.pi = torch.tensor(pi, device = "cuda")

    @forward
    def forward(
        self,
        inclination=None, sky_rot=None, line_broadening=None
    ):
        rot_x, rot_y, rot_z = DoRotation(self.img_x, self.img_y, self.img_z, inclination, sky_rot)

        v_abs = self.velocity_model.velocity(rot_x, rot_y, rot_z).nan_to_num(posinf=0, neginf=0)
        
        theta_rot = torch.atan2(rot_y, rot_x)
        v_x = -v_abs * torch.sin(theta_rot)
        v_y = v_abs * torch.cos(theta_rot)

        v_x_r, v_y_r, v_z_r = DoRotationT(v_x, v_y, self.v_z, inclination, sky_rot)
        v_los_keops = LazyTensor(v_z_r.unsqueeze(-1).expand(self.image_res, self.image_res, self.image_res, 1)[:, :, None, :, :])

        source_img_cube = self.intensity_model.brightness(rot_x, rot_y, rot_z)

        intensity_cube = source_img_cube.unsqueeze(-1)
        sig_sq = line_broadening**2
        kde_dist = (-1*(self.cube_z_l_keops - v_los_keops)**2 / sig_sq).exp()
        cube = (kde_dist @ intensity_cube) * (1/torch.sqrt(2*self.pi*sig_sq))
        cube_final = torch.squeeze(cube)

        cube_final_3D = cube_final.permute(2, 0, 1)  # (velocity_res, image_res, image_res)

        # Expand to (N=1, C=1, D, H, W)
        cube_5d = cube_final_3D.unsqueeze(0).unsqueeze(0)

        # Compute integer pooling sizes (assumes integral ratios)
        kernel_d = self.velocity_upscale
        kernel_h = self.image_upscale
        kernel_w = self.image_upscale

        # Use average pooling to downsample
        cube_downsampled = F.avg_pool3d(
            cube_5d,
            kernel_size=(kernel_d, kernel_h, kernel_w),
            stride=(kernel_d, kernel_h, kernel_w)
        )
        # Shape => (1, 1, velocity_res_out, image_res_out, image_res_out)
        cube_downsampled = cube_downsampled.squeeze(0).squeeze(0)
        torch.cuda.empty_cache()
        return cube_downsampled