import torch
from math import pi
from caskade import Module, forward, Param
from pykeops.torch import LazyTensor
from supermage.utils.math_utils import DoRotation, DoRotationT

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
    velocity_res : int
        Internal number of pixels for velocity grid (used for non-aliased velocity calculations).
    velocity_res_out : int
        Final (downsampled) number of pixels along velocity dimension.
    velocity_min, velocity_max : float
        Velocity range (in m/s).
    cube_fov : float
        Spatial extent of the cube (pc).
    image_res : int
        Internal 3D number of pixels (x, y, z) used for fine spatial calculations.
    image_res_out : int
        Final (downsampled) number of image pixels (2D) returned for the cube's spatial dimensions.
    """
    def __init__(self, velocity_model, intensity_model, velocity_res, velocity_res_out, velocity_min, velocity_max, cube_fov, image_res, image_res_out):
        super().__init__()
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        self.inclination = Param("inclination", None)
        self.sky_rot = Param("sky_rot", None)
        self.line_broadening = Param("line_broadening", None)

        #Image grid
        self.image_res = image_res
        meshgrid = generate_meshgrid(cube_fov, image_res, device="cuda")
        self.img_x = meshgrid[0]
        self.img_y = meshgrid[1]
        self.img_z = meshgrid[2]

        # Velocity grid
        self.v_z = torch.zeros_like(meshgrid[0], device = "cuda")
        velocity_min_pc = velocity_min / 3.086e16
        velocity_max_pc = velocity_max / 3.086e16
        cube_z_labels = torch.linspace(velocity_min_pc, velocity_max_pc, velocity_res, device = "cuda")
        cube_z_labels = cube_z_labels * torch.ones((image_res, image_res, velocity_res), device = "cuda")
        self.cube_z_l_keops = LazyTensor(cube_z_labels.unsqueeze(-1).expand(image_res, image_res, velocity_res, 1)[:, :, :, None, :])

        #Constant
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

        return torch.movedim(cube_final, -1, 0)

class CubeSimulator_GPT(Module):
    def __init__(
        self,
        velocity_model,
        intensity_model,
        # "Upsampled" internal resolution
        upsample_galaxy,
        upsample_velocity,
        # "Final" output resolution
        galaxy_res,
        velocity_res,
        # Spatial domain extents (e.g. arrays or scalar ranges)
        img_x,  # e.g. a 1D tensor or [xmin, xmax]
        img_y,
        img_z,
        # Velocity domain extents
        velocity_min,
        velocity_max
    ):
        """
        Parameters
        ----------
        velocity_model : Module
            Your velocity field model with a .velocity(x, y, z) method.
        intensity_model : Module
            Your intensity/brightness model with a .brightness(x, y, z) method.
        upsample_galaxy : int
            Internal 3D grid size (x, y, z) used for fine spatial calculations.
        upsample_velocity : int
            Internal velocity grid size used for fine velocity calculations.
        galaxy_res : int
            Final (downsampled) image resolution (2D) returned.
        velocity_res : int
            Final (downsampled) velocity resolution returned.
        img_x, img_y, img_z : array-like or float range
            Spatial domain extents or grids. For example, you could store
            [xmin, xmax] or you already have 1D arrays for each axis.
        velocity_min, velocity_max : float
            Velocity range (in whatever units, e.g. pc/s).
        """

        super().__init__()

        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        # The main "physical" parameters as caskade Params
        self.inclination = Param("inclination", None)
        self.sky_rot = Param("sky_rot", None)
        # We will store the final requested output resolutions as parameters
        self.gal_res = Param("gal_res", galaxy_res)
        self.velocity_res = Param("velocity_res", velocity_res)
        self.velocity_min = Param("velocity_min", velocity_min)
        self.velocity_max = Param("velocity_max", velocity_max)
        self.line_broadening = Param("line_broadening", None)

        # Store the upsample sizes
        self.upsample_galaxy = upsample_galaxy
        self.upsample_velocity = upsample_velocity

        # Keep track of the raw input domain extents (or full 1D coordinate arrays)
        self.img_x = img_x
        self.img_y = img_y
        self.img_z = img_z

        # -------------------------------------------------
        # 1) Generate a high-resolution spatial meshgrid
        #    We'll store these as buffers so they are moved
        #    with the module (e.g., onto GPU).
        # -------------------------------------------------
        # If img_x, img_y, img_z are just [min, max],
        # create a linspace. If they are full 1D grids, adapt as needed.
        x_vals = torch.linspace(img_x[0], img_x[-1], upsample_galaxy)
        y_vals = torch.linspace(img_y[0], img_y[-1], upsample_galaxy)
        z_vals = torch.linspace(img_z[0], img_z[-1], upsample_galaxy)
        X, Y, Z = torch.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        # Register as buffers so they move with .to(device)
        self.register_buffer("Xgrid", X)
        self.register_buffer("Ygrid", Y)
        self.register_buffer("Zgrid", Z)

        # -------------------------------------------------
        # 2) Generate a high-resolution velocity grid
        #    Also stored as a buffer for convenience.
        # -------------------------------------------------
        vel_linspace = torch.linspace(velocity_min, velocity_max, upsample_velocity)
        self.register_buffer("velocity_grid", vel_linspace)

    @forward
    def forward(
        self,
        inclination=None,
        sky_rot=None,
        line_broadening=None,
        # We *could* allow velocity_min, velocity_max, etc. to be adjusted here.
        # But since we stored them as Params, you can also just set them externally.
    ):
        """
        Return:
        -------
        cube : torch.Tensor of shape (velocity_res, galaxy_res, galaxy_res)
            The final downsampled data cube.
        """
        # Unpack the final resolutions & velocity range from the Param
        gal_res = int(self.gal_res)
        velocity_res = int(self.velocity_res)
        velocity_min = self.velocity_min
        velocity_max = self.velocity_max

        # For clarity, rename upsample sizes
        up_gal = self.upsample_galaxy
        up_vel = self.upsample_velocity

        # 1) Rotate the *high-res* spatial grid based on inclination/sky_rot
        rot_x, rot_y, rot_z = DoRotation(
            self.Xgrid, self.Ygrid, self.Zgrid, inclination, sky_rot
        )

        # 2) Compute absolute velocity on that high-res grid
        v_abs = self.velocity_model.velocity(rot_x, rot_y, rot_z).nan_to_num(
            posinf=0, neginf=0
        )

        # 3) Compute components in the plane
        theta_rot = torch.atan2(rot_y, rot_x)
        v_x = -v_abs * torch.sin(theta_rot)
        v_y = v_abs * torch.cos(theta_rot)

        # 4) Rotate those velocity vectors back to get line-of-sight
        v_x_r, v_y_r, v_z_r = DoRotationT(
            v_x, v_y, torch.zeros_like(v_x), inclination, sky_rot
        )

        # 5) Compute intensity model
        source_img_cube = self.intensity_model.brightness(rot_x, rot_y, rot_z)

        # 6) Construct the high-res velocity labels for each spatial pixel
        #    shape => (up_gal, up_gal, up_vel)
        #    We'll broadcast the 1D velocity_grid across x,y.
        velocity_grid_3d = self.velocity_grid.view(1, 1, up_vel).expand(
            up_gal, up_gal, up_vel
        )

        # 7) Prepare for the KeOps kernel
        #    v_los is shape (up_gal, up_gal, up_gal)
        #    We have a "point" dimension for keops
        cube_z_l_keops = LazyTensor(velocity_grid_3d.unsqueeze(-1)[:, :, :, None])
        v_los_keops = LazyTensor(v_z_r.unsqueeze(-1)[:, :, :, None])

        # 8) Evaluate the broadening kernel (Gaussian in velocity)
        intensity_cube = source_img_cube.unsqueeze(-1)  # shape => (up_gal, up_gal, up_gal, 1)
        sig_sq = line_broadening**2

        kde_dist = (-1 * (cube_z_l_keops - v_los_keops) ** 2 / sig_sq).exp()
        # shape of kde_dist => (up_gal, up_gal, up_vel, up_gal)

        # 9) Weighted sum over the "los" dimension => final shape (up_gal, up_gal, up_vel, 1)
        cube_upsampled = (kde_dist @ intensity_cube) * (
            1.0 / torch.sqrt(2 * torch.tensor(pi, device=kde_dist.device) * sig_sq)
        )
        cube_upsampled = cube_upsampled.squeeze(-1)  # (up_gal, up_gal, up_vel)

        # -------------------------------------------------------------
        # 10) Downsample from (up_gal, up_gal, up_vel) to (gal_res, gal_res, velocity_res)
        #     using interpolation in 3D or applying a stepwise approach
        #     (2D + 1D). Here we show a single 3D trilinear interpolation.
        # -------------------------------------------------------------
        # Expand so that interpolate sees (N,C,D,H,W).
        # We'll treat the velocity dimension as "depth" (D),
        # and x,y as H,W.
        cube_upsampled_5d = cube_upsampled.unsqueeze(0).unsqueeze(0)  # (1,1, up_gal, up_gal, up_vel)
        # We want final shape => (gal_res, gal_res, velocity_res).
        # So pass size=(D,H,W) to interpolate, i.e. D=velocity_res, H=gal_res, W=gal_res
        cube_downsampled = F.interpolate(
            cube_upsampled_5d,
            size=(velocity_res, gal_res, gal_res),
            mode="trilinear",  # or "area" / "nearest" / ...
            align_corners=False
        )
        # Back to shape => (velocity_res, gal_res, gal_res)
        cube_final = cube_downsampled.squeeze(0).squeeze(0)

        return cube_final