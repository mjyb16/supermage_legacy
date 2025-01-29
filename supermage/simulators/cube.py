import torch
from math import pi
from caskade import Module, forward, Param
from pykeops.torch import LazyTensor
from supermage.utils.math_utils import DoRotation, DoRotationT

class CubeSimulator(Module):
    def __init__(self, velocity_model, intensity_model, img_x, img_y, img_z):
        super().__init__()
        self.velocity_model = velocity_model
        self.intensity_model = intensity_model

        self.inclination = Param("inclination", None)
        self.sky_rot = Param("sky_rot", None)
        self.gal_res = Param("gal_res", None)
        self.velocity_res = Param("velocity_res", None)
        self.velocity_min = Param("velocity_min", None)
        self.velocity_max = Param("velocity_max", None)
        self.line_broadening = Param("line_broadening", None)

        self.img_x = img_x
        self.img_y = img_y
        self.img_z = img_z

    @forward
    def forward(
        self,
        inclination=None, sky_rot=None,
        gal_res=None, velocity_res=None,
        velocity_min=None, velocity_max=None,
        line_broadening=None
    ):
        rot_x, rot_y, rot_z = DoRotation(self.img_x, self.img_y, self.img_z, inclination, sky_rot)
        #print(rot_x)

        v_abs = self.velocity_model.velocity(rot_x, rot_y, rot_z).nan_to_num(posinf=0, neginf=0)
        #print(v_abs)
        
        theta_rot = torch.atan2(rot_y, rot_x)
        v_x = -v_abs * torch.sin(theta_rot)
        v_y = v_abs * torch.cos(theta_rot)

        v_x_r, v_y_r, v_z_r = DoRotationT(v_x, v_y, torch.zeros_like(v_x, device = "cuda"), inclination, sky_rot)
        #print(v_z_r)

        source_img_cube = self.intensity_model.brightness(rot_x, rot_y, rot_z)
        #print(source_img_cube)

        velocity_min_pc = velocity_min / 3.086e16
        velocity_max_pc = velocity_max / 3.086e16

        cube_z_labels = torch.linspace(velocity_min_pc, velocity_max_pc, velocity_res, device = "cuda")
        cube_z_labels = cube_z_labels * torch.ones((gal_res, gal_res, velocity_res), device = "cuda")

        cube_z_l_keops = LazyTensor(cube_z_labels.unsqueeze(-1).expand(gal_res, gal_res, velocity_res, 1)[:, :, :, None, :])
        v_los_keops = LazyTensor(v_z_r.unsqueeze(-1).expand(gal_res, gal_res, gal_res, 1)[:, :, None, :, :])

        intensity_cube = source_img_cube.unsqueeze(-1)
        sig_sq = line_broadening**2
        kde_dist = (-1*(cube_z_l_keops - v_los_keops)**2 / sig_sq).exp()
        cube = (kde_dist @ intensity_cube) * (1/torch.sqrt(2*torch.tensor(pi, device = "cuda")*sig_sq))
        cube_final = torch.squeeze(cube)

        return torch.movedim(cube_final, -1, 0)