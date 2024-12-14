import torch
from caustics import Module, forward, Param

#MODIFY MODIFY MODIFY!!!!!!!

class LensedCubeSimulator(Module):
    def __init__(self, cube_simulator, lens_model):
        super().__init__()
        self.cube_simulator = cube_simulator
        self.lens_model = lens_model

        self.z_background = Param("z_background", None)
        self.r_galaxy = Param("r_galaxy", None)
        self.fov = Param("fov", None)
        self.npix = Param("npix", None)

    @forward
    def forward(
        self, img_x, img_y, img_z,
        z_background=None, r_galaxy=None, fov=None, npix=None
    ):
        cube_unlensed = self.cube_simulator.forward(img_x, img_y, img_z)
        lensed_result = self.lens_model.forward(
            cube_unlensed,
            z_background=z_background,
            r_galaxy=r_galaxy,
            gal_res=self.cube_simulator.gal_res.value,
            velocity_res=self.cube_simulator.velocity_res.value,
            fov=fov,
            npix=npix
        )
        return lensed_result