import torch
from caskade import Module, forward, Param
import numpy as np

#MODIFY MODIFY MODIFY!!!!!!!

class VisibilityCubeSimulator(Module):
    def __init__(self, cube_simulator, data_u, data_v, paduv, freqs, dish_diameter=12):
        super().__init__()
        self.cube_simulator = cube_simulator
        self.data_u = data_u
        self.data_v = data_v
        self.paduv = paduv
        self.freqs = freqs
        self.dish_diameter = dish_diameter

    @forward
    def forward(self, img_x, img_y, img_z, npix):
        cube = self.cube_simulator.forward(img_x, img_y, img_z)

        padding = int((npix - cube.shape[0]) / 2)
        cube_prefft = torch.tensor(
            np.pad(cube.cpu().numpy(), pad_width=((padding, padding), (padding, padding), (0, 0))),
            device=img_x.device
        )

        vis = forward_lensed_vis(
            cube_prefft, 
            self.data_u, 
            self.data_v, 
            False, 
            self.paduv, 
            self.freqs,
            self.dish_diameter, 
            device=img_x.device, 
            plot=False
        ).cpu().numpy()

        return vis