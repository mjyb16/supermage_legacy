import numpy as np
import torch
from torch.nn.functional import avg_pool2d
from supermage.utils.coord_utils import e_radius, pixel_size_background
import matplotlib.pyplot as plt


def velocity_map_torch(cube, velocity_res=600, velocity_min=-2e5, velocity_max=2e5):
    # Convert velocity range from m/s to pc/s
    v_min_pc = velocity_min# / 3.086e16
    v_max_pc = velocity_max# / 3.086e16
    
    # Create velocity array
    velocities = torch.linspace(v_min_pc, v_max_pc, velocity_res, device=cube.device)
    
    # Calculate intensity-weighted average velocity
    vel_map = torch.sum(cube * velocities[None, None, :], dim=2) / torch.sum(cube, dim=2)
    
    return vel_map

def velocity_map(cube, velocity_res = 600, velocity_min = -1.2e3, velocity_max = 1.2e3):
    vel_map = np.dot(cube, np.linspace(velocity_min, velocity_max, velocity_res)) * (velocity_max - velocity_min) / velocity_res / (cube.sum(2))
    return vel_map

