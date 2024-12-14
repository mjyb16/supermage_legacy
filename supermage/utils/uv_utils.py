import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools 

c = 299792458 #m/s

def sim_uv_cov(obs_length, z_background, frequency = 1900.537e9):
    # CREDIT: YASHAR HEZAVEH AND CHATGPT.....I didn't write this code :)
    np.random.seed(7)
    
    integration_time = 5.0 / 3600.0  # in hours
    n_samples = int(np.ceil(obs_length / integration_time))
    obs_length = n_samples * integration_time
    
    obs_start_time = (np.random.random() - 0.5) * 5.0
    n_antenna = int(np.ceil(10 + np.random.random() * 50))
    max_baseline_length = 16000  # in meters
    do = -1 * np.random.random() * np.pi / 2.0
    
    ENU = np.vstack([np.random.random((2, n_antenna)) * max_baseline_length, np.zeros((1, n_antenna))])
    
    lat = -23.02 * np.pi / 180.0  # latitude of ALMA
    ENU_to_xyz = np.array([[0, -np.sin(lat), np.cos(lat)],
                           [1, 0, 0],
                           [0, np.cos(lat), np.sin(lat)]])
    
    obs_length = obs_length * 2 * np.pi / 24
    obs_start_time = obs_start_time * 2 * np.pi / 24
    
    HourAngle = np.linspace(obs_start_time, obs_start_time + obs_length, n_samples)
    
    n_baselines = n_antenna * (n_antenna - 1) // 2
    antennas = np.array(list(itertools.combinations(range(1, n_antenna + 1), 2)))
    
    xyz = np.dot(ENU_to_xyz, ENU)
    B = xyz[:, antennas[:, 1] - 1] - xyz[:, antennas[:, 0] - 1]
    
    u = np.zeros((n_samples, n_baselines))
    v = np.zeros((n_samples, n_baselines))
    
    for i in range(len(HourAngle)):
        ho = HourAngle[i]
        
        Bto_uvw = np.array([[np.sin(ho), np.cos(ho), 0],
                            [-np.sin(do) * np.cos(ho), np.sin(do) * np.sin(ho), np.cos(do)],
                            [np.cos(do) * np.cos(ho), -np.cos(do) * np.sin(ho), np.sin(do)]])
        
        uvw = np.dot(Bto_uvw, B)
        u[i, :] = uvw[0, :]
        v[i, :] = uvw[1, :]
    
    UVGRID, _, _ = np.histogram2d(u.flatten(), v.flatten(), bins=(np.linspace(-1000, 1000, 128), np.linspace(-1000, 1000, 128)))
    noise_rms = UVGRID.copy()
    noise_rms[noise_rms == 0] = np.inf
    noise_rms = 1.0 / np.sqrt(noise_rms)
    
    ants1 = np.tile(antennas[:, 0], n_samples)
    ants2 = np.tile(antennas[:, 1], n_samples)
    
    #My part starts here:
    frequency_z = frequency/(1+z_background)
    wavelength = c / frequency_z
    u = u.flatten() / wavelength
    v = v.flatten() / wavelength
    
    return u, v
    
    
def generate_uv_mask(u, v, nyquist = False, shape = (500, 500), pad_uv = 0.01):
    maxuv = np.max((np.abs(u), np.abs(v)))
    mask_nan, edgex, edgey, binnumber = stats.binned_statistic_2d(u, v, values = np.ones(len(u)), \
                                                                  statistic = "max", bins = shape, \
                                                                  range = ((-maxuv-pad_uv*maxuv, maxuv+pad_uv*maxuv), (-maxuv-pad_uv*maxuv, maxuv+pad_uv*maxuv)))
    mask = np.nan_to_num(mask_nan)
    deltau = 2*(maxuv+pad_uv*maxuv) / shape[0]
    deltav = deltau
    deltal = 1/(shape[0]*deltau) * (180/np.pi) * (3600)
    deltam = deltal
    return mask, deltal, deltam

def generate_binned_data(u, v, values, nyquist = False, shape = (600, 600), pad_uv = 0.01):
    maxuv = np.max((np.abs(u), np.abs(v)))
    binned_nan, edgex, edgey, binnumber = stats.binned_statistic_2d(u, v, values = values, \
                                                                  statistic = "mean", bins = shape, \
                                                                  range = ((-maxuv-pad_uv*maxuv, maxuv+pad_uv*maxuv), (-maxuv-pad_uv*maxuv, maxuv+pad_uv*maxuv)))
    binned = np.nan_to_num(binned_nan)
    return binned

def generate_binned_counts(u, v, nyquist = False, shape = (600, 600), pad_uv = 0.01):
    maxuv = np.max((np.abs(u), np.abs(v)))
    binned_nan, edgex, edgey, binnumber = stats.binned_statistic_2d(u, v, values = np.ones(len(u)), \
                                                                  statistic = "sum", bins = shape, \
                                                                  range = ((-maxuv-pad_uv*maxuv, maxuv+pad_uv*maxuv), (-maxuv-pad_uv*maxuv, maxuv+pad_uv*maxuv)))
    binned = np.nan_to_num(binned_nan, nan = 1)
    binned[binned == 0] = 1
    return binned

def binned_uv_range(u, v, values, nyquist = False, shape = (600, 600), pad_uv = 0.01):
    maxuv = np.max((np.abs(u), np.abs(v)))
    return (-maxuv-pad_uv*maxuv, maxuv+pad_uv*maxuv)

def generate_pb(diameter=12, freq=432058061289.4426, shape=(500, 500), deltal=0.004, device='cpu'):
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq
    fwhm = 1.02 * wavelength / diameter * (180 / torch.pi) * (3600)
    half_fov = deltal * shape[0] / 2

    # Grid for PB
    x = torch.linspace(-half_fov, half_fov, shape[0], device=device)
    y = torch.linspace(-half_fov, half_fov, shape[1], device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')

    # Gaussian PB parameters
    mean = torch.tensor([0.0, 0.0], device=device)  # Mean (center) of the Gaussian
    std = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, device=device))))
    covariance_matrix = torch.tensor([[std**2, 0], [0, std**2]], device=device)  # Covariance matrix

    # 2-D Gaussian PB
    x_y = torch.stack([x.ravel(), y.ravel()], dim=1)
    inv_covariance_matrix = torch.inverse(covariance_matrix)
    diff = x_y - mean

    pb = (
        1 / (2 * torch.pi * torch.sqrt(torch.det(covariance_matrix)))
    ) * torch.exp(-0.5 * torch.sum(diff @ inv_covariance_matrix * diff, dim=1))

    # Reshape the PDF values to match the shape of the grid
    pb = pb.view(x.shape)

    return pb, fwhm
