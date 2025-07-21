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

def gaussian_pb(diameter=12, freq=432058061289.4426, shape=(500, 500), deltal=0.004, device='cpu', dtype = torch.float64):
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq
    fwhm = 1.02 * wavelength / diameter * (180 / torch.pi) * (3600)
    half_fov = deltal * shape[0] / 2

    # Grid for PB
    x = torch.linspace(-half_fov, half_fov, shape[0], device=device, dtype = dtype)
    y = torch.linspace(-half_fov, half_fov, shape[1], device=device, dtype = dtype)
    x, y = torch.meshgrid(x, y, indexing='xy')

    # just the exponent part 
    std = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0, device=device))))
    r2 = x**2 + y**2
    pb  = torch.exp(-0.5 * r2 / std**2)

    return pb/pb.max(), fwhm

def casa_airy_beam(l,m,freq_chan,dish_diameter, blockage_diameter, ipower, max_rad_1GHz, n_sample=10000, device = "cpu"):
    """
    Airy disk function for the primary beam as implemented by CASA
    Credits: Noé Dia et al.
    Parameters
    ----------
    l: float, radians
        Coordinate of a point on the image plane (the synthesis projected ascension and declination).
    m: float, radians
        Coordinate of a point on the image plane (the synthesis projected ascension and declination).
    freq_chan: float, Hz
        Frequency.
    dish_diameter: float, meters
        The diameter of the dish.
    blockage_diameter: float, meters
        The central blockage of the dish.
    ipower: int
        ipower = 1 single dish response.
        ipower = 2 baseline response for identical dishes.
    max_rad_1GHz: float, radians
        The max radius from which to sample scaled to 1 GHz.
        This value can be found in sirius_data.dish_models_1d.airy_disk.
        For example the Alma dish model (sirius_data.dish_models_1d.airy_disk import alma)
        is alma = {'func': 'airy', 'dish_diam': 10.7, 'blockage_diam': 0.75, 'max_rad_1GHz': 0.03113667385557884}.
    n_sample=10000
        The sampling used in CASA for PB math.
    Returns
    -------
    val : float
        The dish response.
    """
    casa_twiddle = (180*7.016*c.value)/((np.pi**2)*(10**9)*1.566*24.5) # 0.9998277835716939

    r_max = max_rad_1GHz/(freq_chan/10**9)
    # print(r_max)
    k = (2*np.pi*freq_chan)/c
    aperture = dish_diameter/2

    if n_sample is not None:
        r = np.sqrt(l**2 + m**2)
        r_inc = ((r_max)/(n_sample-1))
        r = (int(r/r_inc)*r_inc)*aperture*k #Int rounding instead of r = (int(np.floor(r/r_inc + 0.5))*r_inc)*aperture*k
        r = r*casa_twiddle
    else:
        r = np.arcsin(np.sqrt(l**2 + m**2)*k*aperture)
        
    if (r != 0):
        if blockage_diameter==0.0:
            return torch.tensor((2.0*j1(r)/r)**ipower).to(device)
        else:
            area_ratio = (dish_diameter/blockage_diameter)**2
            length_ratio = (dish_diameter/blockage_diameter)
            return torch.tensor(((area_ratio * 2.0 * j1(r)/r   - 2.0 * j1(r * length_ratio)/(r * length_ratio) )/(area_ratio - 1.0))**ipower).to(device)
    else:
        return 1
