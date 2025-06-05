import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d
from supermage.utils.coord_utils import e_radius, pixel_size_background
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy import signal
from scipy import ndimage
from scipy.ndimage import uniform_filter
from scipy.ndimage import rotate


def dirty_cube_tool(vis_bin_re_cube, vis_bin_imag_cube, roi_start, roi_end):
    # Define the region of interest for the cube (pixels 1000 to 1050)
    roi_start, roi_end = 225, 276
    num_frequencies = vis_bin_re_cube.shape[0]  # Total number of frequencies
    
    # Initialize an empty list to store the dirty images
    dirty_cube = []
    
    # Loop over each frequency slice to create the dirty image for each
    for i in range(num_frequencies):
        # Create the complex visibility data for the current frequency slice
        combined_vis = vis_bin_re_cube[i] + 1j * vis_bin_imag_cube[i]
        
        # Perform the inverse FFT to get the dirty image in the image plane
        dirty_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(combined_vis), norm = "backward"))
        
        # Take the real part (intensity map) and restrict to the region of interest
        dirty_image_roi = np.abs(dirty_image)[roi_start:roi_end, roi_start:roi_end]
        
        # Append the region of interest for this frequency to the dirty cube
        dirty_cube.append(dirty_image_roi)
    
    # Stack all frequency slices to form a 3D array (dirty cube)
    dirty_cube = np.stack(dirty_cube, axis=-1)
    return dirty_cube

# Eric's mask making code
def smooth_mask(cube, sigma = 2, hann = 5, clip = 0.0002):  # updated by Eric
    """
    Apply a Gaussian blur, using sigma = 4 in the velocity direction (seems to work best), to the uncorrected cube.
    The mode 'nearest' seems to give the best results.
    :return: (ndarray) mask to apply to the un-clipped cube
    """
    smooth_cube = uniform_filter(cube, size=[sigma, sigma, 0], mode='constant')
    Hann_window=signal.windows.hann(hann)
    smooth_cube=signal.convolve(smooth_cube,Hann_window[np.newaxis,np.newaxis,:],mode="same")/np.sum(Hann_window)
    print("RMS of the smoothed cube in mJy/beam:",np.sqrt(np.nanmean(smooth_cube[0]**2))*1e3)
    mask=(smooth_cube > clip)
    mask_iter = mask.T # deliberately make them the same variable, convenient for updating

    print('final mask sum',np.sum(mask))
    return mask_iter.T

def freq_to_vel_systemic(freq, systemic_velocity, line = "co21"):
    """
    Return velocity offsets (in km/s) relative to a systemic velocity (in km/s)
    given an array of frequencies (in GHz).
    """
    # Speed of light in km/s
    c = const.c.value/1e3
    # Rest frequency of the CO(2-1) line in Hz
    co21_rest_freq = 230.538
    if line == "co21":
        blueshifted_co21_freq = co21_rest_freq * (1 - systemic_velocity / c)
        velocities = c * (1 - freq / co21_rest_freq) - systemic_velocity
        return velocities, blueshifted_co21_freq

def freq_to_vel_systemic_torch(freq, systemic_velocity, line = "co21", device = "cuda"):
    """
    Return velocity offsets (in km/s) relative to a systemic velocity (in km/s)
    given an array of frequencies (in GHz).
    """
    # Speed of light in km/s
    c = torch.tensor(const.c.value, dtype = torch.float64, device = device)/1e3
    # Rest frequency of the CO(2-1) line in Hz
    co21_rest_freq = torch.tensor(230.538, dtype = torch.float64, device = device)
    if line == "co21":
        blueshifted_co21_freq = co21_rest_freq * (1 - systemic_velocity / c)
        velocities = c * (1 - freq / co21_rest_freq) - systemic_velocity
        return velocities, blueshifted_co21_freq

def freq_to_vel_absolute_torch(freq, line="co21", device="cuda"):
    c = torch.tensor(const.c.value, dtype=torch.float64, device=device) / 1e3
    co21_rest_freq = torch.tensor(230.538, dtype=torch.float64, device=device)
    velocities = c * (1 - freq / co21_rest_freq)
    return velocities, co21_rest_freq

def velocity_map_torch(cube, velocities):      
    # Calculate intensity-weighted average velocity
    vel_map = torch.sum(cube * velocities[None, None, :], dim=2) / torch.sum(cube, dim=2)
    
    return vel_map

def velocity_map(cube, velocities):    
    
    # Calculate intensity-weighted average velocity
    vel_map = np.sum(cube * velocities[None, None, :], axis=2) / np.sum(cube, axis=2)
    
    return vel_map

def rotate_spectral_cube(cube, angle):
    """
    Rotate a spectral image cube by a specific angle.
    
    Parameters:
    cube (numpy.ndarray): The spectral image cube with shape (channels, height, width)
    angle (float): The rotation angle in degrees
    
    Returns:
    numpy.ndarray: The rotated spectral image cube
    """
    # Get the dimensions of the cube
    channels, height, width = cube.shape
    
    # Create an empty array to store the rotated cube
    rotated_cube = np.zeros_like(cube)
    
    # Rotate each channel
    for i in range(channels):
        rotated_cube[i] = ndimage.rotate(cube[i], angle, reshape=False, mode='wrap', cval=0.0)
    
    return rotated_cube

def make_elliptical_gaussian_kernel(bmaj_arcsec, bmin_arcsec, bpa_deg, pixel_scale_arcsec, size_factor=6.0):
    """
    Create 2D elliptical Gaussian kernel using NumPy, returns a torch tensor.
    """
    # Convert FWHM to sigma in pixel units
    sigma_x = bmaj_arcsec / pixel_scale_arcsec / 2.355
    sigma_y = bmin_arcsec / pixel_scale_arcsec / 2.355
    theta = np.deg2rad(bpa_deg)

    # Determine kernel size (odd)
    size_x = int(size_factor * sigma_x) | 1
    size_y = int(size_factor * sigma_y) | 1
    x = np.arange(size_x) - size_x // 2
    y = np.arange(size_y) - size_y // 2
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Rotate coordinates
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    # 2D elliptical Gaussian
    kernel = np.exp(-0.5 * ((X_rot / sigma_x) ** 2 + (Y_rot / sigma_y) ** 2))
    kernel /= np.sum(kernel)

    return torch.tensor(kernel, dtype=torch.float32)

def convolve_cube_with_beam(cube, kernel):
    """
    Convolve spatial dimensions of a 3D cube with the given 2D kernel.
    cube: torch.Tensor of shape [H, W, V]
    kernel: 2D torch.Tensor
    Returns a cube of the same shape.
    """
    H, W, V = cube.shape
    cube = cube.permute(2, 0, 1).unsqueeze(1)  # [V, 1, H, W]
    kernel = kernel.to(cube.device).unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
    
    # Ensure symmetric padding
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2

    cube_conv = F.conv2d(cube, kernel, padding=(pad_h, pad_w), groups=1)
    return cube_conv.squeeze(1).permute(1, 2, 0)  # Back to [H, W, V]
