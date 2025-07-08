import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d
from supermage.utils.coord_utils import e_radius, pixel_size_background
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy.nddata import Cutout2D
from scipy import signal
from scipy import ndimage
from scipy.ndimage import uniform_filter
from scipy.ndimage import rotate
from typing import Tuple, Union


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

def freq_to_vel_systemic_torch(freq, systemic_velocity, line = "co21", device = "cuda", dtype = torch.float64):
    """
    Return velocity offsets (in km/s) relative to a systemic velocity (in km/s)
    given an array of frequencies (in GHz).
    """
    # Speed of light in km/s
    c = torch.tensor(const.c.value, dtype = dtype, device = device)/1e3
    # Rest frequency of the CO(2-1) line in Hz
    co21_rest_freq = torch.tensor(230.538, dtype = dtype, device = device)
    if line == "co21":
        blueshifted_co21_freq = co21_rest_freq * (1 - systemic_velocity / c)
        velocities = c * (1 - freq / co21_rest_freq) - systemic_velocity
        return velocities, blueshifted_co21_freq

def freq_to_vel_absolute_torch(freq, line="co21", device="cuda", dtype = torch.float64):
    c = torch.tensor(const.c.value, dtype=dtype, device=device) / 1e3
    co21_rest_freq = torch.tensor(230.538, dtype=dtype, device=device)
    velocities = -c * (freq - co21_rest_freq) / co21_rest_freq
    return velocities, co21_rest_freq

def velocity_map_torch(cube, velocities):      
    # Calculate intensity-weighted average velocity
    vel_map = torch.sum(cube * velocities[None, None, :], dim=2) / torch.sum(cube, dim=2)
    
    return vel_map

def velocity_map(cube, velocities):    
    
    # Calculate intensity-weighted average velocity
    vel_map = np.sum(cube * velocities[None, None, :], axis=2) / np.sum(cube, axis=2)
    
    return vel_map

def create_pvd(rotated_cube, slice_start, slice_end):
    """
    Rotated cube: shape (n_minor_axis, n_major_axis, n_freq)
    """
    return np.flip(np.rot90(rotated_cube[slice_start:slice_end, :, :].sum(axis = 0)))

def rotate_spectral_cube_center_offset_arcsec(
    cube_in: np.ndarray,
    angle_deg: float,
    center_offset_arcsec: Tuple[float, float] = (0.0, 0.0),
    pixel_scale: float = 1.0,
    pad_mode: str = "constant",
    pad_cval: Union[int, float] = 0.0,
    interp_order: int = 3,
):
    """
    Rotate a spectral cube around a point specified as an *arcsecond offset*
    from the cube’s geometric centre.

    Parameters
    ----------
    cube : ndarray, shape (ny, nx, nchan)
        Spectral cube (channel-first).
    angle_deg : float
        Counter-clockwise rotation angle (degrees).
    center_offset_arcsec : (dx, dy)
        Offset from the cube centre in arcseconds:
            dx > 0 → right,  dy > 0 → up.
        Fractions allowed.
    pixel_scale : float
        Arcseconds per pixel (or any unit per pixel),
        used for the offset conversion *and* to report the new extent.
    pad_mode / pad_cval / interp_order
        Passed through to `np.y0.item()pad` and `scipy.ndimage.rotate`.

    Returns
    -------
    rotated_cube : ndarray
        Padded & rotated cube.
    extent : ((x_min, x_max), (y_min, y_max))
        Spatial extent in the same physical units as `pixel_scale`.
        The rotation point is at (0, 0).
    """
    ny, nx, n_chan = cube_in.shape

    # ------------------------------------------------------------------
    # 1. Convert arcsecond offset → pixel offset
    # ------------------------------------------------------------------
    dx_arcsec, dy_arcsec = center_offset_arcsec
    dx_pix = dx_arcsec / pixel_scale
    dy_pix = dy_arcsec / pixel_scale

    # geometric centre of the original image
    cx_orig = (nx - 1) / 2.0
    cy_orig = (ny - 1) / 2.0

    # absolute pixel coordinates of the rotation point
    x0 = cx_orig + dx_pix
    y0 = cy_orig + dy_pix

    # ------------------------------------------------------------------
    # 2. Pad so that (x0, y0) becomes the image centre
    # ------------------------------------------------------------------
    left   = x0
    right  = nx - 1 - x0
    top    = y0
    bottom = ny - 1 - y0

    half_width  = int(np.ceil(max(left, right )))
    half_height = int(np.ceil(max(top , bottom)))

    nx_pad = 2 * half_width  + 1
    ny_pad = 2 * half_height + 1

    pad_left   = half_width  - int(np.floor(left))
    pad_right  = half_width  - int(np.floor(right))
    pad_top    = half_height - int(np.floor(top))
    pad_bottom = half_height - int(np.floor(bottom))

    pad_width = (
        (0, 0),                      # spectral axis
        (pad_top, pad_bottom),       # y
        (pad_left, pad_right),       # x
    )

    cube = np.moveaxis(cube_in, -1, 0)

    cube_padded = np.pad(
        cube, pad_width, mode=pad_mode, constant_values=pad_cval
    )

    # ------------------------------------------------------------------
    # 3. Rotate every channel about the new centre
    # ------------------------------------------------------------------
    rotated_cube = np.empty_like(cube_padded)
    for k in range(n_chan):
        rotated_cube[k] = ndimage.rotate(
            cube_padded[k],
            angle_deg,
            reshape=False,
            order=interp_order,
            mode=pad_mode,
            cval=pad_cval,
        )

    # ------------------------------------------------------------------
    # 4. Compute physical extent
    # ------------------------------------------------------------------
    cx_new = (nx_pad - 1) / 2.0
    cy_new = (ny_pad - 1) / 2.0
    x_min = -(cx_new) * pixel_scale
    x_max = +(nx_pad - 1 - cx_new) * pixel_scale
    y_min = -(cy_new) * pixel_scale
    y_max = +(ny_pad - 1 - cy_new) * pixel_scale
    dx = (x_max - x_min)/rotated_cube.shape[2] 
    dy = (y_max - y_min)/rotated_cube.shape[1]
    extent = (x_min - dx/2, x_max+dx/2, y_min - dy/2, y_max + dy/2)
    #extent = (x_min, x_max, y_min, y_max)
    
    return np.moveaxis(rotated_cube, 0, -1), extent

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

def make_elliptical_gaussian_kernel(bmaj_arcsec, bmin_arcsec, bpa_deg, pixel_scale_arcsec, size_factor=6.0, dtype = torch.float32):
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

    return torch.tensor(kernel, dtype=dtype)

def make_elliptical_gaussian_kernel_compatible(
    xpixels, ypixels, beamSize, pixel_scale_arcsec=1.0, cent=None, size_factor=None, dtype=np.float64
):
    """
    Mimics makebeam(), returning a trimmed elliptical Gaussian PSF based on beam size.
    Returns a NumPy array (not Torch tensor).
    """

    # Default center
    if cent is None:
        cent = [xpixels / 2, ypixels / 2]

    # Handle beamSize input
    beamSize = np.array(beamSize, dtype=np.float64)
    try:
        if len(beamSize) == 2:
            beamSize = np.append(beamSize, 0)
        if beamSize[1] > beamSize[0]:
            beamSize[1], beamSize[0] = beamSize[0], beamSize[1]
        if beamSize[2] >= 180:
            beamSize[2] -= 180
    except:
        beamSize = np.array([beamSize, beamSize, 0], dtype=np.float64)

    bmaj, bmin, bpa = beamSize

    # Convert FWHM to sigma in pixels
    st_dev = beamSize[:2] / pixel_scale_arcsec / 2.355
    sigma_x, sigma_y = st_dev

    # Directional factor (to match makebeam logic)
    rot_rad = np.radians(bpa)
    dirfac = np.sign(np.tan(rot_rad)) if np.tan(rot_rad) != 0 else 1.0

    # Create coordinate grid
    y, x = np.indices((int(ypixels), int(xpixels)), dtype=dtype)
    x -= cent[0]
    y -= cent[1]

    # Quadratic form coefficients
    a = (np.cos(rot_rad) ** 2) / (2 * sigma_y**2) + (np.sin(rot_rad) ** 2) / (2 * sigma_x**2)
    b = (dirfac * (np.sin(2 * rot_rad) ** 2)) / (4 * sigma_y**2) - (dirfac * (np.sin(2 * rot_rad) ** 2)) / (4 * sigma_x**2)
    c = (np.sin(rot_rad) ** 2) / (2 * sigma_y**2) + (np.cos(rot_rad) ** 2) / (2 * sigma_x**2)

    psf = np.exp(-1 * (a * x ** 2 - 2 * b * x * y + c * y ** 2))

    # Threshold
    psf[psf < 1e-5] = 0.0

    # Determine cut direction based on PA
    if 45 < bpa < 135:
        flat = np.sum(psf, axis=1)
    else:
        flat = np.sum(psf, axis=0)

    idx = np.where(flat > 0)[0]
    newsize = idx[-1] - idx[0]

    # Ensure odd-sized trimmed kernel
    if newsize % 2 == 0:
        newsize += 1
    else:
        newsize += 2

    # Don't exceed full size
    min_size = min(xpixels, ypixels)
    if newsize > min_size:
        newsize = min_size - 1 if min_size % 2 == 0 else min_size

    # Trim using Cutout2D
    trimmed_psf = Cutout2D(psf, (cent[1], cent[0]), newsize).data

    return trimmed_psf

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
