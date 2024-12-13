import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

def grid(pixel_scale, img_size): 
    """
    Given a pixel scale and a number of pixels in image space, grid the associated Fourier space

    Args:
        pixel_scale (float): Pixel resolution in image space (in arcsec)
        img_size (float/int): Size of the image
    
    Returns:
        edges coordinates of the grid in uv space.     
    """

    # Arcsec to radians: 
    dl = (pixel_scale * u.arcsec).to(u.radian).value
    dm = (pixel_scale * u.arcsec).to(u.radian).value

    du = 1 / (img_size * dl) * 1e-3 # klambda
    dv = 1 / (img_size * dm) * 1e-3 # klambda

    u_min = -img_size/2 * du 
    u_max =  img_size/2 * du 

    v_min = -img_size/2 * dv
    v_max =  img_size/2 * dv

    u_edges = np.linspace(u_min, u_max, img_size + 1)
    v_edges = np.linspace(v_min, v_max, img_size + 1)

    return u_edges, v_edges


# Defining some window functions. We could add more in the future but their effect needs to be taken into account in the forward model. 
def pillbox_window(u, center, pixel_size=0.015, m=1):
    """
    u: coordinate of the data points to be aggregated (u or v)
    center: coordinate of the center of the pixel considered. 
    pixel_size: Size of a pixel in the (u,v)-plane, in arcseconds
    m: size of the truncation of this window (in term of pixel_size)
    """
    return np.where(np.abs(u - center) <= m * pixel_size / 2, 1, 0)


def sinc_window(u, center, pixel_size=0.015, m=1):
    """
    u: coordinate of the data points to be aggregated (u or v)
    center: coordinate of the center of the pixel considered. 
    pixel_size: Size of a pixel in the uv-plane, in arcseconds
    m: size of the truncation of this window (in term of pixel_size)
    """
    return np.sinc(np.abs(u - center) / m / pixel_size)


from scipy.spatial import cKDTree
import numpy as np
from typing import Callable

def bin_data(u, v, values, weights, bins, window_fn, truncation_radius, uv_tree, grid_tree, pairs, statistics_fn="mean", verbose=1):
    """
    u: u-coordinate of the data points to be aggregated
    v: v-coordinate of the data points to be aggregated 
    values: value at the different uv coordinates (i.e. visibility)
    weights: weights for the values
    bins: grid edges
    window_fn: Window function for the convolutional gridding
    truncation_radius: Pixel size in uv-plane
    uv_tree: Pre-built cKDTree for UV data points
    grid_tree: Pre-built cKDTree for grid centers
    pairs: Precomputed list of indices within truncation_radius for each grid center
    statistics_fn: Function or method for computing statistics, such as "mean" or "std"
    verbose: Verbose level for debugging
    """
    u_edges, v_edges = bins
    n_coarse = 0
    grid = np.zeros((len(u_edges) - 1, len(v_edges) - 1))

    # Use precomputed pairs for each grid cell center
    for k, data_indices in enumerate(pairs):
        if len(data_indices) > 0:
            # Coordinates and center of the current cell
            u_center, v_center = grid_tree.data[k]
            value = values[data_indices]
            weight = weights[data_indices] * window_fn(u[data_indices], u_center) * window_fn(v[data_indices], v_center)
            
            if weight.sum() > 0:
                i, j = divmod(k, len(v_edges) - 1)
                
                if statistics_fn == "mean":
                    grid[j, i] = (value * weight).sum() / weight.sum()
                
                elif statistics_fn == "std":
                    # Set indices to data_indices initially
                    indices = data_indices
                    
                    # Calculate initial weight with m=1
                    m = 1
                    value = values[indices]
                    weight = weights[indices] * window_fn(u[indices], u_center, m=m) * window_fn(v[indices], v_center, m=m)
                    
                    # Check and adjust for larger neighborhood if needed
                    while (weight > 0).sum() < 5:
                        m += 0.1
                        indices = uv_tree.query_ball_point([u_center, v_center], m * truncation_radius, p=1, workers=6)
                        value = values[indices]
                        weight = weights[indices] * window_fn(u[indices], u_center, m=m) * window_fn(v[indices], v_center, m=m)
                    
                    # Increment n_coarse only if m > 1
                    if m > 1:
                        n_coarse += 1
                    
                    # Calculate effective sample size and apply weighted std calculation
                    importance_weights = window_fn(u[indices], u_center, m=m) * window_fn(v[indices], v_center, m=m)
                    n_eff = np.sum(importance_weights)**2 / np.sum(importance_weights**2)
                    grid[j, i] = np.sqrt(np.cov(value, aweights=weight, ddof=0)) * (n_eff / (n_eff - 1)) * 1 / (np.sqrt(n_eff))
                
                elif statistics_fn == "count":
                    grid[j, i] = (weight > 0).sum()
                
                elif isinstance(statistics_fn, Callable):
                    grid[j, i] = statistics_fn(value, weight)
    
    if verbose:
        print(f"Number of coarsened pixels: {n_coarse}")
    return grid