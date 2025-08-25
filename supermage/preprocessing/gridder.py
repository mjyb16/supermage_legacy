import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, Callable, Sequence, Union
from numpy.typing import ArrayLike
from scipy.special import i0  # Modified Bessel I0
try:
    from scipy.special import pro_ang1
    _HAVE_PRO_ANG1 = True
except ImportError:
    _HAVE_PRO_ANG1 = False


# Defining some window functions. We could add more in the future but their effect needs to be taken into account in the forward model. 

# ------------------------------------------------------------------
#  Kaiser–Bessel window
# ------------------------------------------------------------------
def kaiser_bessel_window(u: ArrayLike,
                         center: float,
                         *,
                         pixel_size: float = 0.015,
                         m: int = 6,
                         beta: Optional[float] = None,
                         normalize: bool = True) -> np.ndarray:
    """
    1D Kaiser–Bessel interpolation window (separable in u, v).
    
    Parameters
    ----------
    u : coordinates (array)
    center : float
        Grid-cell center coordinate (same units as u).
    pixel_size : float
        Grid pixel size in uv units (arcsec, wavelengths, etc).
    m : int
        *Total* support width in pixel units (i.e. covers m * pixel_size).
        Effective half-width = 0.5 * m * pixel_size.
    beta : float, optional
        Shape parameter. If None, a heuristic is used (oversampling≈2 assumed):
            beta ≈ π * sqrt( (m/2)^2 - 0.8 )
        (Beatty / Fessler style). Increase beta -> steeper taper.
    normalize : bool
        Normalize so that window(center) == 1.

    Returns
    -------
    w : ndarray (same shape as u)
    """
    half_width = 0.5 * m * pixel_size
    dist = np.abs(u - center)

    w = np.zeros_like(u, dtype=float)
    mask = dist <= half_width
    if not np.any(mask):
        return w

    if beta is None:
        # Simple default good for oversamp ~2
        beta = np.pi * np.sqrt((m / 2.0)**2 - 0.8)

    # Normalized distance in [-1,1]
    t = dist[mask] / half_width   # ∈ [0,1]
    # KB functional form
    arg = beta * np.sqrt(1 - t**2)
    denom = i0(beta)
    w_vals = i0(arg) / denom
    if normalize:
        # Already normalized to 1 at t=0; still, guard numerically.
        w_vals /= w_vals.max()
    w[mask] = w_vals
    return w


# ---- Schwab 1984 (CASA-style) spheroidal rational approximation coefficients ----
_SPH_A = np.array([0.01624782, -0.05350728, 0.1464354, -0.2347118,
                   0.2180684, -0.09858686, 0.01466325], dtype=float)  # a6..a0
_SPH_B1 = 0.2177793  # denominator linear coeff

def _spheroid_scalar(eta: float) -> float:
    """Scalar version (mirrors your original), returns large number if |eta|>1."""
    if abs(eta) > 1.0:
        return 0.0  # you had 1e30 then multiplied by (1-eta^2)-> negative huge; better just 0
    n = eta*eta - 1.0
    # Horner evaluation for numerator in descending powers of n (a6 n^6 + ... + a0)
    num = (((((( _SPH_A[0]*n + _SPH_A[1])*n + _SPH_A[2])*n + _SPH_A[3])*n + _SPH_A[4])*n + _SPH_A[5])*n + _SPH_A[6])
    den = _SPH_B1 * n + 1.0
    return num / den

def spheroid_vec(eta):
    """Vectorized spheroidal rational approximation (|eta|<=1)."""
    eta = np.asarray(eta, dtype=float)
    out = np.zeros_like(eta)
    mask = np.abs(eta) <= 1.0
    if np.any(mask):
        n = eta[mask]**2 - 1.0
        # Evaluate numerator via Horner
        a6,a5,a4,a3,a2,a1,a0 = _SPH_A
        num = ((((((a6*n + a5)*n + a4)*n + a3)*n + a2)*n + a1)*n + a0)
        den = _SPH_B1 * n + 1.0
        out[mask] = num / den
    return out

def casa_pswf_window(u,
                     center,
                     *,
                     pixel_size: float = 0.015,
                     m: int = 5,
                     normalize: bool = True):
    """
    CASA/Schwab prolate-spheroidal gridding kernel (separable 1-D form).
    Credit: Ryan Loomis via beams_and_weighting

    Parameters
    ----------
    u : array-like
        Coordinates (same units as `center`).
    center : float
        Grid point center coordinate.
    pixel_size : float
        Pixel spacing in `u` units.
    m : int
        Total *integer* support (number of pixels). Schwab kernel typically uses m=5.
    normalize : bool
        Normalize so that peak value (at center) is 1.

    Returns
    -------
    w : ndarray
        Window weights; zero outside half-width m/2 * pixel_size.
    """
    u = np.asarray(u, dtype=float)
    # Distance in pixels
    dpix = (u - center) / pixel_size
    half_width = m / 2.0  # in pixels
    # Dimensionless eta (|eta|<=1 inside support)
    eta = dpix / half_width
    w = (1.0 - eta**2) * spheroid_vec(eta)
    # Zero outside support
    w[np.abs(eta) > 1.0] = 0.0

    if normalize:
        # Peak at eta=0
        w0 = spheroid_vec(0.0)  # (1 - 0)*spheroid(0)
        if w0 != 0:
            w /= w0
    return w

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

def build_window_LUT(window_fn,
                     m: int,
                     pixel_size: float,
                     n_samples: int = 4096,
                     **window_kwargs):
    """
    Precompute symmetric samples of a window over its support for fast interpolation.

    Returns
    -------
    coords : 1D array of distances (≥0) where window sampled
    values : window(distances) with center at 0
    """
    # We sample distances from 0 .. half_width
    half_width = 0.5 * m * pixel_size
    d = np.linspace(0.0, half_width, n_samples)
    # Evaluate window by calling with u = center + d AND u = center - d (evenness assumed)
    # We'll exploit even symmetry, so just evaluate once.
    vals = window_fn(center + d, center, pixel_size=pixel_size, m=m, **window_kwargs)
    # Force last sample to 0 (or near) for numeric cleanliness
    vals[-1] = 0.0
    return d, vals


def evaluate_from_LUT(dist_array, d_samples, v_samples):
    """
    dist_array : |u - center| distances
    d_samples, v_samples : LUT from build_window_LUT
    Linear interpolation (could switch to np.interp).
    """
    dist_array = np.asarray(dist_array)
    half_width = d_samples[-1]
    out = np.zeros_like(dist_array, dtype=float)
    mask = dist_array <= half_width
    out[mask] = np.interp(dist_array[mask], d_samples, v_samples)
    return out


class LUTWindow:
    """
    Callable object matching the signature window(u, center).
    Wraps a precomputed lookup table for a specific kernel configuration.
    """
    __slots__ = ("pixel_size", "m", "d_samples", "v_samples")

    def __init__(self, pixel_size, m, d_samples, v_samples):
        self.pixel_size = pixel_size
        self.m = m
        self.d_samples = d_samples
        self.v_samples = v_samples

    def __call__(self, u, center):
        u = _ensure_array(u)
        dist = np.abs(u - center)
        return evaluate_from_LUT(dist, self.d_samples, self.v_samples)
def bin_data(u, v, values, weights, bins,
             window_fn: Callable,
             truncation_radius,
             uv_tree: cKDTree,
             grid_tree: cKDTree,
             pairs: Sequence[Sequence[int]],
             statistics_fn="mean",
             verbose=1,
             window_kwargs: Optional[Dict] = None):
    """
    Parameters
    ----------
    window_fn : callable
        Should accept (u_array, center) ONLY (other params captured via partial or LUTWindow).
        Must return non-negative weights, zero outside support.
    window_kwargs : dict, optional
        (Not used if you already froze params via partial/LUTWindow; kept for backwards compat.)
    """
    u_edges, v_edges = bins
    Nu = len(u_edges) - 1
    Nv = len(v_edges) - 1
    grid = np.zeros((Nu, Nv), dtype=float)

    n_coarse = 0
    # Iterate per grid center
    for k, data_indices in enumerate(pairs):
        if not data_indices:
            continue

        u_center, v_center = grid_tree.data[k]
        # 1D separable window
        wu = window_fn(u[data_indices], u_center)
        wv = window_fn(v[data_indices], v_center)
        w = weights[data_indices] * wu * wv

        if w.sum() <= 0:
            continue

        val = values[data_indices]
        i, j = divmod(k, Nv)   # careful with ordering (Nu major?)

        if statistics_fn == "mean":
            grid[j, i] = np.sum(val * w) / np.sum(w)

        elif statistics_fn == "std":
            # Expand adaptively like your original version. Start with given support m=1 concept.
            # We'll mimic your adaptive m logic by gradually enlarging search radius (L1).
            indices = data_indices
            local_w = w
            effective = (local_w > 0).sum()
            expand = 1.0
            while effective < 5:
                expand += 0.1
                indices = uv_tree.query_ball_point([u_center, v_center],
                                                   expand * truncation_radius,
                                                   p=1, workers=6)
                val = values[indices]
                wu = window_fn(u[indices], u_center)
                wv = window_fn(v[indices], v_center)
                local_w = weights[indices] * wu * wv
                effective = (local_w > 0).sum()
            if expand > 1.0:
                n_coarse += 1
            # Effective sample size
            imp = wu * wv
            n_eff = (imp.sum() ** 2) / (np.sum(imp**2) + 1e-12)
            # Weighted variance
            mean_val = np.sum(val * local_w) / np.sum(local_w)
            var = np.sum(local_w * (val - mean_val)**2) / np.sum(local_w)
            # Unbiased-ish scaling with effective n
            grid[j, i] = np.sqrt(var) * np.sqrt(n_eff / (max(n_eff - 1, 1))) * (1 / np.sqrt(n_eff))

        elif statistics_fn == "count":
            grid[j, i] = (w > 0).sum()

        elif callable(statistics_fn):
            grid[j, i] = statistics_fn(val, w)

    if verbose:
        print(f"Number of coarsened pixels: {n_coarse}")
    return grid

def bin_data_old(u, v, values, weights, bins, window_fn, truncation_radius, uv_tree, grid_tree, pairs, statistics_fn="mean", verbose=1):
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