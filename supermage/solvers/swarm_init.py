import torch
from torch.quasirandom import SobolEngine

# ──────────────────────────────────────────────────────────────
# 1. Sobol' sampler for uniform priors
# ──────────────────────────────────────────────────────────────
def sobol_sample(param_bounds,           # list/tuple/array of shape (D, 2)
                 n_samples, 
                 *, 
                 dtype=torch.float64,
                 device="cpu",
                 scramble=True,
                 seed=None):
    """
    Draw `n_samples` Sobol' points within hyper‑rectangular bounds.
    
    Parameters
    ----------
    param_bounds : sequence[(low, high), …] or (D, 2) tensor
        Uniform prior box for each parameter.
    n_samples    : int
        Number of Sobol' points to generate.
    dtype, device, scramble, seed : usual Torch options.
    
    Returns
    -------
    samples : Tensor[n_samples, D]  in the requested dtype/device
    """
    bounds = torch.as_tensor(param_bounds, dtype=dtype, device=device)
    low, high = bounds[:, 0], bounds[:, 1]
    engine = SobolEngine(dimension=len(bounds), scramble=scramble, seed=seed)
    u = engine.draw(n_samples).to(dtype=dtype, device=device)           # ∈ [0,1)
    return low + (high - low) * u                                       # rescale


# ──────────────────────────────────────────────────────────────
# 2. Serial (memory‑friendly) Sobol' swarm wrapper
# ──────────────────────────────────────────────────────────────
def sobol_swarm_opt(lm_fn,                    # your lm_direct
                    param_bounds, 
                    n_particles,
                    *,
                    dtype      = torch.float64,
                    device     = "cpu",
                    lm_kwargs  = None,        # extra kwargs to lm_fn
                    verbose    = True):
    """
    Runs `lm_fn` once for each Sobol‑initialised particle.
    Only one particle lives in memory at any moment.
    
    Returns
    -------
    best_X   : Tensor[D]      – parameters of the best run
    best_val : float or Tensor (scalar) – objective value (e.g. χ²) of the best run
    history  : list of dicts  – log of every particle
    """
    if lm_kwargs is None:
        lm_kwargs = {}

    # 2‑a. Generate Sobol' particles
    particles = sobol_sample(param_bounds, n_particles, 
                             dtype=dtype, device=device, seed = 16)
    
    best_val = torch.tensor(float("inf"), dtype=dtype, device=device)
    best_X   = None
    best_L   = None
    history  = []

    # 2‑b. Serial optimisation loop
    for i, x0 in enumerate(particles, 1):
        # Important: clone so lm_fn can modify the tensor safely
        result = lm_fn(x0.clone(), **lm_kwargs)
        X_opt, L_opt, chi2_opt = result                  # your lm_direct returns these

        # Log & book‑keeping
        history.append(dict(idx=i, X=X_opt.detach().cpu(), 
                            chi2=float(chi2_opt), L=float(L_opt)))
        if chi2_opt < best_val:
            best_val = chi2_opt.detach()
            best_X   = X_opt.detach().clone()
            best_L   = L_opt

        if verbose:
            print(f"[{i:>3}/{n_particles}] χ²={chi2_opt.item():.4g}   "
                  f"best={best_val.item():.4g}")

    return best_X, best_val, best_L, history