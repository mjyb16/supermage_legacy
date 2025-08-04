import math, torch, numpy as np

def log_like_gaussian(theta, Y_obs, forward_func, Cinv):
    """
    theta is a (D,) torch Tensor.
    The forward model `forward_flat` must return a tensor with same
    shape & device as Y_obs.
    """
    fY   = forward_func(theta)
    dY   = Y_obs - fY
    chi2 = (dY.square() * Cinv).sum()
    return -0.5 * chi2

def log_prior_tophat(theta, low, high):
    """Flat prior inside the box, –inf outside (works on a single θ)."""
    device = low.device
    dtype = low.dtype
    in_box = (theta >= low).all() & (theta <= high).all()
    return torch.tensor(0., device = device, dtype = dtype) if in_box else torch.tensor(-torch.inf, device = device, dtype = dtype)

def mala(
    log_prob_fn,                                # returns scalar torch.Tensor
    init,                                       # (n_chains, D) numpy OR torch
    n_steps=2_000,
    step_size=3e-1,
    mass_matrix=None,                           # None → identity
    progress=True,
):
    x       = init
    dtype = x.dtype
    device = x.device
    n_chains, D = x.shape
    I       = torch.eye(D, device=device, dtype=dtype)
    M_inv   = I if mass_matrix is None else torch.as_tensor(
                  np.linalg.inv(mass_matrix), dtype=dtype, device=device)
    chol    = torch.linalg.cholesky(M_inv)      # lower‑triangular

    samples = torch.zeros((n_steps, n_chains, D),
                          dtype=dtype, device=device)
    acc_mask= torch.zeros((n_steps, n_chains), dtype=torch.bool, device=device)

    pbar = range(n_steps)
    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(pbar, desc="MALA")

    log_p = torch.stack([log_prob_fn(xi) for xi in x])     # (n_chains,)

    for t in pbar:
        # ---- forward & grad --------------------------------------------
        x.requires_grad_(True)
        grads = torch.stack([torch.autograd.grad(log_prob_fn(xi), xi)[0] for xi in x])
    
        noise = (chol @ torch.randn(n_chains, D, device=device, dtype=dtype).mT).mT
        prop  = x + 0.5 * step_size**2 * (grads @ M_inv) + step_size * noise
    
        log_p_prop = torch.stack([log_prob_fn(xi) for xi in prop])
        log_alpha  = log_p_prop - log_p
        accept     = torch.log(torch.rand_like(log_alpha)) < log_alpha
    
        # ---- state update (must be out of autograd) ---------------------
        with torch.no_grad():
            x[accept]     = prop[accept]
            log_p[accept] = log_p_prop[accept]
        x = x.detach()            # prepare a fresh leaf for the next step
    
        samples[t]  = x
        acc_mask[t] = accept

        if progress:
            pbar.set_postfix(acc_rate=float(acc_mask[:t+1].float().mean()))

    return samples.cpu().numpy(), acc_mask.cpu().numpy()