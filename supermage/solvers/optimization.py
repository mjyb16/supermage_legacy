import torch
import numpy as np
from torch.func import jvp, vjp
from torch.func import jacrev, jacfwd  # (PyTorch ≥2.0; for older versions import from functorch)
from torch.autograd import grad as torch_grad


def cg_solve_static(hvp, b, *, maxiter=50, tol=1e-6):
    """
    Conjugate-gradient that is *vmap-safe*:
    * fixed number of Python iterations (maxiter),
    * no data-dependent `break`,
    * no `.item()` or Tensor→bool conversion.

    Args
    ----
    hvp : callable(v) -> H·v              Hessian-vector product closure
    b   : (N,) Tensor                     right-hand side
    tol : float                           residual L2 tolerance (used for masking)
    """
    x  = torch.zeros_like(b)
    r  = b.clone()
    p  = r.clone()
    rs = torch.dot(r, r).unsqueeze(0)     # shape (1,) so we can `where`

    for _ in range(maxiter):
        Ap    = hvp(p)
        alpha = rs / (torch.dot(p, Ap).clamp_min(1e-12)).unsqueeze(0)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsn   = torch.dot(r_new, r_new).unsqueeze(0)

        # stop updating once residual < tol
        active = (torch.sqrt(rsn) >= tol).float()     # 0. or 1.
        beta   = torch.where(active.bool(), rsn / rs, torch.zeros_like(rsn))

        # in-place masked updates
        x  = torch.where(active.bool(), x_new, x)
        p  = torch.where(active.bool(), r_new + beta * p, p)
        r  = torch.where(active.bool(), r_new, r)
        rs = torch.where(active.bool(), rsn, rs)

    return x

def cg_solve(hvp, b, x0=None, tol=1e-6, maxiter=20):
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    r = b - hvp(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    for i in range(maxiter):
        Ap = hvp(p)
        alpha = rs_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x


def lm_cg_autograd_stable(
    X, Y, f, n_chi,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,
    L_min=1e-9, L_max=1e9,
    stopping=1e-4,
    verbose=True,
):
    if C is None:
        Cinv = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv = 1.0 / C
    else:
        Cinv = torch.linalg.inv(C)

    # Use torch.autograd.grad for explicit gradient calculation
    from torch.autograd import grad as torch_grad

    def get_loss_and_grad(x_in):
        x_ = x_in.detach().requires_grad_(True)
        fX = f(x_)
        dY = Y - fX
        chi2 = (dY**2 * Cinv).sum()
        loss = 0.5 * chi2
        grad = torch_grad(loss, x_, allow_unused=True)[0]
        return chi2.detach(), loss.detach(), grad

    def hvp(v, current_x, current_L):
        x_ = current_x.detach().requires_grad_(True)
        fX = f(x_)
        dY = Y - fX
        loss = 0.5 * (dY**2 * Cinv).sum()
        grad_L = torch_grad(loss, x_, create_graph=True, allow_unused=True)[0]
        hvp_L = torch_grad((grad_L * v).sum(), x_, retain_graph=True, allow_unused=True)[0]
        return hvp_L + current_L * v

    chi2, _, grad = get_loss_and_grad(X)

    if verbose:
        print(f"{'Iter':>4} | {'chi2/n':>12} | {'chi2_new/n':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>5}")
        print("-"*65)

    for it in range(max_iter):
        hvp_for_cg = lambda v: hvp(v, X, L)
        h = cg_solve(hvp_for_cg, -grad, maxiter=cg_maxiter, tol=cg_tol)

        chi2_new, _, _ = get_loss_and_grad(X + h)

        actual_reduction = chi2 - chi2_new
        pred_reduction = -torch.dot(h, grad) - 0.5 * torch.dot(h, hvp(h, X, 0.0))
        
        # --- START: THIS IS THE FIX ---
        # A valid step should result in a positive predicted reduction.
        # If not, the quadratic approximation is poor, and the step should be rejected.
        if pred_reduction > 0:
            rho = actual_reduction / pred_reduction
        else:
            # Force rejection of the step if the model predicts an increase in chi-squared
            rho = torch.tensor(-1.0) 
        # --- END: THIS IS THE FIX ---

        accepted = (rho >= epsilon)
        if accepted:
            X, chi2 = X + h, chi2_new
            L = max(L / L_dn, L_min)
            _, _, grad = get_loss_and_grad(X) # Recalculate gradient at the new point
        else:
            L = min(L * L_up, L_max)

        if verbose:
            # Use chi2/n_chi for the current accepted value in the printout
            current_chi2_val = chi2 if accepted else get_loss_and_grad(X)[0]
            print(f"{it:4d} | {current_chi2_val.item()/n_chi:12.4e} | {chi2_new.item()/n_chi:12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(bool(accepted)):>5}")

        if torch.norm(h) < stopping:
            if verbose: print("Stopping: Step size below threshold.")
            break
            
        if torch.isnan(rho) or rho.item() < -100: # break on divergence
            if verbose: print("Stopping: Rho is NaN or diverging.")
            break

    return X, L, chi2

def lm_cg_autograd(
    X, Y, f, n_chi,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,
    L_min=1e-9, L_max=1e9,
    stopping=1e-4,
    verbose=True,
):
    if C is None:
        Cinv = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv = 1.0 / C
    else:
        Cinv = torch.linalg.inv(C)

    def get_loss_and_grad(x_in):
        x_ = x_in.detach().requires_grad_(True)
        fX = f(x_)
        dY = Y - fX
        chi2 = (dY**2 * Cinv).sum()
        loss = 0.5 * chi2
        grad = torch_grad(loss, x_, allow_unused=True)[0]
        return chi2.detach(), loss.detach(), grad

    def hvp(v, current_x, current_L):
        x_ = current_x.detach().requires_grad_(True)
        fX = f(x_)
        dY = Y - fX
        loss = 0.5 * (dY**2 * Cinv).sum()
        grad_L = torch_grad(loss, x_, create_graph=True, allow_unused=True)[0]
        hvp_L = torch_grad((grad_L * v).sum(), x_, retain_graph=True, allow_unused=True)[0]
        return hvp_L + current_L * v

    chi2, _, grad = get_loss_and_grad(X)

    if verbose:
        print(f"{'Iter':>4} | {'chi2/n':>12} | {'chi2_new/n':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>5}")
        print("-"*65)

    for it in range(max_iter):
        hvp_for_cg = lambda v: hvp(v, X, L)
        h = cg_solve(hvp_for_cg, -grad, maxiter=cg_maxiter, tol=cg_tol)

        chi2_new, _, _ = get_loss_and_grad(X + h)

        actual_reduction = chi2 - chi2_new
        pred_reduction = -torch.dot(h, grad) - 0.5 * torch.dot(h, hvp(h, X, 0.0))
        rho = actual_reduction / (pred_reduction + 1e-12)

        accepted = (rho >= epsilon)
        if accepted:
            X, chi2 = X + h, chi2_new
            L = max(L / L_dn, L_min)
            _, _, grad = get_loss_and_grad(X)
        else:
            L = min(L * L_up, L_max)

        if verbose:
            print(f"{it:4d} | {chi2.item()/n_chi:12.4e} | {chi2_new.item()/n_chi:12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(bool(accepted)):>5}")

        if torch.norm(h) < stopping:
            break
            
        # --- THIS IS THE CORRECTED LINE ---
        if torch.isnan(rho) or rho.item() < -100: # break on divergence
            break

    return X, L, chi2

def lm_cg_annealed(
    X, Y, f, n_chi,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,
    L_min=1e-9, L_max=1e9,
    stopping=1e-4,
    verbose=True,
    # --- annealing extras ---
    T_start=10.0, T_end=1.0, n_temps=5,            # temperature schedule
    noise_scale=0.0,                               # step noise amplitude at T_start; scaled ∝ T
    schedule="exp",                                # "exp" or "linear"
):
    """
    Deterministic annealing Levenberg–Marquardt with CG inner solves.

    Tempering: replace Cinv by Cinv/T, so χ²_T = (dY^2 * Cinv/T).sum().
    As T ↓ 1 we recover the true objective.

    noise_scale: std of Gaussian noise added to step h at T=T_start, scaled linearly to 0 at T=1.
    """

    # ---------- prep ----------
    if C is None:
        Cinv_full = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv_full = 1.0 / C
    else:
        Cinv_full = torch.linalg.inv(C)

    # temperature ladder
    if n_temps < 1:
        raise ValueError("n_temps must be ≥ 1")
    if schedule == "exp":
        Ts = torch.logspace(torch.log10(torch.tensor(T_start)),
                            torch.log10(torch.tensor(T_end)),
                            steps=n_temps, device=X.device, dtype=X.dtype)
    elif schedule == "linear":
        Ts = torch.linspace(T_start, T_end, steps=n_temps,
                            device=X.device, dtype=X.dtype)
    else:
        raise ValueError("schedule must be 'exp' or 'linear'")

    def lm_stage(X_init, Cinv_scaled, T, stage_idx):
        nonlocal L  # keep damping across stages (optional; comment to reset per stage)

        def chi2_and_grad(x):
            fY  = f(x)
            dY  = Y - fY
            chi2= (dY**2 * Cinv_scaled).sum()
            _, vjp_fn = vjp(f, x)
            grad = vjp_fn(Cinv_scaled * dY)[0]
            return chi2, grad, dY

        def hvp(v):
            _, jvp_out = jvp(f, (X_stage,), (v,))
            w = Cinv_scaled * jvp_out
            _, vjp_fn = vjp(f, X_stage)
            return vjp_fn(w)[0] + L * v

        X_stage = X_init
        chi2, grad, dY = chi2_and_grad(X_stage)

        if verbose:
            hdr = f"Stage {stage_idx+1}/{n_temps}  T={T:.3g}"
            print(hdr)
            print(f"{'Iter':>4} | {'chi2':>12} | {'chi2_new':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>4}")

        for it in range(max_iter):
            # solve (H+λI)h = g
            h = cg_solve(hvp, grad, maxiter=cg_maxiter, tol=cg_tol)

            # add annealing noise (decays with T)
            if noise_scale > 0.0 and T > 1.0:
                sigma = noise_scale * (T - 1.0) / (T_start - 1.0)
                h = h + sigma * torch.randn_like(h)

            chi2_new, _, _ = chi2_and_grad(X_stage + h)
            expected = torch.dot(h, hvp(h) + grad)
            rho = (chi2 - chi2_new) / torch.abs(expected)

            accepted = (rho >= epsilon)
            if accepted:
                X_stage, chi2 = X_stage + h, chi2_new
                L = max(L / L_dn, L_min)
            else:
                L = min(L * L_up, L_max)

            if verbose:
                print(f"{it:4d} | {chi2.item()*T/n_chi:12.4e} | {chi2_new.item()*T/n_chi:12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(accepted):>4}")

            if torch.norm(h) < stopping:
                break

            # refresh grad at new point (or old if rejected)
            _, grad, _ = chi2_and_grad(X_stage)

        return X_stage, chi2

    # ---------- run ladder ----------
    X_curr, chi2_curr = X, None
    for k, T in enumerate(Ts):
        Cinv_T = Cinv_full / T
        X_curr, chi2_curr = lm_stage(X_curr, Cinv_T, float(T), k)

    return X_curr, L, chi2_curr

def lm_cg_og(
    X, Y, f, n_chi,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,
    L_min=1e-9, L_max=1e9,
    stopping=1e-4,
    verbose=True,
):
    """
    Levenberg-Marquardt with conjugate-gradient step that is *vmap-friendly*:
    – fixed iteration count (no data-dependent breaks)
    – masking with torch.where for all updates
    – no Python branching on tensor values
    """

    # ---- constants as tensors so we can broadcast in torch.where ----
    X       = X.clone()                       # ensure we can write in-place
    dtype   = X.dtype
    device  = X.device
    L       = torch.tensor(L,      dtype=dtype, device=device)
    L_dn_t  = torch.tensor(L_dn,   dtype=dtype, device=device)
    L_up_t  = torch.tensor(L_up,   dtype=dtype, device=device)
    L_min_t = torch.tensor(L_min,  dtype=dtype, device=device)
    L_max_t = torch.tensor(L_max,  dtype=dtype, device=device)
    eps_t   = torch.tensor(epsilon,dtype=dtype, device=device)

    # ---- prepare inverse covariance --------------------------------
    if C is None:
        Cinv = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv = 1.0 / C
    else:
        Cinv = torch.linalg.inv(C)

    # ---- helpers ----------------------------------------------------
    def chi2_and_grad(x):
        fY          = f(x)
        dY          = Y - fY
        chi2_val    = (dY**2 * Cinv).sum()
        _, vjp_fun  = vjp(f, x)
        grad_val    = vjp_fun(Cinv * dY)[0]
        return chi2_val, grad_val

    def hvp(v):
        _, jvp_out = jvp(f, (X,), (v,))
        w          = Cinv * jvp_out
        _, vjp_fun = vjp(f, X)
        return vjp_fun(w)[0] + L * v

    # ---- initial chi² / grad ---------------------------------------
    chi2, grad = chi2_and_grad(X)

    # ---- optional header print -------------------------------------
    if verbose:
        print(f"{'Iter':>4} | {'chi2':>12} | {'chi2_new':>12} | {'λ':>8} | {'ρ':>6} | acc")

    # ---- main LM loop (static length) ------------------------------
    for it in range(max_iter):

        # CG step (vmap-safe implementation you already added)
        h = cg_solve_static(hvp, grad, maxiter=cg_maxiter, tol=cg_tol)

        # proposed new parameters and stats
        chi2_new, grad_new = chi2_and_grad(X + h)
        expected           = torch.dot(h, hvp(h) + grad)
        rho                = (chi2 - chi2_new) / expected.abs()

        accepted           = (rho >= eps_t)

        # masked parameter / state updates
        X    = torch.where(accepted, X + h, X)
        chi2 = torch.where(accepted, chi2_new, chi2)
        grad = torch.where(accepted, grad_new, grad)

        L = torch.where(
            accepted,
            torch.clamp(L / L_dn_t, min=L_min_t),
            torch.clamp(L * L_up_t, max=L_max_t),
        )

        # ---- progress print (only if prints are enabled) -------
        # if verbose:
        #     chi_disp  = (chi2 / n_chi).item()
        #     chi_n_disp= (chi2_new / n_chi).item()
        #     print(f"{it:4d} | {chi_disp:12.4e} | {chi_n_disp:12.4e} | "
        #             f"{L.item():8.2e} | {rho.item():6.3f} | {bool(accepted.item())}")

        # stopping rule (masked; keeps loop static)
        converged   = (torch.norm(h) < stopping)
        # once converged, we could early-exit all subsequent
        # computation with a mask; here we just continue to keep
        # the control-flow static and cheap.

    return X, L, chi2


def lm_direct_trust_region(
    X, Y, f,
    n_chi,
    C=None,
    L      = 1.0,              # initial damping
    max_iter = 50,
    L_min   = 1e-9, L_max = 1e9,
    stopping= 1e-8,
    verbose = True,
):
    """
    Dense (direct‑solve) Levenberg–Marquardt with
    Nocedal‑Wright style λ adaptation (no ε, L_dn, L_up needed).
    Returns
    -------
    X*   : optimised parameters
    L*   : final damping
    chi2 : final χ²
    """
    # -----------------------------------------------------------
    # House‑keeping
    # -----------------------------------------------------------
    X = X.clone()                           # avoid in‑place edits
    if C is None:
        Cinv, is_diag = torch.ones_like(Y), True
    elif C.ndim == 1:
        Cinv, is_diag = 1.0 / C, True
    else:                                   # full covariance
        Cinv, is_diag = torch.linalg.inv(C), False

    def forward_residual(x):
        fY = f(x)
        return fY, Y - fY                   # f(x), residual

    def chi2_from_residual(dY):
        return (dY**2 * Cinv).sum() if is_diag else dY @ Cinv @ dY

    fY, dY = forward_residual(X)
    chi2   = chi2_from_residual(dY)

    if verbose:
        print(f"{'Iter':>4} | {'χ²':>12} | {'χ²_new':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>4}")
        print("-"*60)

    Din  = X.numel()
    eye  = torch.eye(Din, device=X.device, dtype=X.dtype)
    nu   = 2.0                              # growth factor for λ updates

    # -----------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------
    for it in range(max_iter):
        # Jacobian J  (shape: Dout × Din)
        J = jacfwd(f)(X).reshape(-1, Din)

        # Gradient g and Gauss–Newton Hessian H
        if is_diag:
            w_dY = Cinv * dY
            grad = J.T @ w_dY
            H    = J.T @ (J * Cinv.view(-1, 1))
        else:
            w_dY = Cinv @ dY
            grad = J.T @ w_dY
            H    = J.T @ Cinv @ J

        # Solve (H + λI) h = g
        H_damped = H + L * eye
        h        = torch.linalg.solve(H_damped, grad).squeeze(-1)

        # Candidate step
        fY_new, dY_new = forward_residual(X + h)
        chi2_new       = chi2_from_residual(dY_new)

        # ρ = (actual gain) / (predicted gain)
        expected = h @ (H_damped @ h + grad)
        rho      = (chi2 - chi2_new) / expected.abs() if expected.abs() > 1e-32 else X.new_zeros(())

        # ---------- trust‑region λ update -----------------------
        if rho > 0:                          # step accepted
            X, chi2, fY, dY = X + h, chi2_new, fY_new, dY_new
            L  = torch.clamp(L * max(1/3.0, 1 - (2*rho - 1)**3), L_min, L_max)
            nu = 2.0                         # reset growth
            accepted = True
        else:                                # step rejected
            L  = torch.clamp(L * nu, L_min, L_max)
            nu = nu * 2.0                    # exponential growth
            accepted = False
        # -------------------------------------------------------

        if verbose:
            print(f"{it:4d} | {chi2.item()/n_chi:12.4e} | {chi2_new.item()/n_chi:12.4e} "
                  f"| {L:8.2e} | {rho.item():6.3f} | {accepted}")

        # stopping criteria
        if h.norm() < stopping or L >= L_max:
            break

    return X, L, chi2


def lm_direct(
    X, Y, f,
    n_chi,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,   # cg_* kept for signature compatibility (unused)
    L_min=1e-9, L_max=1e9,
    stopping=1e-8,
    verbose=True,
):
    """
    Dense (direct solve) Levenberg–Marquardt matching lm_cg signature but without CG.

    Parameters are identical to lm_cg; cg_maxiter & cg_tol are ignored.

    Returns
    -------
    X      : optimised parameter vector
    L      : final damping value
    chi2   : final chi^2 (scalar tensor)
    """

    # Clone to avoid in-place modification of caller's tensor
    X = X.clone()

    # ------------------------------------------------------------------
    # Prepare inverse covariance / weights
    # ------------------------------------------------------------------
    if C is None:
        # Diagonal weights = 1
        Cinv = torch.ones_like(Y)
        is_diag = True
    elif C.ndim == 1:
        Cinv = 1.0 / C
        is_diag = True
    else:
        Cinv = torch.linalg.inv(C)
        is_diag = False

    def forward_residual(x):
        fY = f(x)
        dY = Y - fY
        return fY, dY

    def chi2_from_residual(dY):
        if is_diag:
            return (dY**2 * Cinv).sum()
        else:
            return (dY @ Cinv @ dY)

    # Initial χ²
    fY, dY = forward_residual(X)
    chi2 = chi2_from_residual(dY)

    if verbose:
        print(f"{'Iter':>4} | {'chi2':>12} | {'chi2_new':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>4}")
        print("-"*60)

    Din = X.numel()
    eye = torch.eye(Din, device=X.device, dtype=X.dtype)

    for it in range(max_iter):
        torch.cuda.empty_cache()
        # --------------------------------------------------------------
        # Jacobian J : (Dout, Din)
        # --------------------------------------------------------------
        # jacfwd returns shape of output + shape of input -> (Dout, Din)
        J = jacfwd(f)(X)
        if J.ndim != 2:
            J = J.reshape(-1, Din)  # flatten any structured output just in case
        Dout = J.shape[0]

        # --------------------------------------------------------------
        # Build RHS (called 'grad' in your code) = J^T W dY
        # --------------------------------------------------------------
        if is_diag:
            w_dY = Cinv * dY           # (Dout,)
            grad = J.T @ w_dY          # (Din,)
            # Hessian (Gauss–Newton) H = J^T diag(Cinv) J
            # (Multiply each row of J by sqrt weights, or by weights then J^T)
            # Use broadcasting for efficiency:
            H = J.T @ (J * Cinv.view(-1, 1))
        else:
            w_dY = Cinv @ dY           # (Dout,)
            grad = J.T @ w_dY
            H = J.T @ Cinv @ J         # (Din, Din)

        # Damped system: (H + L I) h = grad
        H_damped = H + L * eye

        # Solve
        h = torch.linalg.solve(H_damped, grad)

        if h.ndim > 1:  # ensure vector
            h = h.squeeze(-1)

        # --------------------------------------------------------------
        # Candidate update
        # --------------------------------------------------------------
        fY_new, dY_new = forward_residual(X + h)
        chi2_new = chi2_from_residual(dY_new)

        # Expected improvement (match cg version): h^T[(H + L I)h + grad]
        expected = h @ (H_damped @ h + grad)
        if expected.abs() < 1e-32:
            rho = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        else:
            rho = (chi2 - chi2_new) / expected.abs()

        accepted = (rho >= epsilon)
        if accepted:
            X = X + h
            chi2 = chi2_new
            L = max(L / L_dn, L_min)
            # Recompute residual for next iteration (lazy update okay)
            fY, dY = fY_new, dY_new
        else:
            L = min(L * L_up, L_max)

        if verbose:
            i = 2 if Din > 2 else 0
            H_ii = H[i, i].item()
            #print(f"param[{i}]: grad={grad[i].item():.3e}, H_ii={H_ii:.3e}, L={L:.3e}")
            print(f"{it:4d} | {chi2.item()/n_chi:12.4e} | {chi2_new.item()/n_chi:12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(bool(accepted)):>4} ")

        # Stopping criterion
        if torch.norm(h) < stopping:
            break
        if L >= L_max:
            break

    return X, L, chi2


def adam_optimizer(
    X_init, Y, f,
    C=None,
    max_iter=300,
    lr=1e-2,
    T_start=None,
    T_end=None,
    n_chi=None,          # ignored for now, kept for compatibility
    verbose=True,
):
    X = X_init.clone().detach().requires_grad_(True)

    # Prepare inverse covariance (assume diagonal or full)
    if C is None:
        Cinv = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv = 1.0 / C
    else:
        Cinv = torch.linalg.inv(C)

    # Gaussian χ² function
    def chi2_fn(X):
        fY = f(X)
        dY = Y - fY
        return (dY**2 * Cinv).sum()

    # Adam optimizer setup
    optimizer = torch.optim.Adam([X], lr=lr)

    # Cosine annealing temperature schedule (for optional noise injection)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=lr * 0.1)

    for it in range(max_iter):
        optimizer.zero_grad()
        chi2 = chi2_fn(X)
        chi2.backward()

        if torch.isnan(X.grad).any() or torch.isnan(chi2):
            print("NaNs encountered, stopping early.")
            break

        optimizer.step()
        scheduler.step()

        # Optional: temperature annealing for simulated annealing behavior
        if T_start is not None and T_end is not None:
            T = T_end + 0.5 * (T_start - T_end) * (1 + torch.cos(torch.tensor(it / max_iter * 3.14159)))
            noise = torch.randn_like(X) * T.sqrt() * 1e-4
            with torch.no_grad():
                X.add_(noise)

        if verbose and it%10==0:
            print(f"{it:4d} | chi² = {chi2.item()/n_chi:.4e} | M_bh : {X[2].item():.4f}")
            print(X.detach())

    return X.detach(), None, chi2_fn(X.detach())
