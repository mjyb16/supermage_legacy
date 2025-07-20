import torch
from torch.func import jvp, vjp
from torch.func import jacrev, jacfwd  # (PyTorch ≥2.0; for older versions import from functorch)

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

def lm_cg(
    X, Y, f,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,
    L_min=1e-9, L_max=1e9,
    stopping=1e-4,
    verbose=True,               # <-- new flag
):
    # prepare Cinv
    if C is None:
        Cinv = torch.ones_like(Y)
    elif C.ndim == 1:
        Cinv = 1.0 / C
    else:
        Cinv = torch.linalg.inv(C)

    def chi2_and_grad(x):
        fY  = f(x)
        dY  = Y - fY
        chi2= (dY**2 * Cinv).sum()
        _, vjp_fn = vjp(f, x)
        grad = vjp_fn(Cinv * dY)[0]
        return chi2, grad

    def hvp(v):
        _, jvp_out = jvp(f, (X,), (v,))
        w = Cinv * jvp_out
        _, vjp_fn = vjp(f, X)
        return vjp_fn(w)[0] + L * v

    chi2, grad = chi2_and_grad(X)
    if verbose:
        print(f"{'Iter':>4} | {'chi2':>12} | {'chi2_new':>12} | {'λ':>8} | {'ρ':>6} | {'acc':>4}")
        print("-"*60)

    for it in range(max_iter):
        # solve for step h
        h = cg_solve(hvp, grad, maxiter=cg_maxiter, tol=cg_tol)

        # evaluate new χ²
        chi2_new, _ = chi2_and_grad(X + h)
        expected = torch.dot(h, hvp(h) + grad)
        rho = (chi2 - chi2_new) / torch.abs(expected)

        accepted = (rho >= epsilon)
        # update
        if accepted:
            X, chi2 = X + h, chi2_new
            L = max(L / L_dn, L_min)
        else:
            L = min(L * L_up, L_max)

        if verbose:
            i = 2
            D = X.numel()

            # build a basis vector e_i
            e_i = torch.zeros(D, device=X.device, dtype=X.dtype)
            e_i[i] = 1.0
            
            # apply your hvp to get (H + L·I)·e_i
            Hi_plus_L = hvp(e_i)[i].item()
            
            # subtract off the damping to get the true diagonal H_{ii}
            H_ii = Hi_plus_L - L
            
            #print(f"param[{i}]: grad={grad[i].item():.3e}, H_ii={H_ii:.3e}, L={L:.3e}")
            print(f"{it:4d} | {chi2.item():12.4e} | {chi2_new.item():12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(accepted):>4} M_bh : {X[2]}, {h[2]}, \n {h}")
            print(X)

        if torch.norm(h) < stopping:
            break

        # new gradient
        _, grad = chi2_and_grad(X)

    return X, L, chi2


def lm_direct(
    X, Y, f,
    C=None,
    epsilon=1e-1, L=1.0, L_dn=11.0, L_up=9.0,
    max_iter=50, cg_maxiter=20, cg_tol=1e-6,   # cg_* kept for signature compatibility (unused)
    L_min=1e-9, L_max=1e9,
    stopping=1e-4,
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

        # Solve (use Cholesky if PD, else fallback)
        try:
            # Safer to add a tiny jitter if near-singular
            h = torch.cholesky_solve(
                torch.cholesky(H_damped, upper=False).transpose(-1, -2) @ grad.unsqueeze(-1),
                torch.cholesky(H_damped, upper=False)
            )  # This is convoluted; simpler to just use solve unless you *need* cholesky
        except:
            # Fallback to generic solve
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
            print(f"param[{i}]: grad={grad[i].item():.3e}, H_ii={H_ii:.3e}, L={L:.3e}")
            print(f"{it:4d} | {chi2.item():12.4e} | {chi2_new.item():12.4e} | {L:8.2e} | {rho.item():6.3f} | {str(bool(accepted)):>4} "
                  f"M_bh : {X[i] if Din>2 else X[0]}, {h[i] if Din>2 else h[0]},\n {h}")

        # Stopping criterion
        if torch.norm(h) < stopping:
            break
        if L >= L_max:
            break

    return X, L, chi2
