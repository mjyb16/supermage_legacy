import torch
from torch.func import jvp, vjp

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

