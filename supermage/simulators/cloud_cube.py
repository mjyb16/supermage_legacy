import math, torch
from caskade import Module, Param, forward            # same import style you use

# ----------------------------------------------------------------------
# Helper: equal-probability Gaussian abscissae -------------------------
# ----------------------------------------------------------------------
def gaussian_quantile_offsets(sigma, K, *, device, dtype):
    """
    Return K offsets Δv_k such that each represents an equal-probability bin
    of 𝒩(0, sigma²).  All weights are therefore equal (=1/K).

    Parameters
    ----------
    sigma : torch.Tensor or float  (broadcastable)
    K     : int  – number of velocity samples per cloud
    """
    p_mid  = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K          # (K,)
    # Δv = σ √2 erf⁻¹(2p − 1)
    return sigma * math.sqrt(2.0) * torch.erfinv(2.0 * p_mid - 1.0)           # (K,)
# ----------------------------------------------------------------------


class CloudCatalog(Module):
    """
    One-off Monte-Carlo catalogue of N “clouds”, reused every forward pass.

    Parameters
    ----------
    intensity_model : Caskade Module with .brightness(R) method
    velocity_model  : Caskade Module with .velocity(R)   method
    fov_half        : half-width of the intrinsic galactic plane (same unit as model)
    N_clouds        : number of Monte-Carlo points to draw
    K_vel           : number of Gaussian velocity offsets per cloud
    seed            : RNG seed for deterministic catalogue
    device, dtype   : obvious
    """
    def __init__(
        self,
        intensity_model,
        velocity_model,
        fov_half: float,
        N_clouds: int,
        K_vel: int,
        brightness_init,
        *,
        seed: int = 42,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
        name: str = "clouds",
    ):
        super().__init__(name=name)

        self.device, self.dtype = device, dtype
        self.intensity_model = intensity_model
        self.velocity_model  = velocity_model
        self.K_vel           = K_vel

        # --------------------------------------------------------------
        # 1) DRAW MONTE-CARLO POSITIONS (x_gal, y_gal) -----------------
        # --------------------------------------------------------------
        gen = torch.Generator(device).manual_seed(seed)

        # simple rejection sampling inside a square [-fov,+fov]² --------
        # keep going until we have N_clouds accepted
        pts = []
        while len(pts) < N_clouds:
            xy = (torch.rand(2, device=device, dtype=dtype, generator=gen) * 2 - 1) * fov_half
            R  = torch.hypot(*xy)
            # draw acceptance threshold from 0..I_max (unknown) – we use
            # intensity at (0,0) as crude upper bound (brightness is usually monotonic)
            I0 = self.intensity_model.brightness(0.0, brightness_init)
            if torch.rand(1, generator=gen).item() * I0 <= self.intensity_model.brightness(R, brightness_init):
                pts.append(xy)

        pos_gal = torch.stack(pts)                              # (N,2)
        self.pos_gal0 = pos_gal

        # --------------------------------------------------------------
        # 2) STATIC VELOCITY-GRID OFFSETS (size K_vel) -----------------
        # --------------------------------------------------------------
        Δv_dummy = torch.ones(())  # will be scaled by σ in forward()
        self.dv_template = gaussian_quantile_offsets(
            Δv_dummy, K_vel, device=device, dtype=dtype)

        # --------------------------------------------------------------
        # 3) LEARNABLE / FITTABLE PARAMETERS (global) -----------------
        # --------------------------------------------------------------
        self.inclination     = Param("inclination", None)   # rad
        self.sky_rot         = Param("sky_rot",     None)   # rad
        self.line_broadening = Param("line_broadening", None)
        self.velocity_shift  = Param("velocity_shift", None)

        # Make sure nested models “see” the inclination tensor so their
        # gradients can flow (same trick you used before)
        self.velocity_model.inc = self.inclination


    # ------------------------------------------------------------------
    # Forward pass : produce per-cloud, per-subsample quantities --------
    # ------------------------------------------------------------------
    @forward
    def forward(self, inclination=None, sky_rot=None, line_broadening=None, velocity_shift = None, return_subsamples: bool = False):
        """
        Returns
        -------
        pos_img  : (N,K,2)  sky-plane positions [arcsec or whatever your units]
        vel_chan : (N,K)    LOS velocity for each subsample  [km/s]
        flux     : (N,K)    (unnormalised) flux carried by each subsample
        """
        # ----------------------------------------------------------------
        # 1) Transform base catalogue to current galaxy orientation -------
        # ----------------------------------------------------------------
        x_gal, y_gal = self.pos_gal0.T                                               # (N,)

        cos_i, sin_i  = torch.cos(inclination), torch.sin(inclination)
        cos_pa, sin_pa = torch.cos(sky_rot), torch.sin(sky_rot)

        # Rotate in galaxy plane by PA, then de-project y
        x_rot =  cos_pa * x_gal - sin_pa * y_gal
        y_rot = (sin_pa * x_gal + cos_pa * y_gal) / cos_i

        R = torch.hypot(x_rot, y_rot)                                                # (N,)

        # ----------------------------------------------------------------
        # 2) Intrinsic flux and circular velocity -------------------------
        # ----------------------------------------------------------------
        flux_cloud   = self.intensity_model.brightness(R)                            # (N,)
        v_circ_cloud = self.velocity_model.velocity(R)                               # (N,)

        # LOS component : v_los = v_circ sin(i) cos(θ)
        cos_theta = x_rot / (R + 1e-12)
        v_los     = v_circ_cloud * sin_i * cos_theta + velocity_shift           # (N,)

        # ----------------------------------------------------------------
        # 3) Velocity broadening via fixed grid --------------------------
        # ----------------------------------------------------------------
        σ_v   = line_broadening
        Δv_k  = gaussian_quantile_offsets(σ_v, self.K_vel,
                                          device=self.device, dtype=self.dtype)      # (K,)

        # broadcast: (N,1) + (K,) → (N,K)
        vel_chan = v_los.unsqueeze(-1) + Δv_k                                        # (N,K)

        # split total cloud flux equally over K subsamples (relative const.)
        flux_sub = flux_cloud.unsqueeze(-1).expand(-1, self.K_vel) / self.K_vel      # (N,K)

        # ----------------------------------------------------------------
        # 4) Sky-plane coordinates (optionally with external ray-tracing) -
        # ----------------------------------------------------------------
        # For now: simple shift (no lens).  Downstream you can run Caustics
        # in a separate step if you wish, or wrap this call with a LensModule.
        pos_img = torch.stack([x_rot, y_rot], dim=-1).unsqueeze(1)                   # (N,1,2)
        pos_img = pos_img.expand(-1, self.K_vel, -1).clone()                         # (N,K,2)

        if return_subsamples:
            return pos_img, vel_chan, flux_sub        # low-level building blocks
        else:
            # Simple convenience output for scatter-add stage
            return {
                "pos_img":  pos_img,   # (N,K,2)
                "vel_chan": vel_chan,  # (N,K)
                "flux":     flux_sub,  # (N,K)
            }