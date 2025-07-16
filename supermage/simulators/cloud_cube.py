import math, torch
from caskade import Module, Param, forward            # same import style you use
import torch.nn.functional as F
# ----------------------------------------------------------------------
# Helper: equal-probability Gaussian abscissae -------------------------
# ----------------------------------------------------------------------
def gaussian_quantile_offsets(sigma, K, *, device, dtype):
    p_mid = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K
    return sigma * math.sqrt(2.0) * torch.erfinv(2.0 * p_mid - 1.0)

def make_dv_table(N_clouds, K_vel, *, seed, device, dtype):
    """
    Deterministic σ=1 Gaussian jitter table  →  shape (N_clouds, K_vel)
    Uses a scrambled Sobol low‑discrepancy sequence so each row is stratified
    but different.  You can swap this for any generator you like.
    """
    # 1. reproducible uniform [0,1) matrix
    sobol = torch.quasirandom.SobolEngine(
        dimension=K_vel, scramble=True, seed=seed
    )
    u = sobol.draw(int(N_clouds)).to(device=device, dtype=dtype)   # (N,K)

    # 2. map uniform → standard normal N(0,1)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)       # (N,K)


# ----------------------------------------------------------------------
#  MC cloud catalogue           (only forward() was modified)
# ----------------------------------------------------------------------
class CloudCatalog(Module):
    def __init__(
        self,
        intensity_model,
        velocity_model,
        fov_half_pc,
        N_clouds,
        K_vel,
        brightness_init,
        distance_pc,
        sampling_method = "sobol_uniform",
        seed=42,
        device="cuda",
        dtype=torch.float64,
        name="clouds",
    ):
        super().__init__(name)
        self.device, self.dtype = device, dtype
        self.intensity_model, self.velocity_model = intensity_model, velocity_model
        self.K_vel, self.D_pc = K_vel, float(distance_pc)

        # ---------- static MC catalogue ---------------------------------
        if sampling_method == "sobol_uniform":
            # --- low‑discrepancy Sobol points over the square FoV ------
            sobol = torch.quasirandom.SobolEngine(
                dimension=2, scramble=True, seed=seed
            )
            # draw in [-1,1]^2 then scale to pc
            self.pos_gal0 = (sobol.draw(int(N_clouds), dtype = dtype) * 2.0 - 1.0) * fov_half_pc

        elif sampling_method == "proportional":
            # existing rejection sampler (unchanged) --------------------
            gen = torch.Generator(device).manual_seed(seed)
            I0  = self.intensity_model.brightness(0.0, brightness_init)
            pts = []
            while len(pts) < N_clouds:
                xy = (torch.rand(2, device=device, dtype=dtype, generator=gen) * 2 - 1) \
                     * fov_half_pc
                R  = torch.hypot(*xy)
                if torch.rand(1, generator=gen).item() * I0 <= \
                        self.intensity_model.brightness(R, brightness_init):
                    pts.append(xy)
            self.pos_gal0 = torch.stack(pts)                      # (N,2) pc

        elif sampling_method == "uniform":
            # existing pure‑uniform sampler (unchanged) -----------------
            gen = torch.Generator(device).manual_seed(seed)
            self.pos_gal0 = (
                torch.rand((N_clouds, 2), generator=gen,
                           device=device, dtype=dtype) * 2.0 - 1.0
            ) * fov_half_pc

        else:
            raise ValueError(f"Unknown sampling_method '{sampling_method}'.")

        # ---------- velocity‑broadening template ------------------------
        self.dv_template = gaussian_quantile_offsets(
            torch.ones((), device=device, dtype=dtype),
            K_vel, device=device, dtype=dtype,
        )

        self.dv_unit = make_dv_table(N_clouds, K_vel,
                          seed=seed, device=device, dtype=dtype)

        # ---------- global fit parameters -------------------------------
        self.inclination     = Param("inclination", None)   # rad
        self.sky_rot         = Param("sky_rot", None)       # rad
        self.line_broadening = Param("line_broadening", None)
        self.velocity_shift  = Param("velocity_shift", None)
        self.x0              = Param("x0", None)            # ″  (–ΔRA)
        self.y0              = Param("y0", None)            # ″  (+ΔDec)

        # pass inclination to nested model for autograd
        self.velocity_model.inc = self.inclination

    # ------------------------------------------------------------------
    @forward
    def forward(
        self,
        inclination=None,
        sky_rot=None,
        line_broadening=None,
        velocity_shift=None,
        x0=None,
        y0=None,
        return_subsamples: bool = False,
        gaussian_quantile = True
    ):
        # -------- aliases & trig ---------------------------------------
        x_gal, y_gal = self.pos_gal0.T                        # pc
        cos_i, sin_i = torch.cos(inclination), torch.sin(inclination)
        pa      = sky_rot + math.pi / 2.0                  # keep your variable name
        cos_pa  = torch.cos(pa)
        sin_pa  = torch.sin(pa)

        # -------- intrinsic radius & dynamics --------------------------
        R = torch.hypot(x_gal, y_gal)
        flux_cloud = self.intensity_model.brightness(R)
        v_circ     = self.velocity_model.velocity(R)
        cos_theta =  x_gal / (R + 1e-12)            #  +x_gal = receding
        v_los     =  v_circ * sin_i * cos_theta + velocity_shift

        # ------------------------------------------------------------------
        # 2) sky‑plane projection  (inverse of grid simulator)
        # ------------------------------------------------------------------
        x_sky_pc =  cos_pa * x_gal - sin_pa * (y_gal * cos_i)   #  east  (+)
        y_sky_pc =  sin_pa * x_gal + cos_pa * (y_gal * cos_i)   #  north (+)

        # ------------------------------------------------------------------
        # 3) pc → arcsec  + global offsets  (+x0 = shift to the **right**)
        # ------------------------------------------------------------------
        arcsec_per_pc = 206265.0 / self.D_pc
        ra_east   =  x_sky_pc * arcsec_per_pc + x0      # +x0  = shift right
        dec_north =  y_sky_pc * arcsec_per_pc + y0      # +y0  = shift up

        # ------------------------------------------------------------------
        # 4) velocity broadening  (flip sign so red = receding = north)
        # ------------------------------------------------------------------
        if gaussian_quantile:
            Δv_k     = gaussian_quantile_offsets(
                  line_broadening, self.K_vel, device=self.device, dtype=self.dtype)
            vel_chan =  v_los.unsqueeze(-1) + Δv_k
            flux_sub = flux_cloud.unsqueeze(-1).expand(-1, self.K_vel) / self.K_vel
        else:
            Δv_k = line_broadening.unsqueeze(-1) * self.dv_unit       # broadcast σ
            vel_chan = v_los.unsqueeze(-1) + Δv_k                     # (N,K)
        
            flux_sub = (flux_cloud / self.K_vel)[:, None].expand(-1, self.K_vel)

        # ------------------------------------------------------------------
        # 5) broadcast spatial coordinates   **horizontal = RA**
        # ------------------------------------------------------------------
        pos_img = torch.stack([ra_east, dec_north], dim=-1) \
                      .unsqueeze(1).expand(-1, self.K_vel, -1).clone()

        return (pos_img, vel_chan, flux_sub) if return_subsamples else {
            "pos_img": pos_img, "vel_chan": vel_chan, "flux": flux_sub
        }


# ----------------------------------------------------------------------
#  Rasteriser                    (unchanged *except* axis comment)
# ----------------------------------------------------------------------
class CloudRasterizer(Module):
    """
    Rasterises cloudlets onto a data cube.

    pixel x‑index ↔ Dec   (north ↑   → right)
    pixel y‑index ↔ –RA   (east →   → up)
    """
    def __init__(
        self,
        cloudcatalog,
        vel_axis,                 # (Nv,) uniform
        pixel_scale_arcsec,       # ″ / pix  (positive)
        N_pix_x,                  # = NAXIS1
        device="cuda",
        dtype=torch.float32,
        name="raster",
    ):
        super().__init__(name)
        self.device, self.dtype = device, dtype

        # velocity axis -------------------------------------------------
        if not torch.allclose(vel_axis[1:] - vel_axis[:-1],
                              vel_axis[1]  - vel_axis[0]):
            raise ValueError("vel_axis must be uniformly spaced.")
        self.vel0 = vel_axis[0].to(dtype)
        self.dv   = float((vel_axis[1] - vel_axis[0]).item())
        self.Nv   = vel_axis.numel()

        # spatial grid --------------------------------------------------
        self.pixscale = float(pixel_scale_arcsec)          # ″ per pixel
        self.N_pix    = int(N_pix_x)
        
        # pixel centres run from −((N‑1)/2)·Δ … +((N‑1)/2)·Δ
        self.fov_half = 0.5 * (self.N_pix - 1) * self.pixscale   # <── centred!

        self.clouds = cloudcatalog

    @staticmethod
    def _index_and_frac(x):
        i0 = torch.floor(x).to(torch.long)
        return i0, x - i0.to(x.dtype)

    @forward
    def forward(self):
        pos_img, vel_chan, flux = self.clouds.forward(return_subsamples=True)
        M = pos_img.shape[0] * pos_img.shape[1]
        ra  = pos_img[..., 0].reshape(M)        # ″  (east + → right)
        dec = pos_img[..., 1].reshape(M)        # ″  (north + → up)
        v   = vel_chan.reshape(M)
        f   = flux.reshape(M)
        
        # RA → pixel x‑index ;  Dec → pixel y‑index
        ix0, fx = self._index_and_frac((ra  + self.fov_half) / self.pixscale)
        iy0, fy = self._index_and_frac((dec + self.fov_half) / self.pixscale)
        iv0, fv = self._index_and_frac((v - self.vel0) / self.dv)

        mask = (
            (ix0 >= 0) & (ix0 < self.N_pix - 1) &
            (iy0 >= 0) & (iy0 < self.N_pix - 1) &
            (iv0 >= 0) & (iv0 < self.Nv    - 1)
        )
        if mask.sum() == 0:
            return torch.zeros(self.Nv, self.N_pix, self.N_pix,
                               device=self.device, dtype=f.dtype)

        ix0, iy0, iv0 = ix0[mask], iy0[mask], iv0[mask]
        fx,  fy,  fv  = fx [mask], fy [mask], fv [mask]
        f             = f  [mask]

        ix1, iy1, iv1 = ix0 + 1, iy0 + 1, iv0 + 1
        wx0, wy0, wv0 = 1 - fx, 1 - fy, 1 - fv
        wx1, wy1, wv1 =     fx,     fy,     fv

        ix = torch.stack([ix0, ix0, ix0, ix0, ix1, ix1, ix1, ix1], dim=1)
        iy = torch.stack([iy0, iy0, iy1, iy1, iy0, iy0, iy1, iy1], dim=1)
        iv = torch.stack([iv0, iv1, iv0, iv1, iv0, iv1, iv0, iv1], dim=1)
        wx = torch.stack([wx0, wx0, wx0, wx0, wx1, wx1, wx1, wx1], dim=1)
        wy = torch.stack([wy0, wy1, wy0, wy1, wy0, wy1, wy0, wy1], dim=1)
        wv = torch.stack([wv0, wv1, wv0, wv1, wv0, wv1, wv0, wv1], dim=1)

        f_w = f.unsqueeze(1) * (wx * wy * wv)

        idx_flat = (iv * self.N_pix + iy) * self.N_pix + ix
        cube_flat = torch.zeros(self.Nv * self.N_pix * self.N_pix,
                                device=self.device, dtype=f.dtype)
        cube_flat = cube_flat.scatter_add(0, idx_flat.reshape(-1), f_w.reshape(-1))

        return cube_flat.view(self.Nv, self.N_pix, self.N_pix)


class CloudRasterizerOversample(Module):
    # ──────────────────────────────────────────────────────────────────
    def __init__(self,
                 cloudcatalog,
                 vel_axis,              # (Nv,) uniform
                 pixel_scale_arcsec,
                 N_pix_x,
                 oversamp_xy: int = 4,
                 oversamp_v : int = 4,   # NEW: velocity oversampling
                 device: str = "cuda",
                 dtype : torch.dtype = torch.float32,
                 name  : str = "raster"):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.clouds      = cloudcatalog
        self.oversamp_xy = int(oversamp_xy)
        self.oversamp_v  = int(oversamp_v)

        # ── low‑res velocity grid (given) ────────────────────────────
        if not torch.allclose(vel_axis[1:] - vel_axis[:-1],
                              vel_axis[1]  - vel_axis[0]):
            raise ValueError("vel_axis must be uniformly spaced.")
        self.vel0_lo = vel_axis[0].to(dtype)
        self.dv_lo   = float((vel_axis[1] - vel_axis[0]).item())
        self.Nv_lo   = vel_axis.numel()

        # ── high‑res velocity grid (offset by δ) ─────────────────────
        self.dv_hi   = self.dv_lo / self.oversamp_v
        delta        = 0.5 * (self.dv_lo - self.dv_hi)      # centre‑align shift
        self.vel0_hi = self.vel0_lo - delta                 # **key change**
        self.Nv_hi   = self.Nv_lo * self.oversamp_v

        # ── low‑res spatial grid ─────────────────────────────────────
        self.pixscale_lo = float(pixel_scale_arcsec)
        self.N_pix_lo    = int(N_pix_x)
        self.fov_half_lo = 0.5 * (self.N_pix_lo - 1) * self.pixscale_lo

        # ── high‑res spatial grid ────────────────────────────────────
        self.pixscale_hi = self.pixscale_lo / self.oversamp_xy
        self.N_pix_hi    = self.N_pix_lo * self.oversamp_xy
        self.fov_half_hi = 0.5 * (self.N_pix_hi - 1) * self.pixscale_hi

    # ------------------------------------------------------------------
    @staticmethod
    def _index_and_frac(x: torch.Tensor):
        i0 = torch.floor(x).to(torch.long)
        return i0, x - i0.to(x.dtype)

    # ------------------------------------------------------------------
    def _rasterise_hi(self, ra, dec, vel, flux):
        """
        Splat onto the high‑resolution grid (Nv_hi, Nx_hi, Ny_hi)
        using tri‑linear interpolation.
        """
        ix0, fx = self._index_and_frac((ra  + self.fov_half_hi) / self.pixscale_hi)
        iy0, fy = self._index_and_frac((dec + self.fov_half_hi) / self.pixscale_hi)
        iv0, fv = self._index_and_frac((vel - self.vel0_hi)     / self.dv_hi)

        mask = (
            (ix0 >= 0) & (ix0 < self.N_pix_hi - 1) &
            (iy0 >= 0) & (iy0 < self.N_pix_hi - 1) &
            (iv0 >= 0) & (iv0 < self.Nv_hi   - 1)
        )
        if mask.sum() == 0:
            return torch.zeros(self.Nv_hi, self.N_pix_hi, self.N_pix_hi,
                               device=self.device, dtype=flux.dtype)

        ix0, iy0, iv0 = ix0[mask], iy0[mask], iv0[mask]
        fx,  fy,  fv  = fx [mask], fy [mask], fv [mask]
        flux          = flux[mask]

        ix1, iy1, iv1 = ix0 + 1, iy0 + 1, iv0 + 1
        wx0, wy0, wv0 = 1 - fx, 1 - fy, 1 - fv
        wx1, wy1, wv1 =     fx,     fy,     fv

        ix = torch.stack([ix0, ix0, ix0, ix0, ix1, ix1, ix1, ix1], dim=1)
        iy = torch.stack([iy0, iy0, iy1, iy1, iy0, iy0, iy1, iy1], dim=1)
        iv = torch.stack([iv0, iv1, iv0, iv1, iv0, iv1, iv0, iv1], dim=1)
        wx = torch.stack([wx0, wx0, wx0, wx0, wx1, wx1, wx1, wx1], dim=1)
        wy = torch.stack([wy0, wy1, wy0, wy1, wy0, wy1, wy0, wy1], dim=1)
        wv = torch.stack([wv0, wv1, wv0, wv1, wv0, wv1, wv0, wv1], dim=1)

        f_w = flux.unsqueeze(1) * (wx * wy * wv)

        idx_flat = (iv * self.N_pix_hi + iy) * self.N_pix_hi + ix
        cube_flat = torch.zeros(self.Nv_hi * self.N_pix_hi * self.N_pix_hi,
                                device=self.device, dtype=flux.dtype)
        cube_flat.scatter_add_(0, idx_flat.reshape(-1), f_w.reshape(-1))

        return cube_flat.view(self.Nv_hi, self.N_pix_hi, self.N_pix_hi)

    # ------------------------------------------------------------------
    @forward
    def forward(self):
        """
        Returns
        -------
        cube_lo : Tensor  (Nv_lo, N_pix_lo, N_pix_lo)
        """
        pos_img, vel_chan, flux = self.clouds.forward(return_subsamples=True)
        M = pos_img.numel() // 2

        ra  = pos_img[..., 0].reshape(M)
        dec = pos_img[..., 1].reshape(M)
        vel = vel_chan.reshape(M)
        flx = flux.reshape(M)

        # ---- high‑res raster ----------------------------------------
        cube_hi = self._rasterise_hi(ra, dec, vel, flx)
        #     shape: (Nv_hi, N_pix_hi, N_pix_hi)

        # ---- box‑filter / down‑sample in v and (x,y) ----------------
        cube_hi = cube_hi.view(
            self.Nv_lo,  self.oversamp_v,
            self.N_pix_lo, self.oversamp_xy,
            self.N_pix_lo, self.oversamp_xy
        )
        cube_lo = cube_hi.mean((1, 3, 5))           # average over v,x,y sub‑cells

        return cube_lo

class GaussianSplatRasterizerAntialiased(Module):
    """
    Bilinear (floor‑based) rasteriser + symmetric Gaussian blur.
    Matches the original CloudRasterizer centring *exactly*,
    removes aliasing, keeps gradients through the Gaussian.
    """
    def __init__(
        self,
        cloudcatalog,
        vel_axis,                 # (Nv,) uniform
        pixel_scale_arcsec,       # ″ / pix
        N_pix_x,                  # cube size (square)
        sigma_pix=0.8, truncate=3.0,
        device="cuda", dtype=torch.float32,
    ):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.clouds = cloudcatalog

        # ----- velocity axis -----------------------------------------
        if not torch.allclose(vel_axis[1:] - vel_axis[:-1],
                              vel_axis[1]  - vel_axis[0]):
            raise ValueError("vel_axis must be uniformly spaced.")
        self.vel0 = vel_axis[0].to(dtype)
        self.dv   = float((vel_axis[1] - vel_axis[0]).item())
        self.Nv   = vel_axis.numel()

        # ----- spatial grid ------------------------------------------
        self.pixscale = float(pixel_scale_arcsec)
        self.N_pix    = int(N_pix_x)
        self.fov_half = 0.5 * (self.N_pix - 1) * self.pixscale   # ″

        # ----- separable Gaussian kernel (L1‑normalised) ------------
        half = int(math.ceil(truncate * sigma_pix))
        x = torch.arange(-half, half + 1, device=device, dtype=dtype)
        g1 = torch.exp(-0.5 * (x / sigma_pix) ** 2)
        g1 = g1 / g1.sum()
        self.kernel2d = (g1[:, None] * g1[None, :]).unsqueeze(0).unsqueeze(0)

    # ----------------------------------------------------------------
    @staticmethod
    def _index_and_frac(x):
        i0 = torch.floor(x).to(torch.long)
        return i0, x - i0.to(x.dtype)     # 0 ≤ frac < 1

    # ----------------------------------------------------------------
    @forward
    def forward(self):
        pos_img, vel_chan, flux = self.clouds.forward(return_subsamples=True)
        M = pos_img.numel() // 2          # N_clouds * K_vel

        ra  = pos_img[..., 0].reshape(M)
        dec = pos_img[..., 1].reshape(M)
        vel = vel_chan.reshape(M)
        flx = flux.reshape(M)

        # floor‑based indices + fractions (0 ≤ fx,fy,fv < 1) ------------
        ix0, fx = self._index_and_frac((ra  + self.fov_half) / self.pixscale)
        iy0, fy = self._index_and_frac((dec + self.fov_half) / self.pixscale)
        iv0, fv = self._index_and_frac((vel - self.vel0)    / self.dv)

        mask = (
            (ix0 >= 0) & (ix0 < self.N_pix - 1) &
            (iy0 >= 0) & (iy0 < self.N_pix - 1) &
            (iv0 >= 0) & (iv0 < self.Nv   - 1)
        )
        if mask.sum() == 0:
            return torch.zeros(self.Nv, self.N_pix, self.N_pix,
                               device=self.device, dtype=flx.dtype)

        ix0, iy0, iv0 = ix0[mask], iy0[mask], iv0[mask]
        fx,  fy, fv   = fx [mask], fy [mask], fv [mask]
        flx           = flx[mask]

        # ----------------------- ⊞  BUILD 8 WEIGHTS  -------------------
        ix1, iy1, iv1 = ix0 + 1, iy0 + 1, iv0 + 1
        wx0, wy0, wv0 = 1 - fx, 1 - fy, 1 - fv
        wx1, wy1, wv1 =     fx,     fy,     fv

        ix = torch.stack([ix0, ix0, ix1, ix1, ix0, ix0, ix1, ix1], dim=1)
        iy = torch.stack([iy0, iy1, iy0, iy1, iy0, iy1, iy0, iy1], dim=1)
        iv = torch.stack([iv0, iv0, iv0, iv0, iv1, iv1, iv1, iv1], dim=1)

        w  = torch.stack([
            wx0*wy0*wv0,  wx0*wy1*wv0,  wx1*wy0*wv0,  wx1*wy1*wv0,
            wx0*wy0*wv1,  wx0*wy1*wv1,  wx1*wy0*wv1,  wx1*wy1*wv1
        ], dim=1)
        # --------------------------------------------------------------

        cube = torch.zeros(self.Nv, self.N_pix, self.N_pix,
                           device=self.device, dtype=flx.dtype)
        idx_flat = (iv * self.N_pix + iy) * self.N_pix + ix
        cube.view(-1).scatter_add_(0, idx_flat.reshape(-1),
                                   (flx.unsqueeze(1) * w).reshape(-1))

        # Gaussian anti‑alias filter (reflect padding keeps centroid)
        pad = self.kernel2d.size(-1) // 2
        cube = F.pad(cube.unsqueeze(1), (pad, pad, pad, pad), mode="reflect")
        cube = F.conv2d(cube, self.kernel2d).squeeze(1)

        return cube                     # (Nv, N_pix, N_pix)