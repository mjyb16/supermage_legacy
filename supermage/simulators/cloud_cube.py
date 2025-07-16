import math, torch
from caskade import Module, Param, forward            # same import style you use

# ----------------------------------------------------------------------
# Helper: equal-probability Gaussian abscissae -------------------------
# ----------------------------------------------------------------------
def gaussian_quantile_offsets(sigma, K, *, device, dtype):
    p_mid = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K
    return sigma * math.sqrt(2.0) * torch.erfinv(2.0 * p_mid - 1.0)


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
        sampling_method = None,
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
        if sampling_method == "proportional":
            gen = torch.Generator(device).manual_seed(seed)
            I0 = self.intensity_model.brightness(0.0, brightness_init)
            pts = []
            while len(pts) < N_clouds:
                xy = (torch.rand(2, device=device, dtype=dtype, generator=gen) * 2 - 1) \
                     * fov_half_pc
                R = torch.hypot(*xy)
                if torch.rand(1, generator=gen).item() * I0 <= \
                        self.intensity_model.brightness(R, brightness_init):
                    pts.append(xy)
            self.pos_gal0 = torch.stack(pts)                              # (N,2) pc
        else:
            gen = torch.Generator(device).manual_seed(seed)
            self.pos_gal0 = (torch.rand((int(N_clouds), 2), device=device, dtype=dtype, generator=gen) * 2 - 1) * fov_half_pc

        # ---------- velocity‑broadening template ------------------------
        self.dv_template = gaussian_quantile_offsets(
            torch.ones((), device=device, dtype=dtype),
            K_vel, device=device, dtype=dtype,
        )

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
        Δv_k     = gaussian_quantile_offsets(
              line_broadening, self.K_vel, device=self.device, dtype=self.dtype)
        vel_chan =  v_los.unsqueeze(-1) + Δv_k
        flux_sub = flux_cloud.unsqueeze(-1).expand(-1, self.K_vel) / self.K_vel

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