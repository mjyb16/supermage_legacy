import torch, math, torch.nn.functional as F
from caskade import Module, forward, Param
import numpy as np
from supermage.utils.uv_utils import gaussian_pb

class VisibilityCubePadded(Module):
    """
    Identical public API except that padding is done *after* PB‑weighting.
    """
    def __init__(
        self,
        cube_simulator,
        mask,
        freqs,
        npix,                 # final grid side
        pixelscale,           # ″ / pix on the final grid
        dish_diameter: float = 12.0,
    ):
        super().__init__()
        self.cube_simulator = cube_simulator
        self.mask           = mask
        self.freqs          = freqs
        self.npix           = npix
        self.pixelscale     = pixelscale
        self.dish_diameter  = dish_diameter

        self.device = cube_simulator.device
        self.dtype  = cube_simulator.dtype
        self.flux   = Param("flux", None)

        # ── size of the *small* cube returned by cube_simulator ─────────
        self.small_side = cube_simulator.N_pix      # uses the provided attribute
        if self.small_side > self.npix:
            raise ValueError(
                f"cube_simulator.N_pix ({self.small_side}) > npix ({self.npix})."
            )
        if (self.small_side%2) != (self.npix%2):
            raise ValueError(
                "Parity mismatch between cubes! Make shapes the same parity"
            )
        # symmetric padding widths: (left,right,top,bottom)
        pad_tot   = self.npix - self.small_side
        self.pad  = (pad_tot // 2, pad_tot - pad_tot // 2) * 2  # (L,R,T,B)

        # ── primary beams ON THE SMALL GRID ─────────────────────────────
        #   (no need to generate values we’ll pad with zeros later)
        self.pb_small = torch.stack(
            [
                gaussian_pb(
                    diameter=self.dish_diameter,
                    freq=f,
                    shape=(self.small_side, self.small_side),
                    deltal=self.pixelscale,          # same pixel scale
                    device=self.device,
                    dtype=self.dtype,
                )[0]                                # gaussian_pb returns (pb, _)
                for f in freqs
            ],
            dim=0,                                  # (N_chan, S, S)
        )

    # ---------------------------------------------------------------------
    @forward
    def forward(self, plot: bool, flux: torch.Tensor | None = None):
        """
        If `plot` is True → full UV cube masked by `mask`.
        If `plot` is False → only values where mask == True.
        """
        cube_small = self.cube_simulator.forward()          # (N_chan, S, S)

        # 1. multiply by primary beam on the same small grid
        cube_pb = cube_small * self.pb_small                # (N_chan, S, S)
        del cube_small

        # 2. scale to requested total flux *before* padding
        if flux is None:
            flux = self.flux if self.flux is not None else 1.0
        cube_pb = cube_pb * flux / cube_pb.sum()

        # 3. pad *both* spatial axes to (npix, npix)
        cube_pad = F.pad(cube_pb, self.pad, mode="constant", value=0.0)
        del cube_pb

        # 4. FFT  (channel‑wise 2‑D)
        fft_cube = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(cube_pad, dim=(-2, -1)),
                           norm="forward"),
            dim=(-2, -1),
        )
        del cube_pad                                           # memory

        # 5. deliver either the full UV plane or only the sampled points
        if plot:
            return fft_cube * self.mask.float()                # (N_chan, npix, npix)

        def gather(fft_slc, mask_slc):                         # per‑channel
            return fft_slc[mask_slc]
        return torch.vmap(gather)(fft_cube, self.mask)         # (N_chan, N_points)

class VisibilityCube(Module):
    def __init__(
        self, 
        cube_simulator, 
        #data_u, 
        #data_v, 
        #paduv, 
        mask,
        freqs, 
        npix,
        pixelscale,
        dish_diameter=12
    ):
        super().__init__()
        self.cube_simulator = cube_simulator
        #self.data_u = data_u
        #self.data_v = data_v
        #self.paduv = paduv
        self.mask = mask
        self.freqs = freqs
        self.dish_diameter = dish_diameter
        self.npix = npix
        self.pixelscale = pixelscale
        self.flux = Param("flux", None)
        self.device = cube_simulator.device
        self.dtype = cube_simulator.dtype

        #pixelscale_rad = self.pixelscale * (np.pi / 180.0) / 3600.0  # arcsec -> radians
        #self.pixel_area_sr = pixelscale_rad**2

        # Create primary beams
        pbs = []
        for freq in freqs:
            pb, _ = gaussian_pb(
                diameter=dish_diameter,
                freq=freq,
                shape=(npix, npix),
                deltal=pixelscale,
                device=self.device,
                dtype = self.dtype
            )
            pbs.append(pb)
        self.primary_beams = torch.stack(pbs, dim=0)  # shape (N_freq, Nx, Ny)

    @forward
    def forward(self, plot, flux = None):
        cube = self.cube_simulator.forward()

        def pb_cube(x_slice, pb_slice):
            return x_slice*pb_slice

        def fft_channel(x_slice):
            # x_slice, pb_slice: each shape (Nx, Ny)
            fft_result = torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(x_slice), 
                    norm="forward")
                )
            return fft_result

        pb_results = torch.vmap(pb_cube)(cube, self.primary_beams)
        del cube
        torch.cuda.empty_cache()
        #Access velocity shifter, freqs and recalculate velocities in order to get velocity channel width
        normed_result = flux*pb_results/pb_results.sum()
        fft_results = torch.vmap(fft_channel)(normed_result)
        del normed_result
        torch.cuda.empty_cache()
        # fft_results: shape (N_freq, Nx, Ny), dtype=complex64
    
        if plot:
            # Simply multiply by the mask (broadcasting if needed)
            mask_f = self.mask.float()  # shape (N_chan, Nx, Ny)
            return fft_results * mask_f
        else:
            # We want only the masked values. Let's define a small helper 
            # that gathers the complex values from fft_results where mask is True.
            def gather_masked(fft_slc, mask_slc):
                # fft_slc: complex (Nx, Ny)
                # mask_slc: bool (Nx, Ny)
                # gather real + i imag at the True locations
                return fft_slc[mask_slc]  # shape (#Trues,)

            # We'll vmap over (N_chan) as well
            # This will return shape (N_chan, #Trues) if each channel has the same #Trues
            visibilities_result = torch.vmap(gather_masked)(fft_results, self.mask)
            # shape: (N_chan, #Trues)
            return visibilities_result
