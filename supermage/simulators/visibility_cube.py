import torch
from caskade import Module, forward, Param
import numpy as np
from supermage.utils.uv_utils import gaussian_pb

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
        dish_diameter=12,  
        device="cuda"
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
        self.device = device

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
                device=device
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
        #Access velocity shifter, freqs and recalculate velocities in order to get velocity channel width
        normed_result = flux*pb_results/pb_results.sum()
        fft_results = torch.vmap(fft_channel)(normed_result)
        # fft_results: shape (N_freq, Nx, Ny), dtype=complex64
    
        if plot:
            # Simply multiply by the mask (broadcasting if needed)
            mask_f = self.mask.float()  # shape (N_chan, Nx, Ny)
            del cube
            torch.cuda.empty_cache()
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
            del cube
            torch.cuda.empty_cache()
            return visibilities_result
