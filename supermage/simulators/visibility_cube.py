import torch
from caskade import Module, forward, Param
import numpy as np
from supermage.utils.uv_utils import generate_pb

class VisibilityCube(Module):
    def __init__(
        self, 
        cube_simulator, 
        data_u, 
        data_v, 
        paduv, 
        freqs, 
        dish_diameter=12, 
        shape=(500, 500), 
        deltal=0.004, 
        device="cuda"
    ):
        super().__init__()
        self.cube_simulator = cube_simulator
        self.data_u = data_u
        self.data_v = data_v
        self.paduv = paduv
        self.freqs = freqs
        self.dish_diameter = dish_diameter
        self.shape = shape
        self.deltal = deltal
        self.device = device

        # Create primary beams
        pbs = []
        for freq in freqs:
            pb, _ = generate_pb(
                diameter=dish_diameter,
                freq=freq,
                shape=shape,
                deltal=deltal,
                device=device
            )
            pbs.append(pb)
        self.register_buffer("primary_beams", torch.stack(pbs, dim=0))  # shape (N_freq, Nx, Ny)

        # Create or store your uv mask
        uv_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        # fill uv_mask[...] = True where coverage is valid
        self.register_buffer("mask", uv_mask)

    @forward
    def forward(self, plot):
        # 1) Generate the "image-plane" data cube ( Nx, Ny, N_freq ) or whichever order your sim uses
        cube = self.cube_simulator.forward()

        # 2) Reorder to (N_freq, Nx, Ny) if necessary
        #    Suppose your simulator returns shape (Nx, Ny, N_freq):
        cube = cube.permute(2, 0, 1)

        def forward_lensed_vis(
            x,          # shape (N_freq, Nx, Ny)
            mask,       # shape (Nx, Ny) or (N_freq, Nx, Ny) depending on your usage
            primary_beams,  # shape (N_freq, Nx, Ny)
            device="cuda",
            plot=False
        ):
            """
            x: lensed cube, shape (N_freq, Nx, Ny)
            mask: UV mask, shape (Nx, Ny), or possibly (N_freq, Nx, Ny)
            primary_beams: model for each frequency, shape (N_freq, Nx, Ny)
            """
            def fft_channel(x_slice, pb_slice):
                # x_slice, pb_slice: each shape (Nx, Ny)
                fft_result = torch.fft.fftshift(
                    torch.fft.fft2(
                        torch.fft.ifftshift(x_slice * pb_slice, norm="ortho")
                    )
                )
                return fft_result
            
            fft_results = torch.vmap(fft_channel)(x, primary_beams)
            # fft_results: shape (N_freq, Nx, Ny), dtype=complex64
        
            if plot:
                mask_float = mask.float()
                mask_float = mask_float.unsqueeze(0)  # shape (1, Nx, Ny)
        
                visibilities_result = fft_results * mask_float
                return visibilities_result
            else:
                real_part = fft_results.real[:, mask]  # shape (N_freq, #mask_true)
                imag_part = fft_results.imag[:, mask]  # shape (N_freq, #mask_true)
                visibilities_result = real_part + 1j * imag_part
                return visibilities_result

        # 3) Compute the visibilities
        vis = forward_lensed_vis(
            x=cube,
            mask=self.mask,
            primary_beams=self.primary_beams,
            device=self.device,
            plot
        )
        return vis
