import torch
from caskade import Module, forward, Param
from torch import vmap
import caustics
from caustics.light import Pixelated
from torch.nn.functional import avg_pool2d, conv2d


class CubeLens(Module):
    def __init__(
        self,
        lens,
        source_cube,
        pixelscale_source,
        pixelscale_lens,
        pixels_x_source,
        pixels_x_lens,
        upsample_factor,
        name: str = "sim",
    ):
        super().__init__(name)

        self.lens = lens
        self.source_cube = source_cube
        self.upsample_factor = upsample_factor
        self.src = Pixelated(name="source", shape=(pixels_x_source, pixels_x_source), pixelscale=pixelscale_source, image = torch.zeros((pixels_x_source, pixels_x_source)))

        # Create the high-resolution grid
        thx, thy = caustics.utils.meshgrid(
            pixelscale_lens / upsample_factor,
            upsample_factor * pixels_x_lens,
            dtype=torch.float32, device = "cuda"
        )

        self.thx = thx
        self.thy = thy

    @forward
    def forward(self, lens_source = True):
        cube = self.source_cube.forward()
        bx, by = self.lens.raytrace(self.thx, self.thy)

        def lens_channel(image):
            if lens_source:
                return self.src.brightness(bx, by, image = image)
            else:
                return self.src.brightness(self.thx, self.thy, image = image)
        
        # Ray-trace to get the lensed positions
        lensed_cube = vmap(lens_channel)(cube)
        del cube

        # Downsample to the desired resolution
        lensed_cube = avg_pool2d(lensed_cube[:, None], self.upsample_factor)[:, 0]
        torch.cuda.empty_cache()
        return lensed_cube


class CubePosition(Module):
    """
    Generates an off-center (x, y position offset in arcsec) cube with the correct padding to match the FOV of the data. Note that the x offset parameter is in negative RA so that it increases from left to right.
    Parameters
    ----------
    source_cube (Module): The source 3D cube to be lensed.
    pixelscale_source (float): The pixel scale for the source cube.
    pixelscale_lens (float): The pixel scale for the output grid.
    pixels_x_source (int): The number of pixels in the source cube in the x-direction.
    pixels_x_lens (int): The number of pixels in the output grid in the x-direction.
    upsample_factor (int): The factor by which to upsample the image for lensing.
    name (str, optional): The name of the module. Default is "sim".
    """
    def __init__(
        self,
        source_cube,
        pixelscale_source,
        pixelscale_lens,
        pixels_x_source,
        pixels_x_lens,
        upsample_factor,
        name: str = "sim",
    ):
        super().__init__(name)
        
        self.source_cube = source_cube
        self.upsample_factor = upsample_factor
        self.src = Pixelated(name="source", shape=(pixels_x_source, pixels_x_source), pixelscale=pixelscale_source, image = torch.zeros((pixels_x_source, pixels_x_source)))

        # Create the high-resolution grid
        thx, thy = caustics.utils.meshgrid(
            pixelscale_lens / upsample_factor,
            upsample_factor * pixels_x_lens,
            dtype=torch.float32, device = "cuda"
        )

        self.thx = thx
        self.thy = thy

    @forward
    def forward(self):
        cube = self.source_cube.forward()

        def lens_channel(image):
            return self.src.brightness(self.thx, self.thy, image = image)
        
        # Ray-trace to get the lensed positions
        lensed_cube = vmap(lens_channel)(cube)
        del cube

        # Downsample to the desired resolution
        lensed_cube = avg_pool2d(lensed_cube[:, None], self.upsample_factor)[:, 0]
        torch.cuda.empty_cache()
        return lensed_cube