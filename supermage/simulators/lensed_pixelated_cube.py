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
        self.device = source_cube.device
        self.dtype = source_cube.dtype
        self.upsample_factor = upsample_factor
        self.src = Pixelated(name="source", shape=(pixels_x_source, pixels_x_source), pixelscale=pixelscale_source, image = torch.zeros((pixels_x_source, pixels_x_source)))

        # Create the high-resolution grid
        thx, thy = caustics.utils.meshgrid(
            pixelscale_lens / upsample_factor,
            upsample_factor * pixels_x_lens,
            device = source_cube.device, dtype = source_cube.dtype
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