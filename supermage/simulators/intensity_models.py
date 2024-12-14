import torch
from caustics import Module, forward, Param


class ExponentialUnnorm(Module):
    def __init__(self):
        super().__init__()
        self.sigma = Param("sigma", None)
        self.radius = Param("radius", None)
        self.sigma_z = Param("sigma_z", None)
        self.mu_z = Param("mu_z", None)

    @forward
    def brightness(self, x, y, z, sigma=None, radius=None, sigma_z=None, mu_z=None):
        r = torch.sqrt(x**2 + y**2)
        # phi = torch.atan2(y, x)  # Not used in calculation
        intensity = torch.exp(-0.5*(r - radius)**2 / sigma**2 - 0.5*(z - mu_z)**2 / sigma_z**2)
        return intensity


class ExponentialDisk3D(Module):
    """
    A 3D exponential disk intensity profile model, similar to KinMS.sb_profs.expdisk.
    Parameters:
    - I0: Central intensity
    - scale: Radial scale length
    - sigma_z: Vertical dispersion
    - mu_z: Vertical mean offset
    """

    def __init__(self):
        super().__init__()

        # Register parameters
        self.I0 = Param("I0", None)
        self.scale = Param("scale", None)
        self.sigma_z = Param("sigma_z", None)
        self.mu_z = Param("mu_z", None)

    @forward
    def brightness(self, x, y, z, I0=None, scale=None, sigma_z=None, mu_z=None):
        """
        Computes the intensity at positions (x, y, z).

        Parameters
        ----------
        x, y, z : Tensor
            Coordinates at which the intensity is evaluated.
        I0, scale, sigma_z, mu_z : Tensors
            Model parameters provided by caskade.
        """
        r = torch.sqrt(x**2 + y**2)
        print((I0, scale, sigma_z, mu_z))
        intensity = I0 * torch.exp(-r/scale - 0.5*(z - mu_z)**2 / sigma_z**2)
        print(intensity)
        return intensity
    
    
    
def cutoff(r, start, end, device = "cuda"):
    """
    Creates a cutoff in a surface brightness profile between two radii. 
    This is a PyTorch-compatible version of KinMS.sb_profs.cutoff of Davis et al. (2013)
    """
    
    # Convert all entries to PyTorch tensors
    if type(r) is not torch.tensor:
        r=torch.tensor(r, device = device)
    
    return ~((r>=start)&(r<end))