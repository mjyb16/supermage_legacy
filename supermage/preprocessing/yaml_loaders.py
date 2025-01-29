import numpy as np
import torch
import caustics
import caustics.cosmology
import caustics.lenses
from caustics.light.pixelated import Pixelated
import supermage.preprocessing.yashar_pregridded
from supermage.utils.uv_utils import generate_binned_data, generate_binned_counts, generate_uv_mask, gaussian_pb
from supermage.utils.coord_utils import pixel_size_background
import yaml

def generate_meshgrid(grid_extent, gal_res, device = "cuda"):
    """
    Generates grid for simulation
    grid_extent: 2*r_galaxy usually
    """
    return torch.meshgrid(torch.linspace(-grid_extent, grid_extent, gal_res, device = device), torch.linspace(-grid_extent, grid_extent, gal_res, device = device), torch.linspace(-grid_extent, grid_extent, gal_res, device = device), indexing = "ij")


def generate_cube_z_labels(velocity_min, velocity_max, velocity_res, gal_res, device = "cuda"):
    """
    Generates labels for Keops convolution
    """
    return torch.linspace(velocity_min, velocity_max, velocity_res, device = device) * torch.ones((gal_res, gal_res, velocity_res), device = device)



def yashar_gridded_vis_loader(yaml_file):
    with open(yaml_file, 'r') as file:
        parameter_dict = yaml.safe_load(file)
    uv_func = getattr(supermage.gridding.yashar_pregridded, parameter_dict["uv_func"])
    uv_params_list = []
    for x, obj in parameter_dict["uv_params"].items():
        uv_params_list += [obj]
    return uv_func(*uv_params_list)

#############################################################################################################################################################
def simulator_init_mge(yaml_file, deltam, device="cuda"):
    with open(yaml_file, 'r') as file:
        parameter_dict = yaml.safe_load(file)
    
    pixel_scale_back = pixel_size_background(z_background=parameter_dict["background_params"]["redshift_background"], r_cube=parameter_dict["background_params"]["cube_radius"], gal_res=parameter_dict["background_params"]["galaxy_resolution"], H0=parameter_dict["cosmo_params"]["H0"], Om0=parameter_dict["cosmo_params"]["Om0"], Tcmb0=parameter_dict["cosmo_params"]["Tcmb0"])
    
    meshgrid = generate_meshgrid(parameter_dict["background_params"]["cube_radius"], 
                                 parameter_dict["background_params"]["galaxy_resolution"], 
                                 device=device)
    
    cube_z_labels = generate_cube_z_labels(parameter_dict["background_params"]["velocity_min"], 
                                           parameter_dict["background_params"]["velocity_max"], 
                                           parameter_dict["background_params"]["velocity_resolution"], 
                                           parameter_dict["background_params"]["galaxy_resolution"],
                                           device=device)
    
    background_dict = parameter_dict["background_params"].copy()
    intensity_params = torch.tensor([
        background_dict["intensity_params"]["sigma"],
        background_dict["intensity_params"]["radius"],
        background_dict["intensity_params"]["sigma_z"],
        background_dict["intensity_params"]["mu_z"]
    ], device=device)
    
    stellar_mass_params = {
        "surf": torch.tensor(np.load(background_dict["stellar_mass_params"]["surf"]), device=device),
        "sigma": torch.tensor(np.load(background_dict["stellar_mass_params"]["sigma"]), device=device),
        "qobs": torch.tensor(np.load(background_dict["stellar_mass_params"]["qobs"]), device=device),
        "inc": torch.tensor(background_dict["stellar_mass_params"]["inc"], device=device),
        #"dist": torch.tensor(background_dict["stellar_mass_params"]["dist"], device=device),
        "M_to_L": torch.tensor(background_dict["stellar_mass_params"]["M_to_L"], device=device)
    }
    
    del background_dict['intensity_func'], background_dict['intensity_params'], background_dict['stellar_mass_func'], background_dict['stellar_mass_params']
    background_params_tensors = {key: torch.tensor(value, device=device) for key, value in background_dict.items()}
    
    return meshgrid, cube_z_labels, intensity_params, stellar_mass_params, background_params_tensors, pixel_scale_back
    
#############################################################################################################################################################
def simulator_init_einasto(yaml_file, deltam, device = "cuda"):
    with open(yaml_file, 'r') as file:
        parameter_dict = yaml.safe_load(file)
    
    pixel_scale_back = pixel_size_background(z_background=parameter_dict["background_params"]["redshift_background"], r_cube=parameter_dict["background_params"]["cube_radius"], gal_res=parameter_dict["background_params"]["galaxy_resolution"], H0=parameter_dict["cosmo_params"]["H0"], Om0=parameter_dict["cosmo_params"]["Om0"], Tcmb0=parameter_dict["cosmo_params"]["Tcmb0"])
    
    meshgrid = generate_meshgrid(parameter_dict["background_params"]["cube_radius"], parameter_dict["background_params"]["galaxy_resolution"], device = "cuda")
    cube_z_labels = generate_cube_z_labels(parameter_dict["background_params"]["velocity_min"], parameter_dict["background_params"]["velocity_max"], parameter_dict["background_params"]["velocity_resolution"], parameter_dict["background_params"]["galaxy_resolution"])
    
    
    
    background_dict = parameter_dict["background_params"].copy()
    intensity_params = []
    for x, obj in background_dict["intensity_params"].items():
        intensity_params += [obj]
    stellar_mass_params = []
    for x, obj in background_dict["stellar_mass_params"].items():
        stellar_mass_params += [obj]
    
    del background_dict['intensity_func'], background_dict['intensity_params'], background_dict['stellar_mass_func'], background_dict['stellar_mass_params']
    background_params_tensors = {key: torch.tensor(value, device = device) for key, value in background_dict.items()}
    return meshgrid, cube_z_labels, torch.tensor(intensity_params, device = device), torch.tensor(stellar_mass_params, device = device), background_params_tensors, pixel_scale_back
    
#############################################################################################################################################################
def simulator_init_simple(yaml_file, deltam, device = "cuda"):
    with open(yaml_file, 'r') as file:
        parameter_dict = yaml.safe_load(file)
    
    pixel_scale_back = pixel_size_background(z_background=parameter_dict["background_params"]["redshift_background"], r_cube=parameter_dict["background_params"]["cube_radius"], gal_res=parameter_dict["background_params"]["galaxy_resolution"], H0=parameter_dict["cosmo_params"]["H0"], Om0=parameter_dict["cosmo_params"]["Om0"], Tcmb0=parameter_dict["cosmo_params"]["Tcmb0"])
    
    cosmology = getattr(caustics.cosmology, parameter_dict["cosmo_func"])(name="cosmo")
    cosmology.to(dtype=torch.float32, device = device)
    lens_param_list = []
    for x, obj in parameter_dict["lens_params"].items():
        lens_param_list += [obj]
    lens_list = []
    for x, obj in parameter_dict["lens_funcs"].items():
        lens_list += [getattr(caustics.lenses, obj)(cosmology, z_l = parameter_dict["lens_params"]["lens_redshift"])]
    lens = caustics.SinglePlane(lenses=lens_list, cosmology=cosmology, z_l=parameter_dict["lens_params"]["lens_redshift"])
    src = Pixelated(name="background", shape=(parameter_dict["background_params"]["galaxy_resolution"], parameter_dict["background_params"]["galaxy_resolution"]), pixelscale = pixel_scale_back, x0 = 0, y0 = 0).to(device = device)
    sim = caustics.Lens_Source(lens=lens, source=src, pixelscale=deltam, pixels_x=parameter_dict["uv_params"]["npix"], z_s=parameter_dict["background_params"]["redshift_background"]).to(device = device)
    
    meshgrid = generate_meshgrid(parameter_dict["background_params"]["cube_radius"], parameter_dict["background_params"]["galaxy_resolution"], device = "cuda")
    cube_z_labels = generate_cube_z_labels(parameter_dict["background_params"]["velocity_min"], parameter_dict["background_params"]["velocity_max"], parameter_dict["background_params"]["velocity_resolution"], parameter_dict["background_params"]["galaxy_resolution"])
    
    
    
    background_dict = parameter_dict["background_params"].copy()
    intensity_params = []
    for x, obj in background_dict["intensity_params"].items():
        intensity_params += [obj]
    stellar_mass_params = []
    for x, obj in background_dict["stellar_mass_params"].items():
        stellar_mass_params += [obj]
    
    del background_dict['intensity_func'], background_dict['intensity_params'], background_dict['stellar_mass_func'], background_dict['stellar_mass_params']
    background_params_tensors = {key: torch.tensor(value, device = device) for key, value in background_dict.items()}
    
    return sim, meshgrid, cube_z_labels, torch.tensor(intensity_params, device = device), torch.tensor(stellar_mass_params, device = device), background_params_tensors, torch.tensor(lens_param_list[1:], device = device)

# Create tensorloader for simulation parameters (returns functions and tensors containing the arguments for those functions + any functions being passed as arguments, ie intensity_func and stellar_mass_func)
#For loop for intensity_func and stellar_mass_func params, [dict["parameter"]] for global engine params

