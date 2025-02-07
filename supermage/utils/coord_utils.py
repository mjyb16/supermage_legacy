import torch
import numpy as np
from astropy.coordinates import Distance
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM as lCDM

def e_radius(z_lens, M, H0, Om0, Tcmb0):
    "Einstein radius in arcseconds for mass M"
    cosmo = lCDM(H0=H0, Om0=Om0, Tcmb0=Tcmb0)
    d_ang = (cosmo.comoving_distance(z_lens).to(u.m))/(1+z_lens) #Angular diameter distance in meters
    return np.sqrt((4*c.G*M / (d_ang * c.c**2)).value)*(180/np.pi)*(3600) #Einstein radius in arcsec

def fov_background(z_background, r_cube, H0, Om0, Tcmb0):
    """
    z_background: background galaxy redshift
    r_galaxy: radius of galaxy in pc
    Other args: Cosmology parameters
    """
    cosmo = lCDM(H0=H0, Om0=Om0, Tcmb0=Tcmb0)
    d_ang = cosmo.comoving_distance(z_background)/(1+z_background) #In Mpc
    return (2*(r_cube/1e6)/d_ang*(180/torch.pi)*(3600)).value #Angular size of the background cube

def pixel_size_background(z_background, r_cube, gal_res, H0, Om0, Tcmb0):
    angular_size = fov_background(z_background, r_cube, H0, Om0, Tcmb0)
    return angular_size/gal_res

def fov_lensed(z_lens, factor, M, H0, Om0, Tcmb0):
    e_r = e_radius(z_lens, M = M, H0=H0, Om0=Om0, Tcmb0=Tcmb0)
    return 2*factor*e_r

def pixel_size_lensed(z_lens, sim_res, factor, M, H0, Om0, Tcmb0):
    fov = fov_calc(z_lens, factor, M, H0, Om0, Tcmb0)
    return fov/sim_res

def arcsec_to_parsec(distance_mpc, angular_size_arcsec):
    """
    Convert angular sizes in arcseconds to distances in parsecs.
    
    Parameters:
    distance_mpc (numpy.ndarray): Distance to galaxy
    angular_size_arcsec (numpy.ndarray): Array of angular sizes in arcseconds.
    
    Returns:
    numpy.ndarray: Array of distances in parsecs.
    """
    # Convert distance from Mpc to parsecs
    distance_pc = distance_mpc * 1e6
    
    # Convert angular size from arcseconds to radians
    angular_size_rad = np.radians(angular_size_arcsec / 3600)
    
    # Calculate physical size using the small angle approximation
    physical_size_pc = distance_pc * angular_size_rad
    
    return physical_size_pc

def parsec_to_arcsec(distance_mpc, physical_size_pc):
    """
    Convert physical sizes in parsecs to angular sizes in arcseconds.
    
    Parameters:
    -----------
    distance_mpc : float or np.ndarray
        Distance to the galaxy in megaparsecs (Mpc).
    physical_size_pc : float or np.ndarray
        Physical sizes in parsecs.

    Returns:
    --------
    np.ndarray
        Angular sizes in arcseconds.
    """
    # Convert distance from Mpc to parsecs
    distance_pc = distance_mpc * 1e6
    
    # Using the small-angle approximation:
    # physical_size_pc = distance_pc * angular_size_rad
    # => angular_size_rad = physical_size_pc / distance_pc
    angular_size_rad = physical_size_pc / distance_pc
    
    # Convert from radians to arcseconds
    # 1 radian = (180/pi) degrees, and 1 degree = 3600 arcseconds
    # so 1 radian = 206265 arcseconds
    angular_size_arcsec = np.degrees(angular_size_rad) * 3600
    
    return angular_size_arcsec