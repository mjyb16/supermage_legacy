import numpy as np
from supermage.utils.uv_utils import generate_binned_data, generate_binned_counts, generate_uv_mask, generate_pb
from supermage.utils.coord_utils import pixel_size_background


def dataloader_var_from_bin(u_file, v_file, sigma_squared_inv_file, npix, paduv, frequencies_file, dish_diameter):
    data_u = np.fromfile(u_file)
    data_v = np.fromfile(v_file)
    data_var = 1/np.fromfile(sigma_squared_inv_file)

    var = data_var[0::2] + 1j*data_var[1::2]
    mask_ints, deltal, deltam = generate_uv_mask(data_u, data_v, shape=(npix, npix), pad_uv=paduv)
    mask = mask_ints > 0
    fov = deltam * npix

    # Binned variances
    binned_var_real = generate_binned_data(data_u, data_v, var.real, nyquist=False, shape=(npix, npix), pad_uv=paduv)
    binned_var_imag = generate_binned_data(data_u, data_v, var.imag, nyquist=False, shape=(npix, npix), pad_uv=paduv)
    binned_counts = generate_binned_counts(data_u, data_v, nyquist=False, shape=(npix, npix), pad_uv=paduv)
    binned_var_real = binned_var_real / binned_counts
    binned_var_imag = binned_var_imag / binned_counts
    binned_var_real = binned_var_real[mask]
    binned_var_imag = binned_var_imag[mask]

    frequencies = np.load(frequencies_file)
    primary_beams = np.zeros((frequencies.shape[0], npix, npix))
    for i in range(frequencies.shape[0]):
        primary_beams[i], _ = generate_pb(diameter=dish_diameter, freq=frequencies[i], shape=(npix, npix), deltal=deltal)

    return deltam, fov, binned_var_real, binned_var_imag, mask, primary_beams