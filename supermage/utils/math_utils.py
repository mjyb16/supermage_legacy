import torch

def _func_R_x(theta, device = "cuda"):
    result = torch.zeros((3, 3), device = device)
    result[0, 0] = 1
    result[1, 1] = torch.cos(theta)
    result[1, 2] = -torch.sin(theta)
    result[2, 1] = torch.sin(theta)
    result[2, 2] = torch.cos(theta)
    return result

def _func_R_y(psi, device = "cuda"):
    result = torch.zeros((3, 3), device = device)
    result[0, 0] = torch.cos(psi)
    result[0, 2] = torch.sin(psi)
    result[1, 1] = 1
    result[2, 0] = -torch.sin(psi)
    result[2, 2] = torch.cos(psi)
    return result

def _func_R_z(phi, device = "cuda"):
    result = torch.zeros((3, 3), device = device)
    result[0, 0] = torch.cos(phi)
    result[0, 1] = -torch.sin(phi)
    result[1, 0] = torch.sin(phi)
    result[1, 1] = torch.cos(phi)
    result[2, 2] = 1
    return result

def DoRotation(x, y, z, theta, phi, device = "cuda"):
    """Rotate a meshgrid. Theta is inclination, phi is rotation in sky plane"""
    shape = x.shape
    # Clockwise, 2D rotation matrix
    RotMatrix = torch.matmul(_func_R_y(theta, device = device), _func_R_z(phi, device = device))
    #return torch.einsum('ji, mni -> jmn', RotMatrix, torch.dstack([x, y, z]))
    
    mult = torch.inner(RotMatrix, torch.dstack([x.ravel(),y.ravel(), z.ravel()]))
    xrot = mult[0,:].reshape(shape)
    yrot = mult[1,:].reshape(shape)
    zrot = mult[2,:].reshape(shape)
    return xrot, yrot, zrot


def DoRotationT(x, y, z, theta, phi, device = "cuda"):
    """Multiply a meshgrid by the transpose of rotation matrix. Theta is inclination, phi is rotation in sky plane"""
    shape = x.shape
    # Clockwise, 2D rotation matrix
    RotMatrix = torch.matmul(_func_R_y(theta, device = device), _func_R_z(phi, device = device)).T
    #return torch.einsum('ji, mni -> jmn', RotMatrix, torch.dstack([x, y, z]))
    ### NEED TO UNDERSTAND WHERE NEED TO CONVERT TO FLOAT PRECISION IS COMING FROM (WHY IS X DOUBLE)
    mult = torch.inner(RotMatrix, torch.dstack([x.ravel(),y.ravel(), z.ravel()]).float())
    xrot = mult[0,:].reshape(shape)
    yrot = mult[1,:].reshape(shape)
    zrot = mult[2,:].reshape(shape)
    return xrot, yrot, zrot