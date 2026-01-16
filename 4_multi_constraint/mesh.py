import numpy as np
import torch

def generate_points(thickness, x_min, y_min, x_max, y_max, nx, ny, device=None):
    # Grid
    xs = np.linspace(x_min, x_max, nx+1)
    ys = np.linspace(y_min, y_max, ny+1)
    X, Y = np.meshgrid(xs, ys)

    xx = X.reshape(-1, 1)
    yy = Y.reshape(-1, 1)

    # Define a mask to remove rectangle 
    maskX = xx<=thickness 
    maskY = yy<=thickness
    mask = np.any(np.array([maskX, maskY]),axis = 0)
    
    xx = xx[mask.flatten()]
    yy = yy[mask.flatten()]

    # Physics-based masks
    dbc_mask = (yy == y_max)                 # fixed boundary
    # load_mask = (xx == x_max) & (yy == thickness)  # point load
    load_mask = (xx >= x_max-0.05) & (yy == thickness)  # distributed load

    # Torch tensors
    x = torch.tensor(xx, dtype=torch.float32, requires_grad=True, device=device)
    y = torch.tensor(yy, dtype=torch.float32, requires_grad=True, device=device)

    return x, y, X, Y, dbc_mask, load_mask, mask