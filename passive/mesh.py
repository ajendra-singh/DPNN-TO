import numpy as np
import torch

def generate_points(hole_radius, x_min, y_min, x_max, y_max, nx, ny, device=None):

    # Grid
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)

    xx = X.reshape(-1, 1)
    yy = Y.reshape(-1, 1)

    # Define a mask to create a hole
    hole_center = [(x_min + x_max) / 2, (y_min + y_max) / 2]  # Center of the rectangle
    distance_to_center = np.sqrt((X - hole_center[0])**2 + (Y - hole_center[1])**2)
    mask = distance_to_center >= hole_radius
    # Apply the mask to remove points inside the hole
    xm = xx[mask.flatten()]
    ym = yy[mask.flatten()]

    # Physics-based masks
    dbc_mask = (xm == x_min)                 # fixed boundary
    load_mask = (xm == x_max) & (ym == y_min)  # point load

    # Torch tensors
    x = torch.tensor(xm, dtype=torch.float32, requires_grad=True, device=device)
    y = torch.tensor(ym, dtype=torch.float32, requires_grad=True, device=device)

    return x, y, X, Y, dbc_mask, load_mask, mask