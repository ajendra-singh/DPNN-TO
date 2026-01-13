import numpy as np
import torch

def generate_points(x_min, y_min, x_max, y_max, nx, ny, device=None):
    """
    Generates collocation points and physics-based masks
    for a 2D cantilever beam with a point load at (x_max, y_min).

    Returns
    -------
    x, y        : torch.Tensor, shape (N, 1)
    dbc_mask    : np.ndarray (bool)
    load_mask   : np.ndarray (bool)
    """
    # Grid
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)

    xx = X.reshape(-1, 1)
    yy = Y.reshape(-1, 1)

    # Physics-based masks
    dbc_mask = (xx == x_min)                 # fixed boundary
    load_mask = (xx == x_max) & (yy == y_min)  # point load

    # Torch tensors
    x = torch.tensor(xx, dtype=torch.float32, requires_grad=True, device=device)
    y = torch.tensor(yy, dtype=torch.float32, requires_grad=True, device=device)

    return x, y, X, Y, dbc_mask, load_mask