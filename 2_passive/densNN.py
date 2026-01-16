import torch
import torch.nn as nn
import numpy as np

# ------Density network operating on Fourier-encoded coordinates----

class densNN(nn.Module):
    def __init__(self, in_dim, depth, width):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, width))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

        self.feature_net = nn.ModuleList(layers)
        self.density_layer = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.density_layer.weight)
        nn.init.zeros_(self.density_layer.bias)

        self.norm = nn.BatchNorm1d(width)
        self.act = nn.LeakyReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, z):
        for layer in self.feature_net:
            z = self.act(self.norm(layer(z)))
        return self.out_act(self.density_layer(z))

def fourier_map(x, y, proj):
    """
    Fourier feature mapping for 2D inputs.
    """
    coords = torch.cat([x, y], dim=1)
    phase = 2 * np.pi * (coords @ proj)
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)