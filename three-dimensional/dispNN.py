import torch
import torch.nn as nn
import numpy as np

class SinLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=30.0, first=False, bias=True):
        super().__init__()
        self.omega = omega
        self.first = first
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self._init_weights(in_dim)

    def _init_weights(self, in_dim):
        with torch.no_grad():
            if self.first:
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                bound = np.sqrt(6.0 / in_dim) / self.omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

# -------------------------Displacement network-------------------
class dispNN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, omega_in=10.0, omega_hidden=20.0):
        super().__init__()
        layers = [SinLayer(input_dim, hidden_dim, omega=omega_in, first=True)]
        for _ in range(num_layers - 2):
            layers.append(SinLayer(hidden_dim, hidden_dim, omega=omega_hidden))
        final = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega_hidden
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x, y, z):
        coords = torch.cat([x, y, z], dim=1)
        return self.network(coords)