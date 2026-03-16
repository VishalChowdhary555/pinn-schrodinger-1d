import torch
import torch.nn as nn
from config import X_MIN, X_MAX

class PINNWaveFunction(nn.Module):
    def __init__(self, hidden_dim=64, n_hidden=4):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.energy = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward_raw(self, x):
        return self.net(x)

    def forward(self, x, potential_type="harmonic"):
        raw = self.forward_raw(x)

        if potential_type == "box":
            xi = (x - X_MIN) / (X_MAX - X_MIN)
            envelope = xi * (1.0 - xi)
            return envelope * raw

        if potential_type == "harmonic":
            envelope = torch.exp(-0.1 * x**2)
            return envelope * raw

        if potential_type == "finite_well":
            envelope = torch.exp(-0.05 * x**2)
            return envelope * raw

        return raw
