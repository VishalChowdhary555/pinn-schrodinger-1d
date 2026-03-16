import torch
from config import A, V0

def potential(x, potential_type="harmonic"):
    if potential_type == "box":
        return torch.zeros_like(x)

    if potential_type == "harmonic":
        return 0.5 * x**2

    if potential_type == "finite_well":
        return torch.where(
            torch.abs(x) <= A / 2,
            torch.zeros_like(x),
            V0 * torch.ones_like(x),
        )

    raise ValueError(f"Unknown potential type: {potential_type}")
