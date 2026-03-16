import torch
from config import DEVICE, X_MIN, X_MAX
from potentials import potential

def second_derivative(y, x):
    dy_dx = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2y_dx2 = torch.autograd.grad(
        dy_dx,
        x,
        grad_outputs=torch.ones_like(dy_dx),
        create_graph=True,
        retain_graph=True,
    )[0]
    return d2y_dx2

def trapz_torch(y, x):
    return torch.trapz(y.squeeze(), x.squeeze())

def normalize_wavefunction(psi, x):
    norm = torch.sqrt(trapz_torch(psi**2, x) + 1e-12)
    return psi / norm

def bc_loss(model, potential_type):
    xb = torch.tensor([[X_MIN], [X_MAX]], dtype=torch.float32, device=DEVICE)
    psi_b = model(xb, potential_type=potential_type)
    return torch.mean(psi_b**2)

def normalization_loss(psi, x):
    return (trapz_torch(psi**2, x) - 1.0) ** 2

def orthogonality_loss(psi, x, previous_states):
    if len(previous_states) == 0:
        return torch.tensor(0.0, device=DEVICE)
    loss = 0.0
    for psi_prev in previous_states:
        overlap = trapz_torch(psi * psi_prev, x)
        loss = loss + overlap**2
    return loss

def schrodinger_residual(model, x, potential_type):
    x.requires_grad_(True)
    psi = model(x, potential_type=potential_type)
    psi_xx = second_derivative(psi, x)
    v = potential(x, potential_type=potential_type)
    e = model.energy
    residual = -0.5 * psi_xx + v * psi - e * psi
    return residual, psi
