import numpy as np
import torch
import torch.optim as optim

from config import (
    DEVICE,
    EPOCHS,
    LR,
    LAMBDA_BC,
    LAMBDA_NORM,
    LAMBDA_ORTH,
    N_COLLOCATION,
    N_PLOT,
    N_STATES,
    POTENTIAL_TYPE,
    X_MAX,
    X_MIN,
)
from model import PINNWaveFunction
from plot_results import save_plots
from utils import (
    bc_loss,
    normalization_loss,
    normalize_wavefunction,
    orthogonality_loss,
    schrodinger_residual,
)

torch.manual_seed(42)
np.random.seed(42)

def train_one_state(state_index, previous_states, potential_type="harmonic"):
    model = PINNWaveFunction(hidden_dim=64, n_hidden=4).to(DEVICE)

    if potential_type == "box":
        L = X_MAX - X_MIN
        guess = (np.pi**2 * (state_index + 1)**2) / (2 * L**2)
    elif potential_type == "harmonic":
        guess = state_index + 0.5
    elif potential_type == "finite_well":
        guess = 0.5 + state_index
    else:
        guess = 1.0

    with torch.no_grad():
        model.energy[:] = torch.tensor([guess], dtype=torch.float32, device=DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    x_train = torch.linspace(X_MIN, X_MAX, N_COLLOCATION, device=DEVICE).view(-1, 1)

    loss_history = []
    energy_history = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        residual, psi = schrodinger_residual(model, x_train, potential_type)
        psi_normed = normalize_wavefunction(psi, x_train)

        loss_pde = torch.mean(residual**2)
        loss_bc = bc_loss(model, potential_type)
        loss_norm = normalization_loss(psi, x_train)
        loss_orth = orthogonality_loss(psi_normed, x_train, previous_states)

        loss = (
            loss_pde
            + LAMBDA_BC * loss_bc
            + LAMBDA_NORM * loss_norm
            + LAMBDA_ORTH * loss_orth
        )

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        energy_history.append(model.energy.item())

        if epoch % 500 == 0 or epoch == EPOCHS - 1:
            print(
                f"State {state_index}, Epoch {epoch:5d}, "
                f"Loss={loss.item():.6e}, "
                f"E={model.energy.item():.6f}, "
                f"PDE={loss_pde.item():.3e}, "
                f"BC={loss_bc.item():.3e}, "
                f"Norm={loss_norm.item():.3e}, "
                f"Orth={loss_orth.item():.3e}"
            )

    x_dense = torch.linspace(X_MIN, X_MAX, N_PLOT, device=DEVICE).view(-1, 1)
    with torch.enable_grad():
        psi_dense = model(x_dense, potential_type=potential_type)
    psi_dense = normalize_wavefunction(psi_dense, x_dense).detach()

    return model, x_dense.detach(), psi_dense, np.array(loss_history), np.array(energy_history)

def main():
    print(f"Using device: {DEVICE}")
    previous_states = []
    trained_models = []
    all_x = None
    all_psi = []
    all_E = []
    all_loss_histories = []
    all_energy_histories = []

    for n in range(N_STATES):
        print(f"\nTraining state {n} for potential: {POTENTIAL_TYPE}")
        model, x_dense, psi_dense, loss_hist, energy_hist = train_one_state(
            n, previous_states, potential_type=POTENTIAL_TYPE
        )

        previous_states.append(psi_dense)
        trained_models.append(model)
        all_x = x_dense
        all_psi.append(psi_dense.cpu().numpy().flatten())
        all_E.append(model.energy.item())
        all_loss_histories.append(loss_hist)
        all_energy_histories.append(energy_hist)

    print("\nFinal learned energies:")
    for i, e in enumerate(all_E):
        print(f"State {i}: E = {e:.6f}")

    save_plots(
        all_x=all_x,
        all_psi=all_psi,
        all_E=all_E,
        all_loss_histories=all_loss_histories,
        all_energy_histories=all_energy_histories,
        potential_type=POTENTIAL_TYPE,
        out_dir="results",
    )

if __name__ == "__main__":
    main()
