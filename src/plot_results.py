from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from potentials import potential

def save_plots(all_x, all_psi, all_E, all_loss_histories, all_energy_histories, potential_type, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_np = all_x.cpu().numpy().flatten()
    v_np = potential(all_x, potential_type).detach().cpu().numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(x_np, v_np, label="V(x)", linewidth=2)
    for i, psi_np in enumerate(all_psi):
        plt.plot(x_np, psi_np + all_E[i], label=f"psi_{i}(x) + E_{i}")
        plt.axhline(all_E[i], linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("Energy / Wavefunction")
    plt.title(f"PINN Schrödinger Solutions: {potential_type}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "wavefunctions.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, psi_np in enumerate(all_psi):
        plt.plot(x_np, psi_np**2, label=f"|psi_{i}(x)|^2")
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(f"Probability Densities: {potential_type}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "probability_densities.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    for i, e in enumerate(all_E):
        plt.hlines(e, xmin=0, xmax=1, linewidth=2, label=f"n={i}, E={e:.4f}")
    plt.xlim(0, 1)
    plt.xticks([])
    plt.ylabel("Energy")
    plt.title(f"Energy Levels: {potential_type}")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "energy_levels.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, hist in enumerate(all_loss_histories):
        plt.plot(hist, label=f"State {i}")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, hist in enumerate(all_energy_histories):
        plt.plot(hist, label=f"State {i}")
    plt.xlabel("Epoch")
    plt.ylabel("Energy estimate")
    plt.title("Energy Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "energy_convergence.png", dpi=200)
    plt.close()
