
# PINN Schrödinger Solver (1D)

Physics-Informed Neural Network (PINN) implementation for solving the **1D time-independent Schrödinger equation** using **PyTorch**.

This project demonstrates how **deep learning can be used as a physics solver** by embedding the governing differential equation directly into the neural network training process.

Instead of learning from labeled data, the model learns solutions purely from **physical constraints and differential equation residuals**.

---

# Project Overview

The goal of this project is to approximate solutions to the **time-independent Schrödinger equation**

-1/2 d²ψ/dx² + V(x)ψ(x) = Eψ(x)

where

| Symbol | Meaning |
|------|------|
| ψ(x) | Quantum wavefunction |
| V(x) | Potential energy |
| E | Energy eigenvalue |

The neural network simultaneously learns:

- the **wavefunction ψ(x)**
- the **energy eigenvalue E**

using automatic differentiation and physics-based loss functions.

---

# Physics-Informed Neural Networks (PINNs)

Traditional machine learning models require labeled data.  
Physics-Informed Neural Networks instead incorporate **physical laws directly into the loss function**.

For the Schrödinger equation:

1. The neural network predicts ψ(x)
2. Automatic differentiation computes ψ''(x)
3. The Schrödinger equation residual is minimized

The loss function includes:

- PDE residual loss
- Boundary condition loss
- Wavefunction normalization
- Orthogonality constraint between eigenstates

---

# Quantum Systems Implemented

## 1. Particle in a Box

Potential:

V(x) = 0

Boundary conditions:

ψ(0) = ψ(L) = 0

Analytical energies:

Eₙ = n²π² / (2L²)

Expected wavefunctions:

ψₙ(x) = √(2/L) sin(nπx/L)

The PINN reproduces:

- sinusoidal eigenfunctions
- quantized energy spectrum
- correct boundary conditions

---

## 2. Harmonic Oscillator

Potential:

V(x) = 1/2 x²

Analytical energies:

Eₙ = n + 1/2

Expected eigenstates:

- Ground state → Gaussian
- First excited → one node
- Second excited → two nodes

The PINN successfully learns:

- correct nodal structure
- symmetric and antisymmetric states
- accurate energy eigenvalues

---

## 3. Finite Potential Well

Potential:

V(x) = 0  for |x| ≤ a/2  
V(x) = V₀ for |x| > a/2

Physical effects captured:

- bound states
- tunneling into forbidden regions
- exponential decay outside the well

This system demonstrates the PINN’s ability to learn **quantum tunneling behavior**.

---

# Model Architecture

The wavefunction is represented using a fully connected neural network.

Input: x

Fully Connected Layers  
Tanh activation  

Hidden Layers (4–6 layers typical)

Output:

ψ(x)

Additionally:

- The **energy eigenvalue E** is implemented as a **trainable parameter**
- Envelope functions enforce physical boundary behavior

---

# Repository Structure

pinn-schrodinger-1d

README.md  
requirements.txt  
LICENSE  
.gitignore  

notebook/  
 pinn_schrodinger_notebook.ipynb  

src/  
 config.py  
 model.py  
 potentials.py  
 utils.py  
 train.py  
 plot_results.py  

results/
---

# Running the Project

Run the training script:

python src/train.py

By default the project trains on the **harmonic oscillator**.

To change the potential edit:

src/config.py

Example:

POTENTIAL_TYPE = "box"

or

POTENTIAL_TYPE = "finite_well"

---

# Example Results

Typical predicted energy levels:

| System | State | Expected | PINN |
|------|------|------|------|
| Harmonic Oscillator | n=0 | 0.5 | ~0.503 |
| Harmonic Oscillator | n=1 | 1.5 | ~1.49 |
| Harmonic Oscillator | n=2 | 2.5 | ~2.51 |

Error is typically **< 1%** for well-trained models.

Plots generated:

- wavefunctions
- probability densities
- energy levels
- training loss curves
- energy convergence

---

# Key Concepts Demonstrated

### Quantum Mechanics

- Schrödinger equation
- eigenvalue problems
- wavefunction normalization
- orthogonality of eigenstates
- quantum tunneling

### Machine Learning

- Physics-informed neural networks
- automatic differentiation
- neural PDE solvers
- neural eigenvalue problems

### Scientific Computing

- differential equation residual minimization
- mesh-free PDE solving
- physics-guided optimization

---

# Limitations

PINNs are elegant but not always the most efficient solver.

| Method | Speed | Accuracy |
|------|------|------|
| Finite Difference | Very Fast | High |
| Spectral Methods | Very Fast | Very High |
| PINNs | Slower | Good |

However, PINNs become powerful for:

- high-dimensional PDEs
- irregular domains
- inverse problems
- hybrid physics + data models

---

# Future Improvements

Possible extensions:

Physics Improvements

- analytical solution comparison
- fidelity / overlap metrics
- parity constraints
- adaptive collocation sampling

ML Improvements

- deeper neural architectures
- spectral neural networks
- transformer-based PDE solvers

Quantum Applications

- time-dependent Schrödinger equation
- hydrogen atom PINN
- quantum scattering
- multi-particle wavefunctions

---

# License

MIT License
