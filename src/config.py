import torch

# Choose potential: "box", "harmonic", or "finite_well"
POTENTIAL_TYPE = "harmonic"

# Domain
X_MIN = -5.0
X_MAX = 5.0
N_COLLOCATION = 400
N_PLOT = 1000

# Number of eigenstates
N_STATES = 3

# Training
EPOCHS = 5000
LR = 1e-3

# Loss weights
LAMBDA_BC = 10.0
LAMBDA_NORM = 10.0
LAMBDA_ORTH = 20.0

# Finite well parameters
V0 = 5.0
A = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
