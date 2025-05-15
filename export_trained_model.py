import torch
import numpy as np
import torch.nn as nn
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.neural_spline import CircularAutoregressiveRationalQuadraticSpline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

INPUT_DIM = 40

# === Ricrea il modello ===
base = DiagGaussian(INPUT_DIM)
flows = []
for _ in range(NUM_LAYERS):
    flows.append(
        CircularAutoregressiveRationalQuadraticSpline(
            num_input_channels=INPUT_DIM,
            num_blocks=NUM_BLOCKS,
            num_hidden_channels=NUM_HIDDEN_CHANNELS,
            ind_circ=[],
            num_context_channels=N_CONTEXT_DIMS,
            num_bins=NUM_BINS,
            tail_bound=TAIL_BOUND,
            activation=nn.ReLU,
            dropout_probability=DROPOUT_PROB,
            permute_mask=PERMUTE_MASK,
            init_identity=INIT_IDENTITY
        )
    )

model = ConditionalNormalizingFlow(base, flows)

# === Carica il checkpoint originale ===
checkpoint = torch.load("trained_CNF_family_final_continuous.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

mean_x = checkpoint["mean_x"]
std_x = checkpoint["std_x"]
mean_c = checkpoint["mean_c"]
std_c = checkpoint["std_c"]

# === Salva in formato portabile ===
torch.save(model.state_dict(), "model_weights.pt")
np.savez("context_stats.npz", mean_x=mean_x, std_x=std_x, mean_c=mean_c, std_c=std_c)

print("Esportazione completata: model_weights.pt + context_stats.npz")