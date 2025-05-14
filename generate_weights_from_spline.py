import torch
import numpy as np
import pandas as pd

from normflows import NormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.neural_spline import CoupledRationalQuadraticSpline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Carica modello allenato e setta parametri (devono essere uguali a quelli usati nel training), 
# Ricarica mean/std per denormalizzare
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
mean = checkpoint['mean']  
std = checkpoint['std']
dim = len(mean)
num_layers = NUM_LAYERS

# Ricrea modello
base = DiagGaussian(dim)
flows = []
for i in range(num_layers):
    flows.append(
        CoupledRationalQuadraticSpline(
            num_input_channels=dim,
            num_blocks=NUM_BLOCKS,
            num_hidden_channels=NUM_HIDDEN_CHANNELS,
            num_bins=NUM_BINS,
            tails="linear",
            tail_bound=TAIL_BOUND,
            activation=torch.nn.ReLU,
            dropout_probability=DROPOUT_PROB,
            reverse_mask=bool(i % 2)
        )
    )
model = NormalizingFlow(base, flows)
model.load_state_dict(checkpoint['model_state_dict']) #
model.eval()


# Generazione pesi
with torch.no_grad():
    z = model.sample(N_SAMPLES)[0].cpu().numpy()
    weights = z * std + mean  # denormalizzazione

# Salvataggio CSV
df = pd.DataFrame(weights, columns=[f"w_{i}" for i in range(dim)])
df.to_csv(GENERATED_WEIGHTS_CSV, index=False)
print("Pesi generati salvati")
