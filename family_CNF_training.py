# conditional_nspline_circular_train.py adattato per training con contesto binario (4 bit)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.neural_spline import CircularAutoregressiveRationalQuadraticSpline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Dataset
df = pd.read_csv(OUTPUT_DMP_WEIGHTS_CSV, header=None)
n_context_dims = N_CONTEXT_DIMS  
C = df.iloc[:, -n_context_dims:].values.astype(np.float32)  
X = df.iloc[:, :-n_context_dims].values.astype(np.float32)  #

# Normalizzazione
mean_x = X.mean(axis=0)
std_x = X.std(axis=0)
std_x = np.where(std_x == 0, 1.0, std_x)
X = (X - mean_x) / std_x

# Mantieni il contesto binario senza normalizzazione
mean_c = np.zeros_like(C[0])
std_c = np.ones_like(C[0])

# Normlaizzazione contesto geometrico
# mean_c = C.mean(axis=0)
# std_c = C.std(axis=0)
# std_c = np.where(std_c == 0, 1.0, std_c)  # evita divisione per 0
# C = (C - mean_c) / std_c

# Tensori
X = torch.tensor(X, dtype=torch.float32)
C = torch.tensor(C, dtype=torch.float32)

# Modello 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dim = X.shape[1]
context_dim = C.shape[1]
base = DiagGaussian(dim)

flows = []
for _ in range(NUM_LAYERS):
    flows.append(
        CircularAutoregressiveRationalQuadraticSpline(
            num_input_channels=dim,
            num_blocks=NUM_BLOCKS,
            num_hidden_channels=NUM_HIDDEN_CHANNELS,
            ind_circ=[],
            num_context_channels=context_dim,
            num_bins=NUM_BINS,
            tail_bound=TAIL_BOUND,
            activation=nn.ReLU,
            dropout_probability=DROPOUT_PROB,
            permute_mask=PERMUTE_MASK,
            init_identity=INIT_IDENTITY
        )
    )

model = ConditionalNormalizingFlow(base, flows).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Training 
batch_size = BATCH_SIZE
max_iter = MAX_ITER
loss_hist = []

print("Training Conditional Neural Spline Flow...")
for i in tqdm(range(max_iter)):
    idx = torch.randint(0, X.shape[0], (batch_size,))
    x_batch = X[idx].to(device)
    c_batch = C[idx].to(device)  # no rumore 
    c_batch = C[idx].clone().to(device) #rumore

    # AGGIUNTA: leggero rumore al contesto per migliorare generalizzazione
    # noise_eps = 0.02
    # c_batch += torch.empty_like(c_batch).uniform_(-noise_eps, noise_eps)
    # c_batch = torch.clamp(c_batch, -3.0, 3.0)  # contesto normalizzato, quindi valori in range sensato

    log_prob = model.log_prob(x_batch, context=c_batch)
    loss = -log_prob.mean()

    optimizer.zero_grad()
    if not torch.isnan(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    loss_hist.append(loss.item())

    if i % 500 == 0:
        print(f"[Iter {i}] Loss: {loss.item():.4f}")

# Salvataggio modello
torch.save({
    'model_state_dict': model.state_dict(),
    'mean_x': mean_x,
    'std_x': std_x,
    'mean_c': mean_c,
    'std_c': std_c
}, MODEL_PATH)

# Plot loss 
plt.figure(figsize=(8, 5))
plt.plot(loss_hist, label="Training Loss")
plt.xlabel("Iterazione")
plt.ylabel("- Log Likelihood")
plt.title("Curva Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
