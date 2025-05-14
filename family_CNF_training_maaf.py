# family_CNF_training_maaf.py - Training CNF con MaskedAffineAutoregressive (RealNVP-like)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows import MaskedAffineAutoregressive
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


# Caricamento dati 
df = pd.read_csv(OUTPUT_DMP_WEIGHTS_CSV, header=None)
X = df.iloc[:, :-N_CONTEXT_DIMS].values.astype(np.float32)
C = df.iloc[:, -N_CONTEXT_DIMS:].values.astype(np.float32)

# Normalizzazione dei pesi 
mean_x = X.mean(axis=0)
std_x = X.std(axis=0)
std_x = np.where(std_x == 0, 1.0, std_x)
X = (X - mean_x) / std_x

# Mantieni il contesto binario senza normalizzazione
mean_c = np.zeros_like(C[0])
std_c = np.ones_like(C[0])

# Tensori
X = torch.tensor(X, dtype=torch.float32)
C = torch.tensor(C, dtype=torch.float32)

# Definizione del modello CNF 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim = X.shape[1]
base = DiagGaussian(dim)
flows = [
    MaskedAffineAutoregressive(
        features=dim,
        hidden_features=HIDDEN_FEATURES,
        context_features=N_CONTEXT_DIMS,
        num_blocks=NUM_BLOCKS,
        use_residual_blocks=True,
        dropout_probability=DROPOUT_PROB,
        activation=nn.ReLU()
    ) for _ in range(NUM_LAYERS)
]

model = ConditionalNormalizingFlow(base, flows)
model.train()

# Ottimizzazione
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
loss_hist = []
print("Training Conditional Neural Spline Flow...")
for i in tqdm(range(MAX_ITER)):
    idx = torch.randint(0, X.shape[0], (BATCH_SIZE,))
    x_batch = X[idx].to(device)
    c_batch = C[idx].to(device)

    log_prob = model.log_prob(x_batch, context=c_batch)
    loss = -log_prob.mean()

    optimizer.zero_grad()
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
    loss_hist.append(loss.item())

    if i % 500 == 0:
        print(f"[Iter {i}] Loss: {loss.item():.4f}")

# Salvataggio
torch.save({
    'model_state_dict': model.state_dict(),
    'mean_x': mean_x,
    'std_x': std_x,
    'mean_c': np.zeros(N_CONTEXT_DIMS),
    'std_c': np.ones(N_CONTEXT_DIMS)
}, MODEL_PATH)

# Plot loss 
plt.figure(figsize=(8, 5))
plt.ylim(-500, 500)  # Limita l'asse y per una visualizzazione migliore
plt.plot(loss_hist, label="Training Loss")
plt.xlabel("Iterazioni")
plt.ylabel("Negative Log-Likelihood")
plt.title("Training Loss - MAAF")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_maaf.png")
plt.show()
