import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from normflows import NormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.neural_spline import CoupledRationalQuadraticSpline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Carica dataset
df = pd.read_csv("dmp_weights_xy.csv", header=None) # skiprows=1, per non avere l'intestazione
X = df.values.astype(np.float32)

# Normalizzazione (mean-std)
# mean = X.mean(axis=0)
# std = X.std(axis=0)
# Evita divisioni per zero durante la normalizzazione
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0  # Previene NaN

X = (X - mean) / std
X = torch.tensor(X, dtype=torch.float32)

# Verifica NaN
if np.isnan(X).any():
    print("Attenzione: ci sono NaN in X!")

dim = X.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = DiagGaussian(dim)

# Costruzione flow
num_layers = NUM_LAYERS
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

model = NormalizingFlow(base, flows).to(device)

# Ottimizzatore
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = BATCH_SIZE
max_iter = MAX_ITER
loss_hist = []

print("Training Coupled Rational Quadratic Spline...")

# Training loop
for i in tqdm(range(max_iter)):
    idx = torch.randint(0, X.shape[0], (batch_size,))
    x_batch = X[idx].to(device)

    loss = -model.log_prob(x_batch).mean()
    optimizer.zero_grad()
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
    loss_hist.append(loss.item())

# Plot della loss
plt.figure(figsize=(8, 5))
plt.plot(loss_hist)
plt.title("Training Loss (-log likelihood)")
plt.xlabel("Iterazione")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Campioni generati + denormalizzazione
model.eval()
with torch.no_grad():
    samples = model.sample(1000)[0].cpu().numpy()
    samples = samples * std + mean

# PCA per visualizzazione
from sklearn.decomposition import PCA
pca = PCA(n_components=2,whiten=True)
X_pca = pca.fit_transform(X.numpy())  # X è normalizzato!
samples_pca = pca.transform(samples)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolor='k')
plt.title("Distribuzione Reale (PCA)")

plt.subplot(1, 2, 2)
plt.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.5, edgecolor='k')
plt.title("Distribuzione Generata da Coupled RQ Spline (PCA)")

plt.tight_layout()
plt.show()

# Salva il modello e le statistiche per normalizzazione
torch.save({
    'model_state_dict': model.state_dict(),
    'mean': mean,
    'std': std
}, MODEL_PATH)

print("Modello salvato in MODEL_PATH")
