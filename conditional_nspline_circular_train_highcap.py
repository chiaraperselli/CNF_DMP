# Codice per allenare un Conditional Neural Spline Flow con contesto, 
# dato una distribuzione di pesi e contesto;
# salva il modello allenato e visualizza i campioni generati rispetto alla distribuzione reale.

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

# Carica dataset 
df = pd.read_csv("300_traj_split/train_dataset.csv", header=None)
n_context_dims = 4
C = df.iloc[:, -n_context_dims:].values.astype(np.float32)  # ultime colonne → contesto
X = df.iloc[:, :-n_context_dims].values.astype(np.float32)  # prime colonne → pesi

# Normalizzazione dei pesi
mean_x = X.mean(axis=0)
std_x = X.std(axis=0)
X = (X - mean_x) / std_x

# Normalizzazione del contesto
mean_c = C.mean(axis=0)
std_c = C.std(axis=0)
C = (C - mean_c) / std_c

# Conversione in tensori
X = torch.tensor(X, dtype=torch.float32)
C = torch.tensor(C, dtype=torch.float32)

# Usa GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parametri modello
dim = X.shape[1]
context_dim = C.shape[1]

# Distribuzione base
base = DiagGaussian(dim)
num_layers = 10
flows = []
for _ in range(num_layers):
    flows.append(
        CircularAutoregressiveRationalQuadraticSpline(
            num_input_channels=dim, # dimensione dei pesi
            num_blocks=2,
            num_hidden_channels=128,
            ind_circ=[],# indici delle dimensioni circolari (in questo caso non sono usati)
            num_context_channels=context_dim,#qui il contesto viene passato al modello
            num_bins=12,# numero di bin (segmenti) per il Rational Quadratic Spline
            tail_bound=3.0,# bound per i valori estremi
            activation=nn.ReLU, 
            dropout_probability=0.05,
            learning_rate = 5e-4,
            permute_mask=True,
            init_identity=True
        )
    )

model = ConditionalNormalizingFlow(base, flows).to(device) # crea il modello di flow condizionato
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # ottimizzatore Adam per l'addestramento

# Training loop
batch_size = 32
max_iter = 20000
loss_hist = []

print("Training Conditional Neural Spline Flow...")

for i in tqdm(range(max_iter)):
    idx = torch.randint(0, X.shape[0], (batch_size,))
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

    # if i % 1000 == 0 and i > 0:
    #     plt.plot(loss_hist)
    #     plt.title(f"Loss fino a iterazione {i}")
    #     plt.xlabel("Iterazione")
    #     plt.ylabel("Loss")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()


# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(loss_hist, label="Training Loss")
plt.xlabel("Iterazione")
plt.ylabel("- Log Likelihood")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# # Generazione di campioni condizionati
# model.eval() # Imposta il modello in modalità di valutazione per la fase di generazione
# with torch.no_grad():
#     # Prendi un contesto reale dal dataset
#     context = C[0].unsqueeze(0) #
#     context = context.to(device).repeat(1000, 1)
#     samples, _ = model.sample(1000, context=context) # Genera 1000 campioni
#     samples = samples.cpu().numpy() * std_x + mean_x # Denormalizza i campioni

# # PCA per visualizzazione (distribuzione reale vs generata)
# X_pca = PCA(n_components=2).fit_transform(df.iloc[:, n_context_dims:].values)
# samples_pca = PCA(n_components=2).fit_transform(samples)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolor='k')
# plt.title("Distribuzione Reale")

# plt.subplot(1, 2, 2)
# plt.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.5, edgecolor='k')
# plt.title("Distribuzione Generata")

# plt.tight_layout()
# plt.show()

# Salvataggio modello allenato
torch.save({
    'model_state_dict': model.state_dict(),
    'mean_x': mean_x,
    'std_x': std_x,
    'mean_c': mean_c,
    'std_c': std_c
}, 'trained_CNF_300_traj-split.pth')
