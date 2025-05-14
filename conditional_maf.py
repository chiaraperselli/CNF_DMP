import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows import MaskedAffineAutoregressive

# Carica dataset con contesto
df = pd.read_csv("fake_dmp_weights_with_context.csv")
n_context_dims = 2
C = df.iloc[:, :n_context_dims].values.astype(np.float32)
X = df.iloc[:, n_context_dims:].values.astype(np.float32)

# Normalizzazione
mean_x = X.mean(axis=0)
std_x = X.std(axis=0)
X = (X - mean_x) / std_x

mean_c = C.mean(axis=0)
std_c = C.std(axis=0)
C = (C - mean_c) / std_c

X = torch.tensor(X, dtype=torch.float32)
C = torch.tensor(C, dtype=torch.float32)

dim = X.shape[1]
context_dim = C.shape[1]
device = torch.device("cpu")

# Base distribution
base = DiagGaussian(dim)

# Costruzione modello con MaskedAffineAutoregressive condizionato
num_layers = 12
flows = []
for _ in range(num_layers):
    flows.append(
        MaskedAffineAutoregressive(
            features=dim,
            hidden_features=128,
            context_features=context_dim,
            num_blocks=2,
            use_residual_blocks=True,
            activation=nn.ReLU(),
            dropout_probability=0.0
        )
    )

model = ConditionalNormalizingFlow(base, flows).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
batch_size = 256
max_iter = 3000
loss_hist = []

print("Training Conditional MaskedAffineAutoregressive Flow...")

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

# Plot della loss
plt.plot(loss_hist)
plt.title("Training Loss (-log likelihood)")
plt.xlabel("Iterazione")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Generazione sample condizionato
model.eval()
with torch.no_grad():
    context = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    context = (context - mean_c) / std_c
    context = context.to(device).repeat(1000, 1)

    samples, _ = model.sample(1000, context=context)
    samples = samples.cpu().numpy() * std_x + mean_x

# PCA per visualizzazione
X_pca = PCA(n_components=2).fit_transform(df.iloc[:, n_context_dims:].values)
samples_pca = PCA(n_components=2).fit_transform(samples)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolor='k')
plt.title("Distribuzione Reale")

plt.subplot(1, 2, 2)
plt.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.5, edgecolor='k')
plt.title("Distribuzione Generata (MAF condizionato)")

plt.tight_layout()
plt.show()

# Salva modello
torch.save({
    'model_state_dict': model.state_dict(),
    'mean_x': mean_x,
    'std_x': std_x,
    'mean_c': mean_c,
    'std_c': std_c
}, "trained_conditional_maf_model.pth")