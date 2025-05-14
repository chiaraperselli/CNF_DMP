import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from normflows import NormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.affine import MaskedAffineFlow 

# Carica dataset
df = pd.read_csv("fake_dmp_weights_highdim.csv")
X = df.values.astype(np.float32)

# Normalizzazione (mean-std)
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
X = torch.tensor(X, dtype=torch.float32)

dim = X.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = DiagGaussian(dim)

# Helper: crea MLP per s() e t()
def create_mlp(input_dim, output_dim, hidden_dim=128):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim)
    )

# Costruzione flow con maschere alternate
num_layers = 12
flows = []
for i in range(num_layers):
    # Maschera alternata: 0 1 0 1 ... / 1 0 1 0 ...
    mask = torch.tensor([(j + i) % 2 for j in range(dim)], dtype=torch.float32)
    s_net = create_mlp(dim, dim)
    t_net = create_mlp(dim, dim)
    flows.append(MaskedAffineFlow(mask, t=t_net, s=s_net))

# Costruisci modello completo
model = NormalizingFlow(base, flows).to(device)

# Ottimizzatore
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 256
max_iter = 3000
loss_hist = []

print("Training RealNVP (custom MaskedAffineFlow) su fake_dmp_weights_highdim...")

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
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.values)
samples_pca = pca.transform(samples)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolor='k')
plt.title("Distribuzione Reale (PCA)")

plt.subplot(1, 2, 2)
plt.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.5, edgecolor='k')
plt.title("Distribuzione Generata da RealNVP (MaskedAffineFlow custom)")

plt.tight_layout()
plt.show()

# Salva il modello e le statistiche per normalizzazione
torch.save({
    'model_state_dict': model.state_dict(),
    'mean': mean,
    'std': std
}, "trained_realNVP_model.pth")

print("Modello salvato in trained_realNVP_model.pth")

