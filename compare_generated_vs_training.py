import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import CartesianDMP
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# CONFIGURAZIONE INIZIALE
# Percorsi
original_dir = "safe_trajectories"  # cartella con CSV originali
generated_csv = GENERATED_WEIGHTS_CSV     # CSV con pesi generati
n_weights_per_dim = N_WEIGHTS_PER_DIM

# Start/goal fissi
start_y = np.array([0.1, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
goal_y  = np.array([1.9, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# 1. Carica traiettorie originali di training
original_trajs = []
for path in sorted(glob.glob(os.path.join(original_dir, "trajectory_*.csv"))):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    original_trajs.append(data[:, 1:3])  # colonne x, y
if 'nominal_traj' not in locals():
    nominal_traj = data[:, 1:3]  # salva la prima come nominale


# 2. Carica pesi generati e simula traiettorie 
generated_trajs = []
weights_matrix = np.loadtxt(generated_csv, delimiter=",", skiprows=1)
if weights_matrix.ndim == 1:
    weights_matrix = weights_matrix.reshape(1, -1)

for weights in weights_matrix:
    weights_full = np.zeros(6 * n_weights_per_dim)
    weights_full[:2 * n_weights_per_dim] = weights  # x, y

    dmp = CartesianDMP(n_weights_per_dim=n_weights_per_dim,
                       smooth_scaling=SMOOTH_SCALING,
                       alpha_y=ALPHA_Y, beta_y=BETA_Y)
    dmp.set_weights(weights_full)
    dmp.configure(start_y=start_y, goal_y=goal_y)
    

    T, Y = dmp.open_loop()
    generated_trajs.append(Y[:, :2])  # x, y
    generated_trajs_array = np.stack(generated_trajs)  # shape: (N, T, 2)


# 3. Plot di confronto
plt.figure(figsize=(10, 6))

# Training (arancione chiaro, sottile)
for traj in original_trajs:
    plt.plot(traj[:, 0], traj[:, 1], color='#f0a202', alpha=0.3, linewidth=0.8)

# Generati (blu scuro, più spesso)
for traj in generated_trajs:
    plt.plot(traj[:, 0], traj[:, 1], color='#005f73', alpha=0.7, linewidth=1.5)

# Start/goal
plt.scatter(start_y[0], start_y[1], color='green', label='Start', s=50, edgecolors='black')
plt.scatter(goal_y[0], goal_y[1], color='red', label='Goal', s=50, edgecolors='black')

plt.title("Confronto traiettorie: Training (arancione) vs Generati (blu)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#Secondo plot confronto 
# === Calcola fascia x-y per le generate
x = np.mean(generated_trajs_array[:, :, 0], axis=0)
y_min = np.min(generated_trajs_array[:, :, 1], axis=0)
y_max = np.max(generated_trajs_array[:, :, 1], axis=0)
y_mean = np.mean(generated_trajs_array[:, :, 1], axis=0)

plt.figure(figsize=(10, 5))

# Training
for traj in original_trajs:
    plt.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.3, linewidth=0.8)

# Generate
for traj in generated_trajs_array:
    plt.plot(traj[:, 0], traj[:, 1], linestyle='--', color='blue', alpha=0.5, linewidth=1.0)

# Nominale
plt.plot(nominal_traj[:, 0], nominal_traj[:, 1], color='red', linewidth=2.5, label='Nominale')

# Fascia lilla
plt.fill_between(x, y_min, y_max,
                 color='mediumpurple', alpha=0.3,
                 label='Fascia generata (range totale)')

# Start / goal
plt.scatter(start_y[0], start_y[1], color='green', label='Start', s=50, edgecolors='black')
plt.scatter(goal_y[0], goal_y[1], color='red', label='Goal', s=50, edgecolors='black')
# Aggiunta alla legenda: linee fittizie
plt.plot([], [], color='gray', linewidth=0.8, alpha=0.3, label='Training')
plt.plot([], [], color='blue', linestyle='--', linewidth=1.0, alpha=0.5, label='Generati')

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Confronto spaziale: training, generate, fascia generata")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()
