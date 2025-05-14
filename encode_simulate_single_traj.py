# Codice che, dato un file CSV con una traiettoria, la carica e la codifica in un DMP, 
# per poi riprodurla e calcolare l'errore rispetto alla traiettoria originale; 
# inoltre salva i pesi della DMP in un file CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import necessario per plot 3D
from movement_primitives.dmp._cartesian_dmp import CartesianDMP
from scipy.spatial.transform import Rotation as R # Import per la rotazione dei quaternioni
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Carica la traiettoria (training data)

df = pd.read_csv("traj_dataset_10contexts/context_1101/traj_ap0_13.csv")  # Cambia il percorso del file CSV
T = df['dt'].values  # Tempo
Y = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values  # Posizione e orientamento

# Inizializza DMP e imita traiettoria (alpha=25.0 e beta=6.25 di default, ma posso cambiarli per vedere se migliora la traiettoria, basta mantenere sempre beta=alfa/4)
dmp = CartesianDMP(n_weights_per_dim=N_WEIGHTS_PER_DIM, execution_time=T[-1],  smooth_scaling=SMOOTH_SCALING, alpha_y=ALPHA_Y, beta_y=BETA_Y)  # tempo di esecuzione è l'ultimo tempo della traiettoria
dmp.imitate(T, Y, regularization_coefficient=REGULARIZATION_COEFFICIENT)

# Riproduci traiettoria con DMP 
T_repro, Y_repro = dmp.open_loop()

# Salva pesi in CSV (una riga) 
weights = dmp.get_weights()
weights_df = pd.DataFrame(weights.reshape(1, -1)) # Trasforma in DataFrame
#weights_df.to_csv("dmp_weights_prova_2.csv", index=False)
#print("Pesi salvati")

# Visualizzazione 2D: confronto tra traiettoria originale e DMP (un plot per ogni dimensione)
plt.figure(figsize=(12, 6))

labels = ['x', 'y', 'z']
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(T, Y[:, i], label='Original')
    plt.plot(T_repro, Y_repro[:, i], '--', label='DMP')
    plt.ylabel(labels[i])
    plt.legend()

plt.xlabel("Time [s]")
plt.suptitle("DMP Position Trajectory: Original vs Reproduced")
plt.tight_layout()
plt.show()

# Plot orientamento (quaternioni) 
plt.figure(figsize=(12, 8))
labels = ['qx', 'qy', 'qz', 'qw']
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(T, Y[:, 3 + i], label='Original')
    plt.plot(T_repro, Y_repro[:, 3 + i], '--', label='DMP')
    plt.ylabel(labels[i])
    plt.legend()

plt.xlabel("Time [s]")
plt.suptitle("DMP Orientation Trajectory (Quaternions): Original vs Reproduced")
plt.tight_layout()
plt.show()

# Visualizzazione 3D della traiettoria nello spazio

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], label='Original', color='blue')
ax.plot(Y_repro[:, 0], Y_repro[:, 1], Y_repro[:, 2], '--', label='DMP', color='orange')
ax.scatter(Y[0, 0], Y[0, 1], Y[0, 2], color='green', label='Start', s=50)
ax.scatter(Y[-1, 0], Y[-1, 1], Y[-1, 2], color='red', label='Goal', s=50)
ax.set_title("3D Trajectory: Original vs DMP")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()

# Calcolo dell'errore tra la traiettoria originale e quella riprodotta

# Verifica dimensioni 
min_len = min(len(T), len(Y), len(Y_repro))

# Tronca tutto alla stessa lunghezza
T = T[:min_len] 
Y = Y[:min_len]
Y_repro = Y_repro[:min_len]

# Supponendo che T e T_repro abbiano stesso passo temporale e durata
mask_5s = T <= 5.0


# Posizione: errore euclideo frame per frame
errors_pos = np.linalg.norm(Y[mask_5s, :3] - Y_repro[mask_5s, :3], axis=1)
mean_pos_cm = np.mean(errors_pos) * 100
max_pos_cm = np.max(errors_pos) * 100

print(f"Errore medio posizione (0–5s): {mean_pos_cm:.2f} cm")
print(f"Errore massimo posizione (0–5s): {max_pos_cm:.2f} cm")

# Orientamento: errore angolare frame per frame (in gradi)
def quat_angle_error(q1, q2):
    # scipy usa [x, y, z, w]
    r1 = R.from_quat(q1) #
    r2 = R.from_quat(q2)
    rel_rot = r2 * r1.inv() 
    return rel_rot.magnitude() * (180 / np.pi)

errors_ori = [
    quat_angle_error(Y[i, 3:7][[1,2,3,0]], Y_repro[i, 3:7][[1,2,3,0]]) # 
    for i in range(min_len) if T[i] <= 5.0
]
mean_ori_deg = np.mean(errors_ori)
max_ori_deg = np.max(errors_ori)

print(f"Errore medio orientamento (0–5s): {mean_ori_deg:.2f}°")
print(f"Errore massimo orientamento (0–5s): {max_ori_deg:.2f}°")
