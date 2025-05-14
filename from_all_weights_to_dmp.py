import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from movement_primitives.dmp import CartesianDMP
from pytransform3d.rotations import quaternion_slerp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# CONFIGURAZIONE INIZIALE 
csv_path = "generated_weights_geometric_context_0110.csv" # File con i pesi generati

start_position = np.array([0.3, 2.7, 0.0])
goal_position = np.array([2.7, 0.45, 0.0])
start_orientation = np.array([0.0, 0.0, 0.0, 1.0])
goal_orientation = np.array([0.0, 0.0, 0.0, 1.0]) 

# CARICAMENTO PESI 
weights_matrix = np.loadtxt(csv_path, delimiter=",")

if weights_matrix.ndim == 1:
    weights_matrix = weights_matrix.reshape(1, -1)

all_generated_trajs = []

# CICLO: GENERA E VISUALIZZA TRAIETTORIE 
for idx, weights in enumerate(weights_matrix):
    weights_full = np.zeros(6 * N_WEIGHTS_PER_DIM)
    weights_full[:2 * N_WEIGHTS_PER_DIM] = weights

    dmp = CartesianDMP(n_weights_per_dim=N_WEIGHTS_PER_DIM,
                       smooth_scaling=SMOOTH_SCALING,
                       alpha_y=ALPHA_Y, beta_y=BETA_Y)
    dmp.set_weights(weights_full)

    start_y = np.array([0.3, 2.7, 0.0, 0.0, 0.0, 0.0, 1.0])
    goal_y  = np.array([2.7, 0.45, 0.0, 0.0, 0.0, 0.0, 1.0])
    dmp.configure(start_y=start_y, goal_y=goal_y)
   
    # Simula
    T, Y = dmp.open_loop()

    all_generated_trajs.append(Y[:, :2])  # salviamo solo x e y


    # Visualizza 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], label=f"Trajectory {idx+1}")
    # ax.scatter(*start_position, c='green', label='Start')
    # ax.scatter(*goal_position, c='red', label='Goal')
    # ax.set_title(f"Traiettoria DMP #{idx+1}")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    
    # Visualizza 2D
    # plt.figure()
    # plt.plot(Y[:, 0], Y[:, 1], label=f"Trajectory {idx+1}")
    # plt.scatter(start_y[0], start_y[1], color='green', label='Start')
    # plt.scatter(goal_y[0], goal_y[1], color='red', label='Goal')
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title(f"Traiettoria DMP 2D #{idx+1}")
    # plt.legend()
    # plt.axis('equal')
    # plt.grid(True)
    # plt.tight_layout()
    # #plt.show()
    # plt.close()

# Plot unico di tutte le traiettorie generate
plt.figure(figsize=(8, 6))
for traj in all_generated_trajs:
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.6)

plt.scatter(start_y[0], start_y[1], color='green', label='Start')
plt.text(start_y[0], start_y[1] + 0.1, 'A', color='red', fontsize=12)
plt.scatter(goal_y[0], goal_y[1], color='red', label='Goal')
plt.text(goal_y[0], goal_y[1] - 0.1, 'B', color='red', fontsize=12)

# Aggiunta delle aperture
aperture_coords = np.array([
    [0.5, 1.5],
    [1.2, 1.5],
    [1.9, 1.5],
    [2.6, 1.5],
])
for i, (x, y) in enumerate(aperture_coords):
    plt.scatter(x, y, color='black', marker='s', s=40)

plt.title("Sampled trajectories for binary context - 0110") 
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
