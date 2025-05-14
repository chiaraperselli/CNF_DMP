import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from movement_primitives.dmp import CartesianDMP
from pytransform3d.rotations import quaternion_slerp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Configurazione iniziale (deve essere la stessa di quella usata per generare i pesi !!)
csv_path = "dmp_weights_prova_2.csv"  # file con i pesi
n_weights_per_dim = 30 
start_position = np.array([0.4, -0.2, 0.1]) # posizione (x,y,z) 
goal_position = np.array([0.6, 0.2, 0.1])
start_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternione (w, x, y, z)
goal_orientation = np.array([1.0, 0.0, 0.0, 0.0])
# Carica i pesi dal file CSV
weights_matrix = np.loadtxt(csv_path, delimiter=",",skiprows=1)  # skiprows=1 fa saltare la prima riga e legge solo la seconda

# Controlla se i pesi sono in un formato corretto 
if weights_matrix.ndim == 1:
    weights_matrix = weights_matrix.reshape(1, -1) # Se i pesi sono in un array monodimensionale, li rimodella in una matrice 2D

# Ciclo sui pesi per generare traiettorie

for idx, weights in enumerate(weights_matrix):
    dmp = CartesianDMP(n_weights_per_dim=n_weights_per_dim, smooth_scaling=True, alpha_y=25.0, beta_y=6.25)
    dmp.set_weights(weights)

    # Configura start e goal
    start_y = np.concatenate([start_position, start_orientation])
    goal_y = np.concatenate([goal_position, goal_orientation])
    dmp.configure(start_y=start_y, goal_y=goal_y)

    # Simula traiettoria 
    T, Y = dmp.open_loop()

    # Visualizza la traiettoria in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], label=f"Trajectory {idx+1}")
    ax.scatter(*start_position, c='green', label='Start')
    ax.scatter(*goal_position, c='red', label='Goal')
    ax.set_title(f"Traiettoria DMP #{idx+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()
