# Codice che, dato un file CSV con pesi e contesto, genera e visualizza le traiettorie DMP corrispondenti;
# poi salva ogni traiettoria in un file csv in formato (t,x,y,z,qx,qy,qz,qw) e crea una directory di output con tutte le traiettorie generate.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from movement_primitives.dmp import CartesianDMP
import os # Import per la gestione dei file
from scipy.interpolate import interp1d # Import per l'interpolazione lineare del tempo  
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import * 

# Configurazione 
csv_path = GENERATED_WEIGHTS_CSV  
n_weights_per_dim = N_WEIGHTS_PER_DIM
n_weights_total = N_WEIGHTS_TOTAL

# Caricamento pesi + contesto da file csv  
data = np.loadtxt(csv_path, delimiter=",")

if data.ndim == 1:
    data = data.reshape(1, -1)

all_trajectories = []  # Per salvare le traiettorie x-y
all_contexts = [] # Per salvare i contesti (start_x, start_y, goal_x, goal_y)

# Ciclo per generare e visualizzare le traiettorie DMP 
for idx, row in enumerate(data):
    # weights = row[:n_weights_total]
    # start_position = row[n_weights_total : n_weights_total+3]
    # goal_position = row[n_weights_total+3 : n_weights_total+6]
    # start_orientation = row[n_weights_total+6 : n_weights_total+10]
    # goal_orientation = row[n_weights_total+10 : n_weights_total+14]

    # caso due dimensioni (x e y)
    weights_full = np.zeros(6 * n_weights_per_dim)
    weights_full[:n_weights_total] = row[:n_weights_total]  # solo x, y
    # Usa solo 2D context: start_x, start_y, goal_x, goal_y
    start_position = np.array([row[n_weights_total], row[n_weights_total+1], 0.0])
    goal_position = np.array([row[n_weights_total+2], row[n_weights_total+3], 0.0])
    start_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    goal_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    dmp = CartesianDMP(n_weights_per_dim=n_weights_per_dim, smooth_scaling=SMOOTH_SCALING, alpha_y=ALPHA_Y, beta_y=BETA_Y)

    start_y = np.concatenate([start_position, start_orientation])
    goal_y = np.concatenate([goal_position, goal_orientation])
   
    
    dmp.configure(
    start_y=start_y,
    goal_y=goal_y,
    start_yd=np.zeros(6),
    goal_yd=np.zeros(6),
    start_ydd=np.zeros(6),
    goal_ydd=np.zeros(6)
)
    dmp.set_weights(weights_full)



    # Calcolo della durata in base alla distanza
    # distance = np.linalg.norm(goal_position - start_position)
    # execution_time = 0.5 + 1.0 * distance  # Durata minima di 0.5 secondi + 1 secondo per ogni metro di distanza (oppure imposto un valore fisso)
    # T, Y = dmp.open_loop(run_t=execution_time)

    # Gestione dinamica dell'execution_time
    if len(row) > n_weights_total + 14:
        # Se il file contiene anche l'execution_time dopo i 14 elementi del contesto, usa direttamente quello
        execution_time_dmp = row[n_weights_total + 14]
    else:
        # Se non c'è execution_time salvato nel file, usa default 10s (oppur setta lo stesso valore usato per le traiettorie di training)
        execution_time_dmp = DEFAULT_EXECUTION_TIME

    # Rollout DMP con execution_time corretto
    T, Y = dmp.open_loop(run_t=execution_time_dmp)

    # Correzione soft della parte finale verso il goal
    alpha = np.linspace(0, 1, 10)[:, None] ** 2

    if np.linalg.norm(Y[-1, :2] - goal_position[:2]) > 0.02:
        Y[-10:, :2] = (1 - alpha) * Y[-10:, :2] + alpha * goal_position[:2]


    all_trajectories.append(Y[:, :2])  # Salva solo x e y
    context_xy = [row[n_weights_total], row[n_weights_total+1], row[n_weights_total+2], row[n_weights_total+3]]
    all_contexts.append(context_xy)


    # #Plot 3D della traiettoria 
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
    # #plt.show()

    # Plot 2D della traiettoria singola
    plt.figure()
    plt.plot(Y[:, 0], Y[:, 1], label=f"Trajectory {idx+1}")
    plt.scatter(start_position[0], start_position[1], color='green', label='Start')
    plt.scatter(goal_position[0], goal_position[1], color='red', label='Goal')
    plt.title(f"Traiettoria DMP 2D #{idx+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.close()

    # # Ora salvo le traiettoria in un file CSV in formato (x,y,z,qx,qy,qz,qw) 

    # # Crea la directory di output 
    # output_dir = OUTPUT_TRAJECTORY_FOLDER
    # os.makedirs(output_dir, exist_ok=True)

    # # Imposta dt fisso
    # dt_fixed = DT_FIXED  # 10ms (frequenza di campionamento 100Hz)

    # # Durata totale della traiettoria impostata uguale al tempo di esecuzione della DMP
    # execution_time = T[-1]
    # #print (f"Tempo di esecuzione della DMP: {execution_time:.2f} secondi")


    # # Numero di passi desiderati
    # n_steps = int(execution_time / dt_fixed) # calcola il numero di passi in base al tempo di esecuzione e al dt fisso

    # # Nuovi timestamp uniformi
    # T_uniform = np.linspace(0.0, execution_time, n_steps) # crea un array di timestamp uniformi da 0 a execution_time con n_steps punti

    # # Interpolazione della traiettoria sui nuovi timestamp
    # interp_func = interp1d(T, Y, axis=0, kind='linear')
    # Y_uniform = interp_func(T_uniform)

    # # Prepara [t, x, y, z, qx, qy, qz, qw]
    # # NOTA: qui mettiamo direttamente T_uniform come primo campo! (cioè il nuovo timestamp uniforme)
    # trajectory_data = np.hstack((T_uniform.reshape(-1, 1), Y_uniform))

    # # Salva il file CSV
    # output_path = os.path.join(output_dir, f"generated_trajectory_{idx+1}.csv")
    # np.savetxt(output_path, trajectory_data, delimiter=",", header="t,x,y,z,qx,qy,qz,qw", comments='')

    # print(f"Traiettoria {idx+1} salvata in {output_path}")


# Plot finale con tutte le traiettorie x-y
plt.figure(figsize=(10, 6))
for i, traj in enumerate(all_trajectories):
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, label=f"T{i+1}")

    # Estrai start e goal dalla lista dei contesti
    context = all_contexts[i]
    start_position = context[:2]
    goal_position = context[2:4]

    # Plot dei punti start/goal
    plt.scatter(start_position[0], start_position[1], c='green', marker='o', s=40, edgecolors='black', zorder=5)
    plt.scatter(goal_position[0], goal_position[1], c='red', marker='X', s=40, edgecolors='black', zorder=5)

plt.title("Tutte le traiettorie generate (x-y)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

