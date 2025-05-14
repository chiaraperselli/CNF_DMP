import os
import numpy as np
import matplotlib.pyplot as plt

# Parametri ambiente
room_width = 3.0
room_height = 3.0
start = np.array([0.3, 2.7])
goal = np.array([2.7, 0.5])
aperture_centers = [0.6, 1.2, 1.8, 2.4]
aperture_y = 1.5
aperture_width = 0.3
aperture_height = 0.1

def plot_context_trajectories(base_dir, context_name):
    context_path = os.path.join(base_dir, context_name)
    if not os.path.exists(context_path):
        raise FileNotFoundError(f"Cartella {context_path} non trovata.")

    # Estrai contesto binario
    # context_bin = [int(c) for c in context_name.split("_")[1]]
    # active_apertures = [i for i, v in enumerate(context_bin) if v == 1]

    # Carica traiettorie
    trajectories = []
    for file in sorted(os.listdir(context_path)):
        if file.endswith(".csv"):
            data = np.loadtxt(os.path.join(context_path, file), delimiter=',', skiprows=1)
            trajectories.append(data[:, 1:3])  # x, y

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], color='green', alpha=0.6)

    # Start e goal
    ax.plot(start[0], start[1], 'ro')
    ax.text(start[0] + 0.05, start[1], 'A', color='red')
    ax.plot(goal[0], goal[1], 'ro')
    ax.text(goal[0] + 0.05, goal[1], 'B', color='red')

    # Disegna ostacoli lungo la parete (tranne le aperture attive)
    # wall_y = aperture_y
    # for i, xc in enumerate(aperture_centers):
    #     if context_bin[i] == 0:
    #         x0 = xc - aperture_width / 2
    #         y0 = wall_y - aperture_height / 2
    #         ax.add_patch(plt.Rectangle((x0, y0), aperture_width, aperture_height, color='gray'))

    # Disegna aperture attive (punti neri)
    # for i in active_apertures:
    #     ax.plot(aperture_centers[i], wall_y, 'ks', markersize=6)

    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_height)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Traiettorie - Contesto {context_name}")
    plt.show()

# Esempio d'uso:
plot_context_trajectories("dataset_family_2000", "1111")
