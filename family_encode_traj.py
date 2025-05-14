# encode_all_traj_with_context.py modificato per il contesto binario
# Versione adattata per traiettorie realistiche con contesto binario sulle aperture

import os
import numpy as np
import pandas as pd
from movement_primitives.dmp._cartesian_dmp import CartesianDMP
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Configurazione
base_dir = TRAJECTORY_FOLDER  
output_file = OUTPUT_DMP_WEIGHTS_CSV  
n_weights_per_dim = N_WEIGHTS_PER_DIM
all_data = []

# Coordinate fisse delle aperture (x, y)
# aperture_coords = np.array([
#     [0.5, 1.5],  # Apertura 1
#     [1.2, 1.5],  # Apertura 2
#     [1.9, 1.5],  # Apertura 3
#     [2.6, 1.5],  # Apertura 4
# ])

# Coordinate x delle 4 aperture (fisse)
aperture_x = [0.5, 1.2, 1.9, 2.6]

dmp_examples = []
max_plots = 500  # Numero di traiettorie da visualizzare

# Loop sulle sottocartelle  
for context_folder in sorted(os.listdir(base_dir)):

    # Cartella tipo "context_1011"
    # if not context_folder.startswith("context_"):
    #     continue

    # context_path = os.path.join(base_dir, context_folder)
    # context_bin = [int(c) for c in context_folder.split("_")[1]]  # lista di 4 bit

    #Cartella tipo '1011'
    if len(context_folder) != 4 or not set(context_folder).issubset({'0', '1'}):
        continue  # ignora cartelle non valide

    context_path = os.path.join(base_dir, context_folder)
    context_bin = [int(c) for c in context_folder]  # lista di 4 bit


    for filename in sorted(os.listdir(context_path)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(context_path, filename))
            T = df['dt'].values
            Y = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

            # DMP
            dmp = CartesianDMP(
                n_weights_per_dim=n_weights_per_dim,
                execution_time=T[-1],
                smooth_scaling=SMOOTH_SCALING,
                alpha_y=ALPHA_Y,
                beta_y=BETA_Y
            )
            dmp.imitate(T, Y)
            weights = dmp.get_weights()  # shape (n_dims, n_weights_per_dim)
            weights_xy = weights[:40].reshape(1, -1)  # solo x, y (2 * 20 = 40 pesi)

            

            # Costruzione contesto geometrico continuo (8 parametri di contesto)
            # context_geom = []
            # for i, is_open in enumerate(context_bin):
            #     if is_open:
            #         context_geom.extend(aperture_coords[i])
            #     else:
            #             context_geom.extend([0.0, 0.0])
            # context_arr = np.array(context_geom).reshape(1, -1)  # shape (1, 8)



            # Costruzione del nuovo contesto: solo coordinate x se aperta, altrimenti 0.0
            context_geom_x = []
            for i, b in enumerate(context_bin):
                if b == 1:
                    context_geom_x.append(aperture_x[i])
                else:
                    context_geom_x.append(0.0)

            context_arr = np.array(context_geom_x).reshape(1, -1)  # shape (1, 4)


            # Concatena pesi + contesto (4 bit)
            # context_arr = np.array(context_bin).reshape(1, -1)


            full_row = np.hstack([weights_xy, context_arr])  # shape (1, 24)
            all_data.append(full_row)


            # salva fino a max_plots traiettorie per il plot
            # if len(dmp_examples) < max_plots:
            #     T_gen, Y_gen = dmp.open_loop()
            #     dmp_examples.append((T_gen, Y_gen[:, :2]))

# Salva in CSV finale
all_data_matrix = np.vstack(all_data)
pd.DataFrame(all_data_matrix).to_csv(output_file, index=False, header=False)
print(f"Salvati {len(all_data)} campioni in '{output_file}' con contesto binario")


# # Plot delle prime N traiettorie
# plt.figure(figsize=(8, 6))
# for T_ex, Y_ex in dmp_examples:
#     plt.plot(Y_ex[:, 0], Y_ex[:, 1], label="DMP Trajectory")
# plt.title(f"Prime {max_plots} traiettorie DMP ricostruite (x, y)")
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
