# Codice che legge tutte le traiettorie in una cartella (una ad una), 
# le codifica in DMP e salva i pesi in un file CSV (una riga di pesi+contesto(include anche start/goal_pos/orient) per ogni traiettoria)

import os
import numpy as np
import pandas as pd
from movement_primitives.dmp._cartesian_dmp import CartesianDMP
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import * 

# Configurazione 
trajectory_folder = TRAJECTORY_FOLDER
#output_file = OUTPUT_DMP_WEIGHTS_CSV
output_file = "dmp_weights_safe_traj_300.csv"
n_weights_per_dim = N_WEIGHTS_PER_DIM
all_data = [] # Lista vuota per memorizzare i pesi e il contesto

# Loop su tutti i file (tutte le traiettorie dimostrate)
files = sorted(os.listdir(trajectory_folder), key=lambda x: int(x.split("_")[1].split(".")[0])) # Ordina i file in base al numero della traiettoria
for filename in files:
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(trajectory_folder, filename))
        T = df['dt'].values
        Y = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

        # Ricava start e goal (posizione + orientamento)
        start_position = Y[0, :3]
        goal_position = Y[-1, :3]
        start_orientation = Y[0, 3:]
        goal_orientation = Y[-1, 3:]

        # Crea DMP e imita la traiettoria
        dmp = CartesianDMP(
            n_weights_per_dim=N_WEIGHTS_PER_DIM,
            execution_time=T[-1],
            smooth_scaling=SMOOTH_SCALING,
            alpha_y=ALPHA_Y,
            beta_y=BETA_Y
        )
        dmp.imitate(T, Y)
        weights = dmp.get_weights().reshape(1, -1)  # shape (1, 180)

        # Concatena pesi + contesto
        context = np.concatenate([start_position, goal_position, start_orientation, goal_orientation]).reshape(1, -1)
        full_row = np.hstack([weights, context])  # shape (1, 194): 180 pesi + 14 contesto (per ogni traiettoria)
        all_data.append(full_row)

# Salva in CSV finale
all_data_matrix = np.vstack(all_data)
pd.DataFrame(all_data_matrix).to_csv(output_file, index=False, header=False)
print(f"{len(all_data)} traiettorie salvate con contesto in '{output_file}'")
