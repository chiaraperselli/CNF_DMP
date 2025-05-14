import os
import numpy as np
import pandas as pd
from movement_primitives.dmp._cartesian_dmp import CartesianDMP
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *  

# Path alla cartella con le traiettorie
trajectory_folder = TRAJECTORY_FOLDER
output_file = OUTPUT_DMP_WEIGHTS_CSV

all_weights = [] # Lista per memorizzare i pesi di tutte le traiettorie

# Cicla su tutti i file .csv della cartella
for filename in sorted(os.listdir(trajectory_folder)):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(trajectory_folder, filename))
        T = df['dt'].values
        Y = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

        dmp = CartesianDMP(
            n_weights_per_dim=N_WEIGHTS_PER_DIM,
            execution_time=T[-1],
            smooth_scaling=SMOOTH_SCALING,
            alpha_y=ALPHA_Y,
            beta_y=BETA_Y
        )
        dmp.imitate(T, Y,regularization_coefficient=REGULARIZATION_COEFFICIENT)
        weights = dmp.get_weights().reshape(1, -1)
        all_weights.append(weights)

# Salva tutti i pesi in un unico CSV
all_weights_matrix = np.vstack(all_weights)
pd.DataFrame(all_weights_matrix).to_csv(output_file, index=False)
print(f"{len(all_weights)} traiettorie processate. Pesi salvati in {output_file}")
