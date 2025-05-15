import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import CartesianDMP
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


# CONFIG 
INPUT_CSV = "dmp_weights_family_final_continuous.csv"
OUTPUT_DIR = "interpolated_weights"
NUM_WEIGHTS = 40
NUM_CONTEXT = 4
aperture_x = [0.5, 1.2, 1.9, 2.6]

# Contesti binari di training 
binary_contexts_train = [
    [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [1,0,0,0],
    [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]
]

# Contesti binari di test 
binary_contexts_test = [
    [0,0,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,1],
    [1,0,1,0]
]

# Funzioni di supporto 
def binary_to_continuous(binary_ctx):
    return [aperture_x[i] if binary_ctx[i] == 1 else 0.0 for i in range(4)]

def binary_to_string(binary_ctx):
    return ''.join(str(b) for b in binary_ctx)

# Crea la cartella di output se non esiste 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carica dataset 
df = pd.read_csv(INPUT_CSV)
X = df.iloc[:, :NUM_WEIGHTS].values
C = df.iloc[:, NUM_WEIGHTS:NUM_WEIGHTS+NUM_CONTEXT].values

# Creazione arrays numpy per i contesti di training  (binari e continui)
train_contexts_continuous = np.array([binary_to_continuous(ctx) for ctx in binary_contexts_train])
train_contexts_bin = np.array(binary_contexts_train)

# Interpolazione per ogni contesto di test 
for binary_test in binary_contexts_test:
    c_test_bin = np.array(binary_test)
    ctx_name = binary_to_string(binary_test)

    # Trova i 2 contesti binari di training più vicini
    dists = np.linalg.norm(train_contexts_bin - c_test_bin, axis=1)  # calcola la distanza euclidea riga per riga
    idxs = np.argsort(dists)[:2] #ordina le distanze e prendi i primi 2 indici

    # Estrai i contesti binari di training più vicini
    binary_A = train_contexts_bin[idxs[0]]
    binary_B = train_contexts_bin[idxs[1]]
    # Conversione dei due contesti binari in continui
    cont_A = binary_to_continuous(binary_A)
    cont_B = binary_to_continuous(binary_B)

    # Seleziona tutte le traiettorie associate a A e B
    mask_A = np.all(C == cont_A, axis=1)
    mask_B = np.all(C == cont_B, axis=1)

    # estrazione dei pesi associati ai contesti A e B
    W_A = X[mask_A]
    W_B = X[mask_B]

    # Media vettoriale dei pesi delle 200 righe per ciascun contesto
    mean_A = np.mean(W_A, axis=0)
    mean_B = np.mean(W_B, axis=0)

    # Interpolazione finale: media dei due vettori medi 
    W_interp = (mean_A + mean_B) / 2

    # Salva in un file nella sottocartella
    output_path = os.path.join(OUTPUT_DIR, f"context_{ctx_name}.csv")
    df_out = pd.DataFrame([W_interp], columns=[f'w{i}' for i in range(NUM_WEIGHTS)])
    df_out.to_csv(output_path, index=False)

    print(f"Salvato: {output_path} ← media di {binary_to_string(binary_A)} + {binary_to_string(binary_B)}")

print("\n Tutti i file interpolati sono stati generati in: interpolated_weights/")

# Aperture (coordinata y fissa, come nel tuo script) 
aperture_coords = np.array([
    [0.5, 1.5],
    [1.2, 1.5],
    [1.9, 1.5], 
    [2.6, 1.5],
])

# Start e Goal fissi 
start_position = np.array([0.3, 2.7, 0.0])
goal_position = np.array([2.7, 0.45, 0.0])
start_orientation = np.array([0.0, 0.0, 0.0, 1.0])
goal_orientation = np.array([0.0, 0.0, 0.0, 1.0])
start_y = np.concatenate([start_position, start_orientation])
goal_y = np.concatenate([goal_position, goal_orientation])

# Loop sui file interpolati 
for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(OUTPUT_DIR, filename)
    weights = pd.read_csv(file_path).values.flatten()

    # Ricostruzione dei pesi completi: 6 DOF ma usiamo solo x e y
    weights_full = np.zeros(6 * N_WEIGHTS_PER_DIM)
    weights_full[:N_WEIGHTS_PER_DIM] = weights[:N_WEIGHTS_PER_DIM]  # x
    weights_full[N_WEIGHTS_PER_DIM:2*N_WEIGHTS_PER_DIM] = weights[N_WEIGHTS_PER_DIM:]  # y

    # Creazione DMP
    dmp = CartesianDMP(n_weights_per_dim=N_WEIGHTS_PER_DIM, smooth_scaling=SMOOTH_SCALING,
                       alpha_y=ALPHA_Y, beta_y=BETA_Y)
    dmp.set_weights(weights_full)
    dmp.configure(
        start_y=start_y,
        goal_y=goal_y,
        start_yd=np.zeros(6),
        goal_yd=np.zeros(6),
        start_ydd=np.zeros(6),
        goal_ydd=np.zeros(6)
    )
    T, Y = dmp.open_loop(run_t=DEFAULT_EXECUTION_TIME)

    # Plot 
    plt.figure(figsize=(7, 5))
    plt.plot(Y[:, 0], Y[:, 1], label="Interpolated DMP Trajectory")
    plt.scatter(0.3, 2.7, color='green', label='Start')
    plt.scatter(2.7, 0.45, color='red', label='Goal')

    for x, y in aperture_coords:
        plt.scatter(x, y, color='black', marker='s', s=40)

    plt.title(f"Interpolated DMP – {filename.replace('.csv', '')}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()
