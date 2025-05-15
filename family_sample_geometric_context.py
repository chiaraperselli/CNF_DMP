import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import CartesianDMP
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.neural_spline import CircularAutoregressiveRationalQuadraticSpline
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Carica il modello CNF 
def load_model(path, dim, context_dim, num_layers):
    base = DiagGaussian(dim)
    flows = []
    for _ in range(num_layers):
        flows.append(
            CircularAutoregressiveRationalQuadraticSpline(
                num_input_channels=dim,
                num_blocks=NUM_BLOCKS,
                num_hidden_channels=NUM_HIDDEN_CHANNELS,
                ind_circ=[],
                num_context_channels=context_dim,
                num_bins=NUM_BINS,
                tail_bound=TAIL_BOUND,
                activation=nn.ReLU,
                dropout_probability=DROPOUT_PROB,
                permute_mask=PERMUTE_MASK,
                init_identity=INIT_IDENTITY
            )
        )
    model = ConditionalNormalizingFlow(base, flows)
    # checkpoint = torch.load(path, map_location=torch.device("cpu"))
    # model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    mean_c = checkpoint['mean_c']
    std_c = checkpoint['std_c']

    model.eval()
    return model, checkpoint['mean_x'], checkpoint['std_x'], checkpoint['mean_c'], checkpoint['std_c']

# Genera pesi da contesto geometrico 
def generate_weights_from_context(context_geom, model_path, samples_per_context):
    dim = N_WEIGHTS_TOTAL
    context_dim = N_CONTEXT_DIMS
    device = torch.device("cpu")

    model, mean_x, std_x, mean_c, std_c = load_model(model_path, dim, context_dim, NUM_LAYERS)
    model.to(device)

    context_geom = (context_geom - mean_c) / std_c
    context_tensor = torch.tensor(context_geom, dtype=torch.float32).repeat(samples_per_context, 1).to(device)

    with torch.no_grad():
        samples, _ = model.sample(samples_per_context, context=context_tensor)
    samples = samples.cpu().numpy() * std_x + mean_x

    # pd.DataFrame(samples).to_csv("generated_weights_geometric_context_0111.csv", index=False, header=False)
    # print(f"Salvati {samples.shape[0]} set di pesi ")
    return samples

# Converte pesi in traiettorie DMP 
def weights_to_dmp_trajectories(weights_matrix):
    all_trajs = []
    for row in weights_matrix:
        weights_full = np.zeros(6 * N_WEIGHTS_PER_DIM)
        weights_full[:N_WEIGHTS_PER_DIM] = row[:N_WEIGHTS_PER_DIM]
        weights_full[N_WEIGHTS_PER_DIM:2*N_WEIGHTS_PER_DIM] = row[N_WEIGHTS_PER_DIM:2*N_WEIGHTS_PER_DIM]

        dmp = CartesianDMP(n_weights_per_dim=N_WEIGHTS_PER_DIM, smooth_scaling=SMOOTH_SCALING,
                           alpha_y=ALPHA_Y, beta_y=BETA_Y)
        dmp.set_weights(weights_full)

        start_position = np.array([0.3, 2.7, 0.0])
        goal_position = np.array([2.7, 0.45, 0.0])
        start_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        goal_orientation = np.array([0.0, 0.0, 0.0, 1.0])

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

        T, Y = dmp.open_loop(run_t=DEFAULT_EXECUTION_TIME)
        all_trajs.append(Y[:, :2])
    return all_trajs

# MAIN per contesto geomtetrico con 8 parametri
# if __name__ == "__main__":
#     print("Inserisci 8 numeri separati da spazio (es: 0.5 1.5 1.2 1.5 0 0 0 0)")
#     user_input = input("Contesto geometrico (8 valori): ")
#     try:
#         context_geom = np.array([float(x) for x in user_input.strip().split()])
#         assert context_geom.shape[0] == 8
#     except:
#         print("ERRORE: devi inserire esattamente 8 numeri separati da spazio.")
#         sys.exit(1)

#     samples = generate_weights_from_context(context_geom.reshape(1, -1), MODEL_PATH, SAMPLES_PER_CONTEXT)
#     trajectories = weights_to_dmp_trajectories(samples)

# MAIN per contesto geomtetrico con 4 parametri (solo coord x aperture)
if __name__ == "__main__":
    print("Inserisci un contesto binario (es: 1001): ")
    user_input = input("Contesto: ")
    try:
        assert len(user_input.strip()) == 4
        context_bin = [int(c) for c in user_input.strip()]
    except:
        print("ERRORE: devi inserire una stringa binaria lunga 4 cifre (es: 1010)")
        sys.exit(1)

    # Coordinate fisse delle aperture
    aperture_x = [0.5, 1.2, 1.9, 2.6]

    # Costruzione del contesto geometrico continuo: solo le x delle aperture aperte, 0 altrimenti
    context_geom = []
    for i, b in enumerate(context_bin):
        if b == 1:
            context_geom.append(aperture_x[i])
        else:
            context_geom.append(0.0)

    context_geom = np.array(context_geom, dtype=np.float32).reshape(1, -1)

    #Normalizzazione Z-score come nel training
    # mean_c = np.array([MEAN_X1, MEAN_X2, MEAN_X3, MEAN_X4], dtype=np.float32).reshape(1, -1)
    # std_c = np.array([STD_X1, STD_X2, STD_X3, STD_X4], dtype=np.float32).reshape(1, -1)
    # std_c = np.where(std_c == 0, 1.0, std_c)  # evitare divisione per 0
    # context_geom = (context_geom - mean_c) / std_c


    samples = generate_weights_from_context(context_geom, MODEL_PATH, SAMPLES_PER_CONTEXT)
    trajectories = weights_to_dmp_trajectories(samples)


    # Plot
    plt.figure(figsize=(8, 6))
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.6)

    # Aggiunta delle aperture
    aperture_coords = np.array([
        [0.5, 1.5],
        [1.2, 1.5],
        [1.9, 1.5], 
        [2.6, 1.5],
        ])
    for i, (x, y) in enumerate(aperture_coords):
        plt.scatter(x, y, color='black', marker='s', s=40)

    plt.scatter(0.3, 2.7, color='green', label='Start')
    plt.scatter(2.7, 0.45, color='red', label='Goal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title("Sampled trajectories - context {user_input}")
    plt.tight_layout()
    plt.show()
