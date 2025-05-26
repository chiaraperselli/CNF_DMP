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
def load_model(model_path, dim, context_dim, num_layers):
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
    # model = ConditionalNormalizingFlow(base, flows)
    # checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    # model.eval()
    model = ConditionalNormalizingFlow(base, flows)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['mean_x'], checkpoint['std_x'], checkpoint['mean_c'], checkpoint['std_c']
    return model
    # return model, checkpoint['mean_x'], checkpoint['std_x'], checkpoint['mean_c'], checkpoint['std_c']

# Genera pesi da contesto geometrico (modello allenato su PC)
def generate_weights_from_context(context_geom, model_path, samples_per_context):
    dim = N_WEIGHTS_TOTAL
    context_dim = N_CONTEXT_DIMS
    device = torch.device("cpu")
    model, mean_x, std_x, mean_c, std_c = load_model(model_path, dim, context_dim, NUM_LAYERS)
    #model = load_model(model_path, dim, context_dim, NUM_LAYERS)
    model.to(device)

    context_geom = (context_geom - mean_c) / std_c
    context_tensor = torch.tensor(context_geom, dtype=torch.float32).repeat(samples_per_context, 1).to(device)

    with torch.no_grad():
        samples, _ = model.sample(samples_per_context, context=context_tensor)
    samples = samples.cpu().numpy() * std_x + mean_x

    # pd.DataFrame(samples).to_csv("generated_weights_geometric_context_0111.csv", index=False, header=False)
    # print(f"Salvati {samples.shape[0]} set di pesi ")
    return samples

# Genera pesi da contesto geometrico (modello allenato su VM)
# def generate_weights_from_context(context_geom, weight_path, stats_path, samples_per_context):
#     dim = N_WEIGHTS_TOTAL
#     context_dim = N_CONTEXT_DIMS
#     device = torch.device("cpu")

#     # Carica modello e statistiche
#     model = load_model(weight_path, dim, context_dim, NUM_LAYERS)
#     model.to(device)

#     stats = np.load(stats_path)
#     mean_x = stats['mean_x']
#     std_x = stats['std_x']
#     mean_c = stats['mean_c']
#     std_c = stats['std_c']

#     # Normalizza il contesto
#     context_geom = (context_geom - mean_c) / std_c
#     context_tensor = torch.tensor(context_geom, dtype=torch.float32).repeat(samples_per_context, 1).to(device)

#     with torch.no_grad():
#         samples, _ = model.sample(samples_per_context, context=context_tensor)
#     samples = samples.cpu().numpy() * std_x + mean_x
#     return samples

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

        start_position = np.array([2.5, 0.5, 0.0])
        goal_position = np.array([1.0, 2.8, 0.0])
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

    #VERSIONE CONTESTO CONTINUO
    # samples = generate_weights_from_context(context_geom, MODEL_PATH, SAMPLES_PER_CONTEXT)
    # trajectories = weights_to_dmp_trajectories(samples)

    ##VERSIONE PER CONTESTO BINARIO 
    context_bin = np.array([int(c) for c in user_input.strip()], dtype=np.float32).reshape(1, -1)
    samples = generate_weights_from_context(context_bin, MODEL_PATH, SAMPLES_PER_CONTEXT)
    trajectories = weights_to_dmp_trajectories(samples)


    # Salvataggio traiettorie in .csv pronte per la valutazione
    # output_dir = f"sampled_traj_output/{user_input}"
    # os.makedirs(output_dir, exist_ok=True)

    # for i, traj in enumerate(trajectories):
    #     df = pd.DataFrame({
    #         "dt": np.linspace(0, DEFAULT_EXECUTION_TIME, len(traj)),
    #         "x": traj[:, 0],
    #         "y": traj[:, 1],
    #         "z": 0.0,
    #         "qx": 0.0,
    #         "qy": 0.0,
    #         "qz": 0.0,
    #         "qw": 1.0
    #     })
    #     df.to_csv(os.path.join(output_dir, f"traj_{i:03d}.csv"), index=False)

    # print(f"\nTraiettorie salvate in: {output_dir}")

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
    plt.title(f"Sampled trajectories - context {user_input}")
    plt.tight_layout()
    plt.show()
