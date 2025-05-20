import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from movement_primitives.dmp import CartesianDMP
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.neural_spline import CircularAutoregressiveRationalQuadraticSpline

# CONFIG
MODEL_PATH = "trained_CNF_300_traj-split.pth"
CONTEXT_CSV = "300_traj_split/test_contexts.csv"
GT_WEIGHTS_CSV = "300_traj_split/test_dataset.csv"
SAMPLES_PER_CONTEXT = 50
OUTPUT_DIR = "sampling_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DMP parameters
N_WEIGHTS_PER_DIM = 30
N_WEIGHTS_TOTAL = 60
N_CONTEXT_DIMS = 4
NUM_LAYERS = 10
NUM_BLOCKS = 2
NUM_HIDDEN_CHANNELS = 128
NUM_BINS = 12
TAIL_BOUND = 3.0
DROPOUT_PROB = 0.05
PERMUTE_MASK = True
INIT_IDENTITY = True
DT = 0.01
DEFAULT_EXECUTION_TIME = 1.0
ALPHA_Y = 48.0
BETA_Y = ALPHA_Y / 4.0
SMOOTH_SCALING = True

sns.set(style="whitegrid", palette="deep", font_scale=1.2)

# === Funzione: carica modello
def load_model(path):
    base = DiagGaussian(N_WEIGHTS_TOTAL)
    flows = [
        CircularAutoregressiveRationalQuadraticSpline(
            num_input_channels=N_WEIGHTS_TOTAL,
            num_blocks=NUM_BLOCKS,
            num_hidden_channels=NUM_HIDDEN_CHANNELS,
            ind_circ=[],
            num_context_channels=N_CONTEXT_DIMS,
            num_bins=NUM_BINS,
            tail_bound=TAIL_BOUND,
            activation=nn.ReLU,
            dropout_probability=DROPOUT_PROB,
            permute_mask=PERMUTE_MASK,
            init_identity=INIT_IDENTITY
        )
        for _ in range(NUM_LAYERS)
    ]
    model = ConditionalNormalizingFlow(base, flows)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["mean_x"], checkpoint["std_x"], checkpoint["mean_c"], checkpoint["std_c"]

# === Funzione: genera traiettoria da pesi (x, y)
def generate_trajectory_2d(weights, start, goal):
    weights_full = np.zeros(6 * N_WEIGHTS_PER_DIM)
    weights_full[:N_WEIGHTS_PER_DIM] = weights[:N_WEIGHTS_PER_DIM]
    weights_full[N_WEIGHTS_PER_DIM:2 * N_WEIGHTS_PER_DIM] = weights[N_WEIGHTS_PER_DIM:N_WEIGHTS_TOTAL]

    dmp = CartesianDMP(n_weights_per_dim=N_WEIGHTS_PER_DIM, smooth_scaling=SMOOTH_SCALING,
                       alpha_y=ALPHA_Y, beta_y=BETA_Y)

    start_position = np.array([start[0], start[1], 0.0])
    goal_position = np.array([goal[0], goal[1], 0.0])
    start_orientation = np.array([0.0, 0.0, 0.0, 1.0])
    goal_orientation = np.array([0.0, 0.0, 0.0, 1.0])
    start_y = np.concatenate([start_position, start_orientation])
    goal_y = np.concatenate([goal_position, goal_orientation])

    dmp.set_weights(weights_full)
    dmp.configure(start_y=start_y, goal_y=goal_y,
                  start_yd=np.zeros(6), goal_yd=np.zeros(6),
                  start_ydd=np.zeros(6), goal_ydd=np.zeros(6))
    _, Y = dmp.open_loop(run_t=DEFAULT_EXECUTION_TIME)
    return Y[:, :2]

# === MAIN
def main():
    model, mean_x, std_x, mean_c, std_c = load_model(MODEL_PATH)

    contexts = pd.read_csv(CONTEXT_CSV, header=None).values.astype(np.float32)
    gt_weights = pd.read_csv(GT_WEIGHTS_CSV, header=None).values[:, :N_WEIGHTS_TOTAL].astype(np.float32)

    for i, (context, gt_w) in enumerate(zip(contexts, gt_weights)):
        start = context[:2]
        goal = context[2:]
        context_norm = (context - mean_c) / std_c
        context_tensor = torch.tensor(context_norm).float().unsqueeze(0).repeat(SAMPLES_PER_CONTEXT, 1)

        with torch.no_grad():
            samples, _ = model.sample(SAMPLES_PER_CONTEXT, context=context_tensor)
        samples = samples.numpy() * std_x + mean_x

        traj_nominal = generate_trajectory_2d(gt_w, start, goal)

        # Plot
        plt.figure(figsize=(7, 6))
        for w in samples:
            traj = generate_trajectory_2d(w, start, goal)
            sns.lineplot(x=traj[:, 0], y=traj[:, 1], alpha=0.3, linewidth=1)

        sns.lineplot(x=traj_nominal[:, 0], y=traj_nominal[:, 1], color='black', linewidth=2, label="Nominale")
        plt.scatter(start[0], start[1], color='green', marker='x', s=80, label="Start")
        plt.scatter(goal[0], goal[1], color='red', marker='o', s=80, label="Goal")
        plt.title(f"Fascio traiettorie – test #{i}")
        plt.axis("equal")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"traj_fascio_{i:02d}.png"))
        plt.close()

if __name__ == "__main__":
    main()
