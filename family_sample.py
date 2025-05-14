# family_sample.py - Sampling pesi DMP condizionati al contesto

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import CartesianDMP
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows import MaskedAffineAutoregressive
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# STEP 1: Carica il modello CNF

def load_model(path, dim, context_dim, num_layers):
    base = DiagGaussian(dim)
    flows = [
        MaskedAffineAutoregressive(
            features=dim,
            hidden_features=NUM_HIDDEN_CHANNELS,
            context_features=context_dim,
            num_blocks=NUM_BLOCKS,
            use_residual_blocks=True,
            dropout_probability=DROPOUT_PROB,
            activation=nn.ReLU()
        ) for _ in range(num_layers)
    ]
    model = ConditionalNormalizingFlow(base, flows)
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['mean_x'], checkpoint['std_x'], checkpoint['mean_c'], checkpoint['std_c']

# STEP 2-3: Sampling + Traiettoria DMP 
def generate_trajectories_from_context(context_str, model_path, samples_per_context):
    dim = N_WEIGHTS_TOTAL
    context_dim = N_CONTEXT_DIMS
    device = torch.device("cpu")
    model, mean_x, std_x, mean_c, std_c = load_model(model_path, dim, context_dim, NUM_LAYERS)
    model.to(device)

    context = np.array([int(c) for c in context_str], dtype=np.float32).reshape(1, -1)
    context_tensor = torch.tensor(context, dtype=torch.float32).repeat(samples_per_context, 1).to(device)

    with torch.no_grad():
        samples, _ = model.sample(samples_per_context, context=context_tensor)
    samples = samples.cpu().numpy() * std_x + mean_x
    #samples = np.clip(samples, -100, 100)


    # Salva i pesi generati 
    weights_df = pd.DataFrame(samples)
    weights_df.to_csv(f"generated_weights_prova4_{context_str}.csv", index=False, header=False)

    all_trajs = []
    for row in samples:
        weights_full = np.zeros(2 * N_WEIGHTS_PER_DIM)
        weights_full[:N_WEIGHTS_PER_DIM] = row[:N_WEIGHTS_PER_DIM]  # x
        weights_full[N_WEIGHTS_PER_DIM:2*N_WEIGHTS_PER_DIM] = row[N_WEIGHTS_PER_DIM:2*N_WEIGHTS_PER_DIM]  # y

        dmp = CartesianDMP(n_weights_per_dim=N_WEIGHTS_PER_DIM, smooth_scaling=SMOOTH_SCALING, alpha_y=ALPHA_Y, beta_y=BETA_Y)

        start_position = np.array([0.3, 2.7, 0.0])
        goal_position = np.array([2.7, 0.3, 0.0])
        start_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        goal_orientation = np.array([1.0, 0.0, 0.0, 0.0])

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

    return all_trajs, context[0]

# MAIN 
if __name__ == "__main__":
    context_str = input("Inserisci un contesto binario a 4 cifre: ")
    trajs, context_bin = generate_trajectories_from_context(context_str, MODEL_PATH, SAMPLES_PER_CONTEXT)