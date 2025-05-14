# Codice che carica il modello CNF allenato e genera nuovi pesi DMP condizionati dato un contesto in input, 
# salvandoli tutti in un file CSV (una riga di pesi+contesto per ogni traiettoria)

import torch
import pandas as pd
import numpy as np
from normflows import ConditionalNormalizingFlow
from normflows.distributions.base import DiagGaussian
from normflows.flows.neural_spline import CircularAutoregressiveRationalQuadraticSpline
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Configurazione 
# IMPORTANTE: i parametri (num_layers, dim, num_hidden_channles ecc) devono essere gli stessi usati nel training del modello
model_path = MODEL_PATH
context_csv_path = CONTEXT_CSV_PATH
output_csv_path = GENERATED_WEIGHTS_CSV
n_weights_total = N_WEIGHTS_TOTAL
n_context_dims = N_CONTEXT_DIMS
num_layers = NUM_LAYERS
samples_per_context = SAMPLES_PER_CONTEXT  # Numero di traiettorie da generare per ogni contesto

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
                permute_mask=True,
                init_identity=True
            )
        )
    model = ConditionalNormalizingFlow(base, flows)
    checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['mean_x'], checkpoint['std_x'], checkpoint['mean_c'], checkpoint['std_c']

def generate_multiple_per_context(context_csv_path, model_path, output_csv_path, samples_per_context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = n_weights_total
    model, mean_x, std_x, mean_c, std_c = load_model(model_path, dim, n_context_dims, num_layers)
    model = model.to(device)

    # Caricamento contesti
    contexts = pd.read_csv(context_csv_path, header=None).values.astype(np.float32)
    contexts_norm = (contexts - mean_c) / std_c # Normalizzazione del contesto
    contexts_tensor = torch.tensor(contexts_norm, dtype=torch.float32).to(device)

    all_rows = [] # Lista per memorizzare i pesi generati

    with torch.no_grad():
        for i, context_tensor in enumerate(contexts_tensor):
            repeated_context = context_tensor.unsqueeze(0).repeat(samples_per_context, 1)
            samples, _ = model.sample(samples_per_context, context=repeated_context)
            samples = samples.cpu().numpy() * std_x + mean_x
            samples = np.clip(samples, -5000, 5000)  # Clipping dei valori per ridurre oscillazioni
            for sample in samples:
                full_row = np.concatenate([sample.flatten(), contexts[i]]) # Concatenazione pesi + contesto
                all_rows.append(full_row)

    all_rows = np.array(all_rows)
    np.savetxt(output_csv_path, all_rows, delimiter=",")
    print(f" Salvati {len(all_rows)} set di pesi DMP con contesto in '{output_csv_path}'")

if __name__ == "__main__":
    generate_multiple_per_context(context_csv_path, model_path, output_csv_path, samples_per_context)
