import pandas as pd
import numpy as np
import os

def split_dataset(input_csv, output_dir, train_size=250, seed=42):
    # Carica CSV senza header
    df = pd.read_csv(input_csv, header=None)

    # Il file ha già 60 pesi + 4 contesto → nessuna estrazione necessaria
    full_df = df

    # Shuffle e reset indice
    full_df_shuffled = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split train/test
    df_train = full_df_shuffled.iloc[:train_size]
    df_test = full_df_shuffled.iloc[train_size:]

    # Crea cartella di output
    os.makedirs(output_dir, exist_ok=True)

    # Percorsi dei file da salvare
    train_path = os.path.join(output_dir, "train_dataset.csv")
    test_path = os.path.join(output_dir, "test_dataset.csv")
    context_path = os.path.join(output_dir, "test_contexts.csv")

    # Salva file CSV
    df_train.to_csv(train_path, index=False, header=False)
    df_test.to_csv(test_path, index=False, header=False)

    # Estrai solo contesto dalle ultime 4 colonne
    df_test_contexts = df_test.iloc[:, -4:]
    df_test_contexts.to_csv(context_path, index=False, header=False)

    print(f"[✓] Split completato:")
    print(f"    → Train: {train_path} ({len(df_train)} righe)")
    print(f"    → Test : {test_path} ({len(df_test)} righe)")
    print(f"    → Test contexts: {context_path}")
    
if __name__ == "__main__":
    # Imposta i parametri direttamente qui
    input_csv = "dmp_weights_safe_traj_300_xy.csv"
    output_dir = "300_traj_split" # Cartella dove salvare i file train/test/context
    train_size = 250  # numero di traiettorie di training

    split_dataset(input_csv, output_dir, train_size)
