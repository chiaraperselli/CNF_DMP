import pandas as pd
import numpy as np
import argparse
import os

def split_dataset(input_csv, output_dir, train_size=230, seed=42):
    # Carica CSV senza header
    df = pd.read_csv(input_csv, header=None)

    # Estrai colonne: primi 60 = pesi, poi contesto dalle colonne 180, 181, 183, 184
    weights_xy = df.iloc[:, :60]
    context_xy = df.iloc[:, [180, 181, 183, 184]]
    full_df = pd.concat([weights_xy, context_xy], axis=1)

    # Shuffle
    full_df_shuffled = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    df_train = full_df_shuffled.iloc[:train_size]
    df_test = full_df_shuffled.iloc[train_size:]

    # Output paths
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_dataset.csv")
    test_path = os.path.join(output_dir, "test_dataset.csv")
    context_path = os.path.join(output_dir, "test_contexts.csv")

    # Save train/test
    df_train.to_csv(train_path, index=False, header=False)
    df_test.to_csv(test_path, index=False, header=False)

    # Estrai solo i 4 valori di contesto dalle ultime colonne del test
    df_test_contexts = df_test.iloc[:, -4:]
    df_test_contexts.to_csv(context_path, index=False, header=False)

    print(f"Dataset suddiviso e salvato:")
    print(f"    → Train: {train_path} ({len(df_train)} righe)")
    print(f"    → Test : {test_path} ({len(df_test)} righe)")
    print(f"    → Test contexts (solo start/goal): {context_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset DMPs for CNF training")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Percorso al file CSV originale (senza header)")
    parser.add_argument("--output_dir", type=str, default="./split_output",
                        help="Cartella dove salvare i file train/test/context")
    parser.add_argument("--train_size", type=int, default=230,
                        help="Numero di righe da usare per il training")
    args = parser.parse_args()

    split_dataset(args.input_csv, args.output_dir, args.train_size)
