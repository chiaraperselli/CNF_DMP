import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Fixed aperture positions in the environment (x, y)
aperture_coords = np.array([
    [0.5, 1.5],
    [1.2, 1.5],
    [1.9, 1.5],
    [2.6, 1.5],
])

def passes_through_aperture(traj_xy, aperture, threshold=0.1):
    """
    Returns True if the trajectory passes within a given threshold of the aperture.
    """
    dists = cdist(traj_xy, aperture.reshape(1, -1))
    return np.min(dists) <= threshold

def evaluate_single_folder(folder_path, binary_context, threshold=0.1):
    """
    Evaluates the percentage of trajectories in a folder that pass through all required apertures.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    total = len(csv_files)
    valid = 0

    for f in csv_files:
        df = pd.read_csv(os.path.join(folder_path, f))
        traj = df[["x", "y"]].values

        # Check if trajectory passes through all expected apertures
        is_valid = True
        for i, bit in enumerate(binary_context):
            if bit:  # aperture should be open
                if not passes_through_aperture(traj, aperture_coords[i], threshold):
                    is_valid = False
                    break
        if is_valid:
            valid += 1

    percentage = 100 * valid / total if total > 0 else 0
    print(f"[{os.path.basename(folder_path)}] {valid}/{total} valid trajectories ({percentage:.2f}%)")
    return percentage

def evaluate_all_subfolders(base_path, threshold=0.1):
    """
    Automatically evaluate all subfolders named as binary contexts (e.g., 0110, 1111...).
    """
    results = []
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and len(folder) == 4 and all(c in "01" for c in folder):
            binary_context = np.array([int(c) for c in folder])
            percent = evaluate_single_folder(folder_path, binary_context, threshold)
            results.append({"Context": folder, "Valid (%)": percent})

    df = pd.DataFrame(results)
    print("\n Overall Results:")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate sampled trajectories based on aperture context.")
    parser.add_argument("path", help="Path to a single folder (e.g., 0110/) or parent directory of all context folders.")
    parser.add_argument("--context", help="Specify a binary context (e.g., 0110) for single-folder evaluation.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Distance threshold for aperture proximity [default: 0.1]")

    args = parser.parse_args()

    if args.context:
        context_array = np.array([int(c) for c in args.context.strip()])
        evaluate_single_folder(args.path, context_array, args.threshold)
    else:
        evaluate_all_subfolders(args.path, args.threshold)
