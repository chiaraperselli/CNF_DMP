import os
import numpy as np
import pandas as pd

# Gate coordinates (fixed in the workspace)
gate_coords = np.array([
    [0.5, 1.5],
    [1.2, 1.5],
    [1.9, 1.5],
    [2.6, 1.5],
])

# Evaluation tolerances (rectangle around the gate)
x_tolerance = 0.1   # tolerance along x-axis (e.g. ±10 cm)
y_tolerance = 0.01  # tolerance along y-axis (e.g. ±1 cm)

def evaluate_trajectory(trajectory, binary_context):
    """
    Check if the trajectory passes through all the gates indicated as open (1)
    """
    x = trajectory[:, 1]
    y = trajectory[:, 2]
    hits = 0

    for i, gate in enumerate(gate_coords):
        if binary_context[i] == 1:
            gx, gy = gate
            # Check if at least one point is within the rectangular gate window
            inside = np.where(
                (np.abs(x - gx) < x_tolerance) &
                (np.abs(y - gy) < y_tolerance)
            )[0]
            if len(inside) > 0:
                hits += 1

    return hits

def evaluate_folder(folder_path):
    """
    Evaluate all trajectories in the folder and return success rate
    """
    total = 0
    correct = 0

    # Extract binary context from folder name (e.g., "0111")
    context_name = os.path.basename(folder_path)
    binary_context = [int(c) for c in context_name]

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            traj = np.loadtxt(file_path, delimiter=",", skiprows=1)
            hits = evaluate_trajectory(traj, binary_context)
            expected = sum(binary_context)
            if hits == expected:
                correct += 1
            total += 1

    return context_name, correct, total, correct / total * 100

def main():
    base_path = "sampled_trajectories"  # Folder with subfolders like "0111", "1010", ...
    results = []

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            context, correct, total, accuracy = evaluate_folder(folder_path)
            results.append((context, correct, total, accuracy))

    df = pd.DataFrame(results, columns=["Context", "Correct", "Total", "Accuracy (%)"])
    df.to_csv("accuracy_results.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
