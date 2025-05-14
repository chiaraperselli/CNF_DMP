import pandas as pd

#estre i primi 60 pesi (30 per x e 30 per y) da un file CSV

# df = pd.read_csv("dmp_weights_safe_traj_fixed.csv", skiprows=1, header=None)
# df_xy = df.iloc[:, :60]  # 30 pesi per x + 30 per y
# df_xy.to_csv("dmp_weights_xy.csv", index=False, header=False)



# Codice per estrarre i pesi x e y da un file CSV e concatenarli con il contesto (start_x, start_y, goal_x, goal_y)

# Percorso file originale
input_csv = "dmp_weights_safe_traj_300.csv"

# Percorso file output
output_csv = "dmp_weights_safe_traj_300_xy.csv"


# Carica il file originale (senza header)
df = pd.read_csv(input_csv, header=None)

# Estrai:
weights_xy = df.iloc[:, :60]  # primi 60 pesi (x e y)
context_xy = df.iloc[:, [180, 181, 183, 184]]  # start_x, start_y, goal_x, goal_y

# Concatenazione finale: prima pesi, poi contesto
final_df = pd.concat([weights_xy, context_xy], axis=1)

# Salva
final_df.to_csv(output_csv, index=False, header=False)
print(f"File salvato in '{output_csv}' con shape {final_df.shape}")
