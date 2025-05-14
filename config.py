# Configurazione dei parametri di DMPs e CNFs

# Parametri DMPs

N_WEIGHTS_PER_DIM = 20
ALPHA_Y = 48
BETA_Y = 12
DEFAULT_EXECUTION_TIME = 10.0
SMOOTH_SCALING = True
REGULARIZATION_COEFFICIENT = 1e-6
DT_FIXED = 0.01  # Tempo di campionamento fisso per la traiettoria
N_CONTEXT_DIMS = 4 # Dimensione del contesto 
#N_WEIGHTS_TOTAL = 6 * N_WEIGHTS_PER_DIM # 6D (x, y, z, qx, qy, qz, qw)
N_WEIGHTS_TOTAL = 2 * N_WEIGHTS_PER_DIM # solo x e y

# Parametri CNFs

NUM_LAYERS = 10
NUM_BLOCKS = 2
NUM_HIDDEN_CHANNELS = 128
#HIDDEN_FEATURES = 256
NUM_BINS = 12
TAIL_BOUND = 3.0
DROPOUT_PROB = 0.05
PERMUTE_MASK = True
INIT_IDENTITY = True
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
MAX_ITER = 50000

SAMPLES_PER_CONTEXT = 50  # Numero di traiettorie da generare per ogni contesto
#N_SAMPLES = 20 # Numero di set di pesi (quindi numero di traiettorie) che voglio generare nella fase di sampling

# File paths

TRAJECTORY_FOLDER = "dataset_family_2000/"
OUTPUT_DMP_WEIGHTS_CSV = "dmp_weights_family_final.csv"
MODEL_PATH = "trained_CNF_family_final_2.pth"  #trained model path
#CONTEXT_CSV_PATH = "context_binary.csv" # new context file
#GENERATED_WEIGHTS_CSV = "generated_weights_traj_safe_300.csv" 
#OUTPUT_TRAJECTORY_FOLDER = "generated_trajectories/"