"""
Training: hyper-parameters for training the model
"""
BATCH_SIZE = 64         # Sequences per batch
NUM_STEPS = 150         # Number of sequence steps per batch
LSTM_SIZE = 2048        # Size of hidden layers in LSTMs
NUM_LAYERS = 2          # Number of LSTM layers
LEARNING_RATE = 0.001   # Learning rate
KEEP_PROB = 0.5         # Dropout keep probability
SAVE_EVERY_N = 5000     # Save every N iterations
EPOCHS = 60

"""
Sampling: params for sampling the model
"""
OUTPUT_LENGTH = 5000
MODEL_VERSION = "v24"
CHECKPOINT_NAME = "trained.ckpt"
PRIME_EVENT = "@\n.\t8ee-/L\n"
#Picks one of top_n most likely next characters
PICK_TOP_N = 5

MEASURE_SYMBOL = "@"
