"""
Training: hyper-parameters for training the model
"""
BATCH_SIZE = 64         # Sequences per batch
NUM_STEPS = 150         # Number of sequence steps per batch
LSTM_SIZE = 128         # Size of hidden layers in LSTMs
NUM_LAYERS = 2          # Number of LSTM layers
LEARNING_RATE = 0.001   # Learning rate
KEEP_PROB = 0.5         # Dropout keep probability
SAVE_EVERY_N = 5000     # Save every N iterations
EPOCHS = 60

"""
Sampling: params for sampling the model
"""
OUTPUT_LENGTH = 1000
MODEL_VERSION = "v25"
CHECKPOINT_NAME = "trained.ckpt"
PRIME_EVENT = "@\n.\t.\t2a/\t.\n"

MEASURE_SYMBOL = "@"
