import tkseem as tk
import torch
from dotless_arabic.constants import *

TOKENIZER_CLASS = tk.WordTokenizer

NUM_LAYERS = 3  # for GRU


MAX_EPOCHS = 100
BATCH_SIZE = 256
HIDDEN_SIZE = 400
DROPOUT_PROB = 0.333
EMBEDDING_SIZE = 400
LEARNING_RATE = 0.001
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

TEST_SIZE = 0.1
VAL_SIZE = 0.01

GPU_DEVICES = [0]
