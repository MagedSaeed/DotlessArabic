import os
import nltk
import random

import torch

from pytorch_lightning import seed_everything


def configure_environment():
    random.seed(42)
    nltk.download("stopwords")
    # os.environ["WANDB_MODE"] = "disabled"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # to see CUDA errors
    torch.cuda.empty_cache()  # to free gpu memory
    seed_everything(42, workers=True)
    # other options: https://stackoverflow.com/questions/15197286/how-can-i-flush-gpu-memory-using-cuda-physical-reset-is-unavailable
