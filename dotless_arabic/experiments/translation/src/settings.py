import os
import nltk
import random

import torch

from tqdm.auto import tqdm
from pytorch_lightning import seed_everything

from dotless_arabic.experiments.meter_classification.src import constants


def configure_environment():
    tqdm.pandas()
    random.seed(42)
    torch.cuda.empty_cache()  # to free gpu memory
    nltk.download("stopwords")
    seed_everything(42, workers=True)
    os.environ["WANDB_MODE"] = "disabled"
    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # to see CUDA errors
    # other options: https://stackoverflow.com/questions/15197286/how-can-i-flush-gpu-memory-using-cuda-physical-reset-is-unavailable
