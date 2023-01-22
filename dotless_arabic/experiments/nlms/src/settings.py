import os

import wandb
from pytorch_lightning import seed_everything

from dotless_arabic.experiments.nlms.src import constants

CONFIGURED = False


def configure_environment():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    seed_everything(constants.RANDOM_SEED, workers=True)
    wandb.login()
