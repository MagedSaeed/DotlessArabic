import time
from pytorch_lightning import Callback


class EpochTimerCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_times = list()
        self.average_epochs_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        end_time = time.time()
        epoch_time = end_time - self.start_time
        self.epochs_times.append(epoch_time)

    def on_train_end(self, trainer, pl_module):
        self.average_epochs_time = sum(self.epochs_times) / len(self.epochs_times)
        pl_module.log(
            name="per-epoch-avg-time",
            value=self.average_epochs_time,
        )
