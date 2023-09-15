from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    EarlyStopping,
)
from pytorch_lightning import Trainer

from ray import air, tune

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from dotless_arabic.experiments.nlms.src.utils import xavier_init
from dotless_arabic.experiments.nlms.src import constants


def train_lm_for_tuning(
    config,
    vocab_size,
    lm_model_class,
    train_dataloader,
    val_dataloader,
    max_epochs=5,
):
    lm_model = lm_model_class(
        vocab_size=vocab_size,
        **config,
    )

    callbacks = list()
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.1,
        patience=6,
        check_finite=True,
    )
    callbacks.append(early_stopping_callback)
    callbacks.append(
        TuneReportCallback(
            {
                "val_loss": "val_loss",
                "val_ppl": "val_ppl",
            },
            on="validation_end",
        )
    )
    # devices = gpu_devices
    xavier_init(model=lm_model)
    trainer = Trainer(
        accelerator="auto",
        deterministic=True,
        callbacks=callbacks,
        gradient_clip_val=3,  # for transformer model, this value might be very small, say 0.25
        max_epochs=max_epochs,
        val_check_interval=0.5,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=max(len(train_dataloader) // 50, 1),
    )
    trainer.validate(
        model=lm_model,
        dataloaders=val_dataloader,
    )

    trainer.fit(
        lm_model,
        train_dataloader,
        val_dataloader,
    )
    # return trainer


def tune_lm_model(
    vocab_size,
    lm_model_class,
    train_dataloader,
    val_dataloader,
):
    lm_model_config = {
        "num_layers": tune.grid_search([2, 3, 4]),
        "hidden_size": tune.grid_search([256, 512]),
        "embedding_size": tune.grid_search([256, 512]),
        "dropout_prob": tune.grid_search([0.2, 0.3333, 0.5]),
        "learning_rate": tune.grid_search([0.01, 0.001]),
    }

    scheduler = ASHAScheduler(grace_period=3)

    reporter = CLIReporter(
        parameter_columns=[
            "num_layers",
            "hidden_size",
            "embedding_size",
            "dropout_prob",
            "learning_rate",
        ],
        metric_columns=[
            "loss",
            "val_loss",
            "ppl",
            "val_ppl",
            "training_iteration",
        ],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_lm_for_tuning,
        vocab_size=vocab_size,
        lm_model_class=lm_model_class,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    resources_per_trial = {"cpu": 8, "gpu": 1}

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            # num_samples=100,
            max_concurrent_trials=1,
        ),
        run_config=air.RunConfig(
            name=f"tune_lm",
            progress_reporter=reporter,
            verbose=2,
        ),
        # param_space=param_space,
        param_space=lm_model_config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    return results.get_best_result().config
