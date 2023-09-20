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

from dotless_arabic.experiments.topic_modeling.src.models import (
    LitTopicModelingModel,
)


def train_topic_modeler_for_tuning(
    config,
    vocab_size,
    train_dataloader,
    val_dataloader,
    max_epochs=20,
):
    lm_model = LitTopicModelingModel(
        vocab_size=vocab_size,
        **config,
    )

    callbacks = list()
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.05,
        patience=20,
        check_finite=True,
    )
    callbacks.append(early_stopping_callback)
    callbacks.append(
        TuneReportCallback(
            {
                "val_loss": "val_loss",
                "val_acc": "val_acc",
            },
            on="validation_end",
        )
    )
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


def tune_topic_modeling_model(
    vocab_size,
    train_dataloader,
    val_dataloader,
):
    topic_modeling_model_config = {
        "num_layers": tune.grid_search([2, 3]),
        "rnn_hiddens": tune.grid_search([128, 256, 512]),
        "rnn_dropout": tune.grid_search([0.25, 0.5, 0.6]),
        "dropout_prob": tune.grid_search([0.25, 0.5, 0.6]),
        "embedding_size": tune.grid_search([128, 256, 512]),
        "learning_rate": tune.grid_search([0.01, 0.001]),
    }

    scheduler = ASHAScheduler(grace_period=10)

    reporter = CLIReporter(
        parameter_columns=[
            "num_layers",
            "rnn_hiddens",
            "rnn_dropout",
            "dropout_prob",
            "dropout_prob",
            "embedding_size",
            "learning_rate",
        ],
        metric_columns=[
            "loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "training_iteration",
        ],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_topic_modeler_for_tuning,
        vocab_size=vocab_size,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    resources_per_trial = {"cpu": 4, "gpu": 1}

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
            name=f"tune_topic_modeler",
            progress_reporter=reporter,
            verbose=2,
        ),
        # param_space=param_space,
        param_space=topic_modeling_model_config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    return results.get_best_result().config
