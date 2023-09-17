import os

from collections import defaultdict
from pathlib import Path
import shutil

from tqdm.auto import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    LearningRateMonitor,
)

import wandb

from farasa.segmenter import FarasaSegmenter

from dotless_arabic.tokenizers import (
    CharacterTokenizer,
    WordTokenizer,
    FarasaMorphologicalTokenizer,
)
from dotless_arabic.experiments.meter_classification.src import constants


# def get_best_checkpoint(
#     text_type,
#     dataset_name,
#     cap_threshold,
#     checkpoints_base_path="MC",
#     tokenizer_class=CharacterTokenizer,
# ):
#     checkpoints_path = f"{checkpoints_base_path}/{dataset_name}/{text_type}/{tokenizer_class.__name__}/{cap_threshold}/checkpoints"
#     for file_name in os.listdir(checkpoints_path):
#         if file_name.startswith("epoch"):
#             return f"{checkpoints_path}/{file_name}"


def get_vocab_size(
    dataset,
    use_tqdm=True,
    undot_text=False,
    return_all_vocab=True,
    tokenizer_class=WordTokenizer,
    vocab_coverage=constants.DEFAULT_VOCAB_COVERAGE,
):
    tokens_frequencies = defaultdict(int)
    if tokenizer_class == FarasaMorphologicalTokenizer:
        segmenter = FarasaSegmenter(interactive=True)
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        if tokenizer_class == FarasaMorphologicalTokenizer:
            for token in tokenizer_class.split_text(
                document,
                segmenter=segmenter,
                undot_text=undot_text,
            ):
                tokens_frequencies[token] += 1
        else:
            for token in tokenizer_class.split_text(
                document,
                undot_text=undot_text,
            ):
                tokens_frequencies[token] += 1

    sorted_words_frequencies = dict(
        sorted(
            tokens_frequencies.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    current_words_count = 0
    vocab = 0
    all_words_counts = sum(sorted_words_frequencies.values())
    for token, counts in sorted_words_frequencies.items():
        current_words_count += counts
        current_coverage = current_words_count / all_words_counts
        vocab += 1
        if current_coverage > vocab_coverage:
            break
    if return_all_vocab:
        return vocab, len(sorted_words_frequencies.keys())
    return vocab


def train_meter_classifier(
    meter_classifier,
    dataset_name,
    wandb_logger,
    tokenizer_class,
    train_dataloader,
    val_dataloader,
    gpu_devices,
    cpu_devices,
    callbacks=[],
    one_run=False,
    text_type="dotted",
    cap_threshold="all",
    use_rich_progressbar=True,
    max_epochs=constants.MAX_EPOCHS,
):
    # remove any previous checkpoints
    checkpoints_path = Path(
        f"MC/{dataset_name}/{text_type}/{tokenizer_class.__name__}/{cap_threshold}/checkpoints"
    )
    shutil.rmtree(checkpoints_path, ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        mode="min",
        save_top_k=1,
        verbose=False,
        save_last=True,
        monitor="val_loss",
        save_weights_only=False,
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        dirpath=f"MC/{dataset_name}/{text_type}/{tokenizer_class.__name__}/{cap_threshold}/checkpoints",
        filename="{epoch}-{val_loss:.3f}-{step}-size-{cap_threshold}",
    )
    callbacks.append(checkpoint_callback)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.025,
        patience=10,
        check_finite=True,
    )
    callbacks.append(early_stopping_callback)
    if use_rich_progressbar:
        callbacks.append(RichProgressBar())
    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )
    callbacks.append(lr_monitor)
    devices = gpu_devices if constants.DEVICE == "cuda" else cpu_devices
    trainer = Trainer(
        devices=devices,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=1,
        fast_dev_run=one_run,
        max_epochs=max_epochs,
        val_check_interval=0.5,
        accelerator=constants.LIGHTING_ACCELERATOR,
        log_every_n_steps=max(len(train_dataloader) // 25, 1),
    )
    trainer.validate(
        model=meter_classifier,
        dataloaders=val_dataloader,
    )
    trainer.fit(
        meter_classifier,
        train_dataloader,
        val_dataloader,
    )
    return trainer


def cap_baits_dataset(dataset, threshold):
    capped_dataset = {}
    meters_counts = defaultdict(int)
    for bait, meter_class in tqdm(dataset.items()):
        if meters_counts[meter_class] >= threshold:
            continue
        capped_dataset[bait] = meter_class
        meters_counts[meter_class] += 1
    return capped_dataset
