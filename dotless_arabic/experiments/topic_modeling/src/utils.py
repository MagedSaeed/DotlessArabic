import math
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

from farasa.segmenter import FarasaSegmenter
from dotless_arabic.processing.processing import process

from dotless_arabic.tokenizers import (
    WordTokenizer,
    FarasaMorphologicalTokenizer,
)
from dotless_arabic.experiments.topic_modeling.src import constants


def get_vocab_size(
    dataset,
    use_tqdm=True,
    undot_text=False,
    return_all_vocab_size=True,
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
    if return_all_vocab_size:
        return vocab, len(sorted_words_frequencies.keys())
    return vocab


def train_topic_modeler(
    topic_modeler,
    wandb_logger,
    tokenizer_class,
    train_dataloader,
    val_dataloader,
    gpu_devices,
    callbacks=None,
    one_run=False,
    text_type="dotted",
    use_rich_progressbar=True,
    max_epochs=constants.MAX_EPOCHS,
):
    # remove any previous checkpoints
    checkpoints_path = Path(f"TM/{text_type}/{tokenizer_class.__name__}/checkpoints")
    if callbacks is None:
        callbacks = []
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
        filename="{epoch}-{val_loss:.3f}-{step}",
        dirpath=f"TM/{text_type}/{tokenizer_class.__name__}/checkpoints",
    )
    callbacks.append(checkpoint_callback)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.005,
        patience=40,
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
    trainer = Trainer(
        devices=list(map(int, gpu_devices.split(","))),
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=1,
        fast_dev_run=one_run,
        max_epochs=max_epochs,
        val_check_interval=0.25,
        accelerator=constants.LIGHTING_ACCELERATOR,
        log_every_n_steps=max(len(train_dataloader) // 25, 1),
    )
    trainer.validate(
        model=topic_modeler,
        dataloaders=val_dataloader,
    )
    trainer.fit(
        topic_modeler,
        train_dataloader,
        val_dataloader,
    )
    return trainer


def get_oovs_rate(
    dataloader,
    unk_token_id=0,
    pad_token_id=1,
):
    dataset = dataloader.dataset
    oovs = 0
    tokens_sum = 0
    for document in dataset.encoded_dataset:
        for token_id in document:
            if token_id == unk_token_id:
                oovs += 1
            elif token_id in [pad_token_id]:
                continue
            tokens_sum += 1
    return f"{(oovs / tokens_sum) * 100:.2f}"


def get_sequence_length(
    dataset,
    use_tqdm=True,
    percentile=constants.SEQUENCE_LENGTH_PERCENTILE,
):
    """
    The dataset is expected to be tokenized
    extra token arg is used to control special tokens.
    In our case, there are two: <bos> and <eos>
    """
    lengths = []
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        lengths.append(len(document))
    lengths = sorted(lengths)
    lengths_count = len(lengths)
    percentile_index = math.floor(lengths_count * percentile)
    length = lengths[percentile_index]
    return length


def filter_empty_items(xs, ys):
    assert len(xs) == len(ys), "xs and ys must have the same length"
    filtered_x, filtered_y = list(), list()
    for x, y in tqdm(zip(xs, ys), total=len(xs)):
        processed_x = process(x)
        if len(processed_x) > 0:
            filtered_x.append(x)
            filtered_y.append(y)
    return filtered_x, filtered_y


def get_balanced_sub_dataset(xs, ys, classes_threshold=1000):
    assert len(xs) == len(ys), "xs and ys must have the same length"
    classes_counts = defaultdict(int)
    sub_xs, sub_ys = list(), list()
    for x, y in tqdm(zip(xs, ys), total=len(xs)):
        if classes_counts[y] > classes_threshold:
            continue
        sub_xs.append(x)
        sub_ys.append(y)
        classes_counts[y] += 1
    return sub_xs, sub_ys
