import math
from pathlib import Path
import shutil
import numpy as np

import torch

from tqdm.auto import tqdm

from torchmetrics.text.bleu import BLEUScore

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    LearningRateMonitor,
)

from dotless_arabic.experiments.translation.src.processing import (
    process_source,
    process_target,
)

from dotless_arabic.tokenizers import WordTokenizer, SentencePieceTokenizer
from dotless_arabic.experiments.translation.src import constants


def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)


def train_translator(
    translator,
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
    checkpoints_path = Path(f"NMT/{text_type}/{tokenizer_class.__name__}/checkpoints")
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
        dirpath=f"NMT/{text_type}/{tokenizer_class.__name__}/checkpoints",
    )
    callbacks = list()
    callbacks.append(checkpoint_callback)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        # min_delta=0.025,
        min_delta=0,
        patience=6,
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
    xavier_init(model=translator)
    trainer = Trainer(
        deterministic=True,
        callbacks=callbacks,
        gradient_clip_val=5,
        max_epochs=max_epochs,
        val_check_interval=0.25,
        accelerator="auto",
        logger=wandb_logger,
        fast_dev_run=one_run,
        devices=list(map(int, gpu_devices.split(","))),
        # log_every_n_steps=max(len(train_dataloader) // 25, 1),
        log_every_n_steps=max(len(train_dataloader) // 25, 1),
    )
    trainer.validate(
        model=translator,
        dataloaders=val_dataloader,
    )
    trainer.fit(
        translator,
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


def create_features_from_text_list(
    text_list,
    tokenizer,
    sequence_length,
    is_source=False,
):
    encoded = list()
    for doc in tqdm(text_list):
        if is_source:
            encoded_doc = tokenizer.encode(process_source(doc))
        else:
            encoded_doc = tokenizer.encode("<bos> " + process_target(doc) + " <eos>")
        encoded_doc = tokenizer.pad(encoded_doc, length=sequence_length)
        encoded_doc = encoded_doc[:sequence_length]
        if not is_source:
            if encoded_doc[-1] != tokenizer.token_to_id(tokenizer.pad_token):
                encoded_doc[-1] = tokenizer.token_to_id("<eos>")
        encoded.append(np.array(encoded_doc))
    return np.array(encoded)


def get_source_tokenizer(
    train_dataset,
    source_language_code,
    tokenizer_class=WordTokenizer,
):
    if tokenizer_class == SentencePieceTokenizer:
        tokenizer = tokenizer_class(vocab_size=10**4)
    else:
        tokenizer = tokenizer_class(
            vocab_size=10**10
        )  # high vocab size, higher than the possible vocab size :)
    tokenizer.train(text="\n".join(train_dataset[source_language_code]))
    # if tokenizer_class != SentencePieceTokenizer:
    #     vocab = {
    #         vocab: freq
    #         for vocab, freq in tokenizer.vocab.items()
    #         if freq > 1 or freq < 0
    #     }
    #     tokenizer.vocab = vocab
    #     tokenizer.vocab_size = len(vocab)
    #     tokenizer.vocab_size
    #     tokenizer.vocab_size
    return tokenizer


def get_target_tokenizer(
    train_dataset,
    target_language_code,
    tokenizer_class=WordTokenizer,
):
    if tokenizer_class == SentencePieceTokenizer:
        tokenizer = tokenizer_class(
            vocab_size=10**4,
            special_tokens=["<bos>", "<eos>"],
        )
    else:
        tokenizer = tokenizer_class(
            vocab_size=10**10,
            special_tokens=["<bos>", "<eos>"],
        )
    tokenizer.train(text="\n".join(train_dataset[target_language_code]))
    # delete 1-freq words if tokenizer class is not Sentencepiece
    if tokenizer_class != SentencePieceTokenizer:
        vocab = {
            vocab: freq
            for vocab, freq in tokenizer.vocab.items()
            if freq > 1 or freq < 0
        }
        tokenizer.vocab = vocab
        tokenizer.vocab_size = len(vocab)
        tokenizer.vocab_size
        tokenizer.vocab_size
    return tokenizer


def get_blue_score(
    model,
    source_sentences,
    target_sentences,
    source_tokenizer,
    target_tokenizer,
    max_sequence_length,
    blue_n_gram=4,
    use_tqdm=True,
):
    if use_tqdm:
        target_sentences = tqdm(target_sentences)
        source_sentences = tqdm(source_sentences)
    targets = [[f"<bos> {sentence} <eos>"] for sentence in target_sentences]
    predictions = [
        model.translate(
            input_sentence=sentence,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            max_sequence_length=max_sequence_length,
        )
        for sentence in source_sentences
    ]
    blue_calculator = BLEUScore(n_gram=blue_n_gram)
    blue_score = blue_calculator(predictions, targets)
    return blue_score


def get_sequence_length(
    dataset,
    use_tqdm=True,
    percentile=constants.SEQUENCE_LENGTH_PERCENTILE,
):
    """
    The dataset is expected to be tokenized.
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
