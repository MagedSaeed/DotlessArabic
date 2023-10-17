import math
from pathlib import Path
import shutil
import numpy as np

import torch

from tqdm.auto import tqdm

from torchmetrics.text import SacreBLEUScore

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    LearningRateMonitor,
)

from dotless_arabic.experiments.translation.src.processing import (
    process_en,
    process_ar,
)
from dotless_arabic.processing.processing import undot

from dotless_arabic.tokenizers import WordTokenizer, SentencePieceTokenizer
from dotless_arabic.experiments.translation.src import constants


def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)


def train_translator(
    translator,
    wandb_logger,
    train_dataloader,
    val_dataloader,
    gpu_devices,
    source_lang,
    target_lang,
    source_tokenizer_class,
    target_tokenizer_class,
    callbacks=None,
    one_run=False,
    text_type="dotted",
    validate_and_fit=True,
    use_rich_progressbar=True,
    max_epochs=constants.MAX_EPOCHS,
):
    # remove any previous checkpoints
    checkpoints_dir = f"NMT/{source_lang}_to_{target_lang}/{source_tokenizer_class.__name__}_to_{target_tokenizer_class.__name__}/{text_type}/checkpoints"
    checkpoints_path = Path(checkpoints_dir)
    if callbacks is None:
        callbacks = []
    if validate_and_fit:
        shutil.rmtree(checkpoints_path, ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        mode="min",
        save_top_k=1,
        verbose=False,
        save_last=True,
        monitor="val_loss",
        dirpath=checkpoints_dir,
        save_weights_only=False,
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        filename="{epoch}-{val_loss:.3f}-{step}",
    )
    callbacks = list()
    callbacks.append(checkpoint_callback)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        # min_delta=0.025,
        min_delta=0,
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
    xavier_init(model=translator)
    trainer = Trainer(
        deterministic=True,
        callbacks=callbacks,
        gradient_clip_val=1,
        max_epochs=max_epochs,
        val_check_interval=0.25,
        # check_val_every_n_epoch=1,
        accelerator="auto",
        logger=wandb_logger,
        fast_dev_run=one_run,
        devices=list(map(int, gpu_devices.split(","))),
        # log_every_n_steps=max(len(train_dataloader) // 25, 1),
        log_every_n_steps=max(len(train_dataloader) // 25, 1),
    )
    if validate_and_fit:
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
    text_list, tokenizer, sequence_length, undot_text=False
):
    encoded = list()
    for doc in tqdm(text_list):
        if undot_text:
            doc = undot(doc)
        encoded_doc = [
            tokenizer.token_to_id(token) for token in tokenizer.split_text(doc)
        ]
        encoded_doc.insert(0, tokenizer.token_to_id("<bos>"))
        encoded_doc.insert(-1, tokenizer.token_to_id("<eos>"))
        encoded_doc = tokenizer.pad(encoded_doc, length=sequence_length)
        encoded_doc = encoded_doc[:sequence_length]
        if encoded_doc[-1] != tokenizer.token_to_id(tokenizer.pad_token):
            encoded_doc[-1] = tokenizer.token_to_id("<eos>")
        encoded.append(np.array(encoded_doc))
    return np.array(encoded)


def get_source_tokenizer(
    train_dataset,
    source_language_code,
    undot_text=False,
    tokenizer_class=WordTokenizer,
):
    if tokenizer_class == SentencePieceTokenizer:
        tokenizer = tokenizer_class(
            vocab_size=4_000,
            special_tokens=["<bos>", "<eos>"],
        )
    else:
        tokenizer = tokenizer_class(
            vocab_size=10**10,
            special_tokens=["<bos>", "<eos>"],
        )
    if undot_text:
        tokenizer.train(
            text="\n".join(tqdm(map(undot, train_dataset[source_language_code])))
        )
    else:
        tokenizer.train(text="\n".join(tqdm(train_dataset[source_language_code])))
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
    undot_text=False,
    tokenizer_class=WordTokenizer,
):
    if tokenizer_class == SentencePieceTokenizer:
        tokenizer = tokenizer_class(
            vocab_size=4_000,
            special_tokens=["<bos>", "<eos>"],
        )
    else:
        tokenizer = tokenizer_class(
            vocab_size=10**10,
            special_tokens=["<bos>", "<eos>"],
        )
    if undot_text:
        tokenizer.train(
            text="\n".join(tqdm(map(undot, train_dataset[target_language_code])))
        )
    else:
        tokenizer.train(text="\n".join(tqdm(train_dataset[target_language_code])))
    # delete 1-freq words if tokenizer class is not Sentencepiece
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


def get_blue_score(
    model,
    source_sentences,
    target_sentences,
    source_tokenizer,
    target_tokenizer,
    max_sequence_length,
    blue_n_gram=4,
    use_tqdm=True,
    show_translations_for=100,
):
    # source_sentences = source_sentences[:100]
    # target_sentences = target_sentences[:100]
    source_sentences = [
        f"<bos> {' '.join(source_tokenizer.split_text(sentence))} <eos>"
        for sentence in source_sentences
    ]
    targets = [[f"<bos> {sentence} <eos>"] for sentence in target_sentences]
    if use_tqdm:
        source_sentences = tqdm(source_sentences)
        target_sentences = tqdm(target_sentences)
    predictions = [
        model.translate(
            input_sentence=sentence,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            max_sequence_length=max_sequence_length,
        )
        for sentence in source_sentences
    ]
    # predictions = list(map(lambda text: text.replace("+ ", ""), predictions))
    # targets = list(
    #     map(
    #         lambda texts: [text.replace("+ ", "") for text in texts],
    #         targets,
    #     )
    # )
    # predictions = list(map(lambda text: text.replace("+ ", ""), predictions))
    # targets = list(
    #     map(
    #         lambda texts: [text.replace("+ ", "") for text in texts],
    #         targets,
    #     )
    # )
    for i, (s, p, t) in enumerate(zip(source_sentences, predictions, targets)):
        print(f"Source: {s}")
        print(f"Prediction: {p}")
        print(f"Target: {t}")
        print("*" * 80)
        if i > show_translations_for:
            break
    sacre_bleu_calculator = SacreBLEUScore(n_gram=blue_n_gram)
    sacre_blue_score = sacre_bleu_calculator(predictions, targets)
    print("sacre blue", sacre_blue_score)
    return sacre_blue_score


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
