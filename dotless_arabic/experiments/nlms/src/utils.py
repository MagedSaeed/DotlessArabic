import os
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
    StochasticWeightAveraging,
    LearningRateFinder,
)


from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from farasa.segmenter import FarasaSegmenter

from dotless_arabic.processing import undot
from dotless_arabic.experiments.nlms.src import constants, datasets
from dotless_arabic.tokenizers import (
    FarasaMorphologicalTokenizer,
    WordTokenizer,
    SentencePieceTokenizer,
)
from dotless_arabic.experiments.nlms.src.models import LitRNNLM, LitTransformerLM


def generate_text(
    lm_model,
    tokenizer,
    sequence_length,
    prompt="<bos> ",
    num_tokens=25,
    device=constants.DEVICE,
    print_generated_token=True,
    temperature=0.5,
):
    lm_model.to(device)
    lm_model.eval()

    generated_text = prompt

    with torch.no_grad():
        hiddens = None
        for index in range(num_tokens):
            # prompt = " ".join(prompt.split()[-sequence_length:])
            # encoded = tokenizer.encode(prompt)
            encoded = [tokenizer.token_to_id(token) for token in prompt.split()]
            encoded = torch.LongTensor([encoded]).to(device)
            if lm_model.__class__ == LitRNNLM:
                output, hiddens = lm_model(encoded, hiddens)
            elif lm_model.__class__ == LitTransformerLM:
                output = lm_model(encoded, device=device)
            output = output[-1, -1, :]
            output = torch.softmax(output / temperature, dim=-1)
            predicted_token_id = torch.multinomial(output, num_samples=1).item()
            counter = 0
            while (
                predicted_token_id
                in [
                    # tokenizer.token_to_id(tokenizer.pad_token),
                    tokenizer.token_to_id(tokenizer.unk_token),
                    tokenizer.token_to_id(
                        "<bos>"
                    ),  # this is in the case where the unk got replaced by bos
                ]
                and counter < 100
            ):
                predicted_token_id = torch.multinomial(
                    output,
                    num_samples=1,
                ).item()
                counter += 1

            predicted_token = tokenizer.id_to_token(predicted_token_id)
            if print_generated_token:
                print("predicting:", predicted_token)
            if predicted_token == "<eos>":
                prompt = "<bos> "
                hiddens = None
                generated_text += f" {predicted_token}\n<bos> "
            else:
                prompt += f" {predicted_token}"
                generated_text += f" {predicted_token}"
            print("prompt is:", prompt)
    return "\n".join(
        # tokenizer.detokenize(line.split()) for line in generated_text.splitlines()
        line.replace(" ", "").replace("<##>", " ")
        for line in generated_text.splitlines()
    )


def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def train_lm(
    lm_model,
    model_type,
    dataset_id,
    wandb_logger,
    vocab_coverage,
    tokenizer_class,
    train_dataloader,
    val_dataloader,
    gpu_devices,
    # cpu_devices,
    callbacks=[],
    one_run=False,
    use_rich_progressbar=True,
    max_epochs=constants.MAX_EPOCHS,
):
    # remove any previous checkpoints
    checkpoints_path = Path(
        f"NLMs/{dataset_id}/{tokenizer_class.__name__}/{model_type}"
    )
    shutil.rmtree(checkpoints_path, ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        mode="min",
        save_top_k=1,
        verbose=False,
        save_last=True,
        monitor="val_loss",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        dirpath=f"NLMs/{dataset_id}/{tokenizer_class.__name__}/{model_type}/checkpoints",
        filename="{epoch}-{val_loss:.3f}-{step}" + f"-{vocab_coverage}",
    )
    callbacks.append(checkpoint_callback)
    # learning rate finder
    # callbacks.append(
    #     LearningRateFinder(
    #         min_lr=1e-08,
    #         max_lr=1,
    #         num_training_steps=1_000,
    #         mode="exponential",
    #         early_stop_threshold=20,
    #         update_attr=True,
    #         attr_name="learning_rate",
    #     )
    # )
    # callbacks.append(StochasticWeightAveraging(0.1 * constants.LEARNING_RATE))
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        check_finite=True,
        min_delta=0.05,
        patience=20,
    )
    callbacks.append(early_stopping_callback)
    if use_rich_progressbar:
        callbacks.append(RichProgressBar())
    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )
    callbacks.append(lr_monitor)
    # devices = gpu_devices.split(",") if constants.DEVICE == "cuda" else cpu_devices
    # initialze the model with xavier initialization
    xavier_init(model=lm_model)
    # raise
    trainer = Trainer(
        devices=list(map(int, gpu_devices.split(","))),
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        # gradient_clip_val=3,  # for transformer model, this value might be very small, say 0.25
        gradient_clip_val=0,  # for RNNs, this value was 3
        fast_dev_run=one_run,
        max_epochs=max_epochs,
        val_check_interval=0.25,
        accelerator=constants.LIGHTING_ACCELERATOR,
        log_every_n_steps=max(len(train_dataloader) // 25, 1),
        # default_root_dir=f"LMsModels/{previous_hiddens}",
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

    # trainer.lm_model = LitNeurlaLanguageModel.load_from_checkpoint(f'Neural-Language-Models-Metrics/{dataset_id}/checkpoints/last.ckpt')
    return trainer


# def calculate_perplexity(
#     lm_model,
#     dataset,
#     tokenizer,
#     sequence_length,
#     use_tqdm=True,
#     undot_text=False,
#     device=constants.DEVICE,
#     batch_size=constants.DEFAULT_BATCH_SIZE,
#     ignore_oovs=False,
# ):
#     # https://towardsdatascience.com/the-relationship-between-perplexity-and-entropy-in-nlp-f81888775ccc
#     # https://stackoverflow.com/a/59219379/4412324
#     lm_dataset = datasets.LanguageModelDataset(
#         dataset=dataset,
#         tokenizer=tokenizer,
#         undot_text=undot_text,
#         sequence_length=sequence_length,
#     )
#     dataloader = DataLoader(
#         shuffle=False,
#         dataset=lm_dataset,
#         batch_size=batch_size,
#         num_workers=constants.CPU_COUNT,
#         collate_fn=datasets.dataset_collate_fn,
#         drop_last=True if len(lm_dataset) > batch_size else False,
#     )
#     lm_model.to(device)
#     lm_model.eval()
#     with torch.no_grad():
#         loader = tqdm(dataloader) if use_tqdm else dataloader
#         losses = list()
#         for batch in loader:
#             inputs, outputs = batch
#             inputs = inputs.to(device)
#             outputs = outputs.to(device)
#             batch_loss = lm_model._get_loss(
#                 (inputs, outputs),
#                 ignore_oovs=ignore_oovs,
#                 loss_reduction="none",
#             )
#             batch_loss = batch_loss.view(-1)
#             batch_loss = batch_loss[batch_loss != 0]
#             losses.extend(batch_loss.detach().cpu().tolist())

#     average_loss = np.mean(losses)
#     perplexity = np.exp(average_loss)
#     return perplexity


def get_tokenizer(
    train_dataset,
    vocab_size,
    undot_text=False,
    tokenizer_class=constants.DEFAULT_TOKENIZER_CLASS,
):
    dataset = train_dataset
    if undot_text:
        dataset = [undot(item) for item in dataset if item.strip()]
    dataset = "\n".join(item for item in dataset if item.strip())
    tokenizer = tokenizer_class(
        vocab_size=vocab_size,
        special_tokens=["<bos>", "<eos>"],
    )
    tokenizer.train(text=dataset)
    tokenizer.vocab["<##>"] = -1  # for the space character
    tokenizer.vocab_size += 1
    return tokenizer


def get_dataloader(
    dataset,
    tokenizer,
    sequence_length,
    shuffle=False,
    drop_last=True,
    undot_text=False,
    workers=constants.CPU_COUNT,
    batch_size=constants.DEFAULT_BATCH_SIZE,
):
    lm_dataset = datasets.LanguageModelDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        undot_text=undot_text,
        sequence_length=sequence_length,
    )
    dataloader = DataLoader(
        shuffle=shuffle,
        dataset=lm_dataset,
        num_workers=workers,
        drop_last=drop_last,
        batch_size=batch_size,
        collate_fn=datasets.dataset_collate_fn,
    )
    return dataloader


# def get_vocab_size(dataset, freq_threshold=3):
#     words_frequencies = defaultdict(int)
#     for document in dataset:
#         for word in document.split():
#             words_frequencies[word] += 1
#     return len([word for word, freq in words_frequencies.items() if freq >= freq_threshold])


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
                # print(tokenizer_class.split_text(document))
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


def get_best_checkpoint(
    dataset_id, tokenizer_class, checkpoints_base_path="NLMs", model_type="RNN"
):
    checkpoints_path = f"{checkpoints_base_path}/{dataset_id}/{tokenizer_class.__name__}/{model_type}/checkpoints"
    for file_name in os.listdir(checkpoints_path):
        if file_name.startswith("epoch"):
            print(f"checkpiont {file_name} found.")
            return f"{checkpoints_path}/{file_name}"
    print("Could NOT find a checkpoint!!")


def get_sequence_length(
    dataset,
    use_tqdm=True,
    extra_tokens=2,
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
    length = lengths[percentile_index] + extra_tokens
    return length


def get_oovs_rate(
    dataloader,
    unk_token_id=0,
    pad_token_id=1,
    special_tokens_ids=[2, 3],
):
    dataset = dataloader.dataset
    oovs = 0
    tokens_sum = 0
    for document in dataset.encoded_dataset:
        for token_id in document:
            if token_id == unk_token_id:
                oovs += 1
            elif token_id in [pad_token_id] + special_tokens_ids:
                continue
            tokens_sum += 1
    return f"{(oovs / tokens_sum) * 100:.2f}"
