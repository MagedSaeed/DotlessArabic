import os
import shutil
import tkseem as tk
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from farasa.segmenter import FarasaSegmenter

import wandb
from dotless_arabic.experiments.nlms.src import constants, datasets


def generate_text(
    lm_model,
    tokenizer,
    sequence_length,
    prompt="",
    num_tokens=10,
    device=constants.DEVICE,
    print_generated_token=True,
    temperature=0.75,
):
    lm_model.to(device)
    lm_model.eval()

    generated_text = prompt

    with torch.no_grad():
        hiddens = None
        for index in range(num_tokens):
            prompt = " ".join(prompt.split()[-sequence_length:])
            encoded = tokenizer.encode(prompt)
            encoded = torch.LongTensor(encoded).to(device)
            output, hiddens = lm_model(encoded, hiddens)
            output = output.squeeze(0)
            if len(output.size()) > 1:
                output = output[-1]
            output = torch.softmax(output / temperature, dim=-1)
            predicted_token_id = torch.multinomial(output, num_samples=1).item()
            counter = 0
            while (
                predicted_token_id
                in [
                    tokenizer.token_to_id(tokenizer.pad_token),
                    tokenizer.token_to_id(tokenizer.unk_token),
                    tokenizer.token_to_id("<eos>"),
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
                generated_text += f" {predicted_token}\n"
            else:
                prompt += f" {predicted_token}"
                generated_text += f" {predicted_token}"
            print("prompt is:", prompt)
    return tokenizer.detokenize(generated_text.split())


def train_lm(
    lm_model,
    dataset_id,
    wandb_logger,
    vocab_coverage,
    tokenizer_class,
    train_dataloader,
    val_dataloader,
    callbacks=[],
    use_rich_progressbar=True,
    max_epochs=constants.MAX_EPOCHS,
    one_run=False,
):
    # remove any previous checkpoints
    checkpoints_path = Path(f"NLMs/{dataset_id}")
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
        dirpath=f"NLMs/{dataset_id}/checkpoints",
        filename="{epoch}-{val_loss:.2f}-{step}"
        + f"-{tokenizer_class.__name__}-{vocab_coverage}",
    )
    callbacks.append(checkpoint_callback)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.005,
        patience=20,
        check_finite=True,
    )
    callbacks.append(early_stopping_callback)
    if use_rich_progressbar:
        callbacks.append(RichProgressBar())
    trainer = Trainer(
        accelerator="auto",
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger,
        fast_dev_run=one_run,
        max_epochs=max_epochs,
        val_check_interval=0.5,
        devices=constants.GPU_DEVICES,
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
    wandb.finish()
    # trainer.lm_model = LitNeurlaLanguageModel.load_from_checkpoint(f'Neural-Language-Models-Metrics/{dataset_id}/checkpoints/last.ckpt')
    return trainer


def calculate_perplexity(
    lm_model,
    dataset,
    tokenizer,
    sequence_length,
    use_tqdm=True,
    device=constants.DEVICE,
    batch_size=constants.BATCH_SIZE,
    ignore_oovs=False,
):
    # https://towardsdatascience.com/the-relationship-between-perplexity-and-entropy-in-nlp-f81888775ccc
    # https://stackoverflow.com/a/59219379/4412324
    lm_dataset = datasets.LanguageModelDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    dataloader = DataLoader(
        shuffle=False,
        dataset=lm_dataset,
        batch_size=batch_size,
        num_workers=constants.CPU_COUNT,
        collate_fn=datasets.dataset_collate_fn,
        drop_last=True if len(lm_dataset) > batch_size else False,
    )
    lm_model.to(device)
    lm_model.eval()
    with torch.no_grad():
        loader = tqdm(dataloader) if use_tqdm else dataloader
        losses = list()
        for batch in loader:
            inputs, outputs = batch
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            batch_loss = lm_model._get_loss(
                (inputs, outputs),
                ignore_oovs=ignore_oovs,
                loss_reduction="none",
            )
            batch_loss = batch_loss.view(-1)
            batch_loss = batch_loss[batch_loss != 0]
            losses.extend(batch_loss.detach().cpu().tolist())

    average_loss = np.mean(losses)
    perplexity = np.exp(average_loss)
    return perplexity


def get_tokenizer(
    train_dataset,
    vocab_size,
    tokenizer_class=constants.DEFAULT_TOKENIZER_CLASS,
):
    with open("tmp_dataset.txt", "w") as f:
        f.write("\n".join(item for item in train_dataset if item.strip()))

    # with open("tmp_dataset.txt") as f:
    #     vocab_size = len(set(
    #         word.strip()
    #         for word in f.read().split()
    #         if word.strip()
    #     ))
    tokenizer = tokenizer_class(
        vocab_size=vocab_size,
        special_tokens=["<bos>", "<eos>"],
    )
    tokenizer.train("tmp_dataset.txt")
    return tokenizer


def get_dataloader(
    dataset,
    tokenizer,
    sequence_length,
    shuffle=False,
    drop_last=True,
    workers=constants.CPU_COUNT,
    batch_size=constants.BATCH_SIZE,
):
    lm_dataset = datasets.LanguageModelDataset(
        dataset=dataset,
        tokenizer=tokenizer,
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
    tokenizer_class,
    use_tqdm=True,
    vocab_coverage=constants.DEFAULT_VOCAB_COVERAGE,
):
    tokens_frequencies = defaultdict(int)
    if tokenizer_class == tk.FarasaMorphologicalTokenizer:
        segmenter = FarasaSegmenter(interactive=True)
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        if tokenizer_class == tk.FarasaMorphologicalTokenizer:
            for token in tokenizer_class.split_text(document, segmenter=segmenter):
                tokens_frequencies[token] += 1
        else:
            for token in tokenizer_class.split_text(document):
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
    return vocab


def get_best_checkpoint(dataset_id, checkpoints_base_path="NLMs"):
    for file_name in os.listdir(f"{checkpoints_base_path}/{dataset_id}/checkpoints"):
        if file_name.startswith("epoch"):
            return f"{checkpoints_base_path}/{dataset_id}/checkpoints/{file_name}"


def get_average_sequence_length(dataset, extra_tokens=2):
    """
    extra token arg is used to control special tokens.
    In our case, there are two: <bos> and <eos>
    """
    lengths = 0
    for document in dataset:
        lengths += len(document.split())
    return int(lengths / len(dataset)) + extra_tokens


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
    return float(f"{(oovs / tokens_sum) * 100:.2f}")


def log_to_file(text, results_file, print_to_console=True):
    with open(results_file, "a") as file:
        file.write("#" * 80)
        file.write("\n")
        for line in text.strip().splitlines():
            file.write(line.strip())
            file.write("\n")
        file.write("#" * 80)
        file.write("\n")
    if print_to_console:
        print("#" * 80)
        for line in text.strip().splitlines():
            print(line.strip())
        print("#" * 80)
