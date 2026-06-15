import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader

from dotless_arabic.experiments.translation.src import constants
from dotless_arabic.experiments.translation.src.utils import (
    create_features_from_text_list,
)


def get_dataloader(
    dataset,
    undot_text,
    source_language_code,
    target_language_code,
    source_tokenizer,
    target_tokenizer,
    target_sequence_length,
    source_sequence_length=None,
    use_tqdm=True,
    shuffle=False,
    drop_last=False,
    batch_size=constants.DEFAULT_BATCH_SIZE,
):
    source_dataset = (
        tqdm(dataset[source_language_code])
        if use_tqdm
        else dataset[source_language_code]
    )
    undot_text_source = undot_text if source_language_code == "ar" else False
    undot_text_target = undot_text if target_language_code == "ar" else False
    target_dataset = (
        tqdm(dataset[target_language_code])
        if use_tqdm
        else dataset[target_language_code]
    )

    encoder_inputs = create_features_from_text_list(
        add_bos=True,
        add_eos=True,
        text_list=source_dataset,
        tokenizer=source_tokenizer,
        undot_text=undot_text_source,
        sequence_length=source_sequence_length,
    )
    decoder_inputs = create_features_from_text_list(
        # add_eos=False,
        text_list=target_dataset,
        tokenizer=target_tokenizer,
        undot_text=undot_text_target,
        sequence_length=target_sequence_length,
    )
    decoder_targets = create_features_from_text_list(
        text_list=target_dataset,
        tokenizer=target_tokenizer,
        undot_text=undot_text_target,
        sequence_length=target_sequence_length,
    )

    # encoded_source_dataset = create_features_from_text_list(
    #     add_bos=False,
    #     add_eos=False,
    #     text_list=source_dataset,
    #     tokenizer=source_tokenizer,
    #     undot_text=undot_text_source,
    #     sequence_length=source_sequence_length,
    # )
    # encoded_target_dataset = create_features_from_text_list(
    #     text_list=target_dataset,
    #     tokenizer=target_tokenizer,
    #     undot_text=undot_text_target,
    #     sequence_length=target_sequence_length,
    # )
    # torch_dataset = TensorDataset(
    #     torch.from_numpy(encoded_source_dataset),
    #     torch.from_numpy(encoded_target_dataset),
    # )
    torch_dataset = TensorDataset(
        torch.from_numpy(encoder_inputs),
        torch.from_numpy(decoder_inputs),
        torch.from_numpy(decoder_targets),
    )
    dataloader = DataLoader(
        torch_dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        batch_size=batch_size,
        num_workers=constants.CPU_COUNT,
    )
    return dataloader
