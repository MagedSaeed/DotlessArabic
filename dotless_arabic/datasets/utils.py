import math, re
from functools import lru_cache
from tqdm.auto import tqdm
from collections import defaultdict

from farasa.segmenter import FarasaSegmenter

from dotless_arabic.tokenizers import FarasaMorphologicalTokenizer

################################################
############ statistics Functions ##############
################################################


@lru_cache()
def tokens_frequency(dataset, use_tqdm=True):
    frequencies = defaultdict(int)
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        for token in document.split():
            frequencies[token] += 1
    frequencies = dict(frequencies)
    return frequencies


def calculate_entropy(tokens_frequency):
    # https://stackoverflow.com/q/43419803/4412324
    # https://stackoverflow.com/a/40496783/4412324
    total_number_of_tokens = sum(tokens_frequency.values())
    entropy = -sum(
        (word_frequency / total_number_of_tokens)
        * math.log2(word_frequency / total_number_of_tokens)
        for word_frequency in tokens_frequency.values()
    )
    return entropy


@lru_cache()
def tokenize_dataset_for_statistics(dataset, tokenizer_class):
    tokenized_dataset = list()
    if tokenizer_class == FarasaMorphologicalTokenizer:
        segmenter = FarasaSegmenter(interactive=True)
    for document in tqdm(dataset):
        if tokenizer_class == FarasaMorphologicalTokenizer:
            tokenized_document = " ".join(
                tokenizer_class.split_text(
                    document,
                    segmenter=segmenter,
                )
            )
        else:
            tokenized_document = " ".join(tokenizer_class.split_text(document))
        tokenized_document = tokenized_document.replace("<##>", "")
        tokenized_document = re.sub("\s+", " ", tokenized_document)
        tokenized_dataset.append(tokenized_document)
    return tokenized_dataset
