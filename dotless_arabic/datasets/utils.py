from functools import lru_cache
from tqdm.auto import tqdm
from collections import defaultdict

################################################
############ statistics Functions ##############
################################################

get_unique_samples_count = lambda dataset: len(dataset)


@lru_cache()
def tokens_frequency(dataset, use_tqdm=True):
    frequencies = defaultdict(int)
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        for token in document.split():
            frequencies[token] += 1
    frequencies = dict(frequencies)
    return frequencies


get_unique_vocab_count = lambda dataset: len(tokens_frequency(tuple(dataset)).keys())

get_all_tokens_count = lambda dataset: sum(tokens_frequency(tuple(dataset)).values())
