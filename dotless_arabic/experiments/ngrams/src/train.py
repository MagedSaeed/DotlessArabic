import os
import json
from tqdm.auto import tqdm
from farasa.segmenter import FarasaSegmenter
from sklearn.model_selection import train_test_split

from dotless_arabic.utils import execute_bash, log_content
from dotless_arabic.experiments.ngrams.src import constants
from dotless_arabic.tokenizers import FarasaMorphologicalTokenizer
from dotless_arabic.experiments.ngrams.src.settings import configure_environment
from dotless_arabic.experiments.ngrams.src.utils import (
    estimate_memory_to_use_by_lm_modeler,
    extract_ngram_counts,
)


def train_lm_model(
    ngram,
    train_dataset,
    dataset_name,
    use_tqdm=True,
    overwrite_files=True,
    print_to_console=True,
    delete_train_file=False,
):
    # useful resources
    # https://github.com/alvations/usaarhat-repo/blob/master/Ken-Options.md
    execute_bash(
        f"mkdir -p {constants.LMs_DIR}/LMs/{dataset_name}/{ngram}",
        print_to_console=print_to_console,
    )
    if (
        not os.path.isfile(f"{constants.LMs_DIR}/LMs/{dataset_name}/train_dataset.txt")
        or overwrite_files
    ):
        print(" Writing train_dataset into file ".center(80, "#"))
        train_dataset = tqdm(train_dataset) if use_tqdm else train_dataset
        with open(
            f"{constants.LMs_DIR}/LMs/{dataset_name}/train_dataset.txt",
            "w",
        ) as train_dataset_file:
            for item in train_dataset:
                train_dataset_file.write(item)
                train_dataset_file.write("\n")
    memory_to_use = estimate_memory_to_use_by_lm_modeler(
        margin=50
    )  # increase the margin not to overhead the workstation
    execute_bash(
        f"kenlm/build/bin/lmplz -o {ngram} -S {memory_to_use}% --discount_fallback < {constants.LMs_DIR}/LMs/{dataset_name}/train_dataset.txt > {constants.LMs_DIR}/LMs/{dataset_name}/{ngram}/lm.arpa",
        print_to_console=print_to_console,
    )
    print("#" * 80)
    print(" Converting .arpa file to binary file. ".center(80, "#"))
    execute_bash(
        f"kenlm/build/bin/build_binary {constants.LMs_DIR}/LMs/{dataset_name}/{ngram}/lm.arpa {constants.LMs_DIR}/LMs/{dataset_name}/{ngram}/lm.binary",
        print_to_console=print_to_console,
    )
    if delete_train_file:
        execute_bash(
            f"rm {constants.LMs_DIR}/LMs/{dataset_name}/train_dataset.txt",
            print_to_console=print_to_console,
        )


def get_perplexity_and_OOVs(
    ngram,
    test_dataset,
    dataset_name,
    overwrite_files=False,
    print_to_console=True,
):
    if (
        not os.path.isfile(f"{constants.LMs_DIR}/LMs/{dataset_name}/test_dataset.txt")
        or overwrite_files
    ):
        with open(
            f"{constants.LMs_DIR}/LMs/{dataset_name}/test_dataset.txt", "w"
        ) as test_dataset_file:
            test_dataset_file.write("\n".join(test_dataset))

    # calculate and dump to a file
    execute_bash(
        f"kenlm/build/bin/query {constants.LMs_DIR}/LMs/{dataset_name}/{ngram}/lm.binary < {constants.LMs_DIR}/LMs/{dataset_name}/test_dataset.txt > {constants.LMs_DIR}/LMs/{dataset_name}/results.txt",
        print_to_console=print_to_console,
    )

    with open(f"{constants.LMs_DIR}/LMs/{dataset_name}/results.txt") as f:
        lines = f.read().splitlines()

    # collect
    perplexity_with_OOVs_line = lines[-4]
    perplexity_without_OOVs_line = lines[-3]
    counts_of_OOVs_line = lines[-2]

    perplexity_with_OOVs = float(
        perplexity_with_OOVs_line.split("Perplexity including OOVs:")[-1].strip()
    )
    perplexity_without_OOVs = float(
        perplexity_without_OOVs_line.split("Perplexity excluding OOVs:")[-1].strip()
    )
    counts_of_OOVs = int(counts_of_OOVs_line.split("OOVs:")[-1].strip())

    return perplexity_with_OOVs, perplexity_without_OOVs, counts_of_OOVs


def get_model_results(dataset_name, ngram, test_dataset):
    # arpa_path = f"{constants.LMs_DIR}/LMs/{dataset_name}/{ngram}/lm.arpa"
    # calcuate ppl and OOVs
    # lm_model = kenlm.Model(arpa_path)
    (
        perplexity_with_OOVs,
        perplexity_without_OOVs,
        counts_of_OOVs,
    ) = get_perplexity_and_OOVs(
        test_dataset=test_dataset,
        ngram=ngram,
        dataset_name=dataset_name,
    )
    # calculate counts for each ngram
    ngram_counts = extract_ngram_counts(
        arpa_path=f"{constants.LMs_DIR}/LMs/{dataset_name}/{ngram}/lm.arpa"
    )
    return dict(
        perplexity_with_OOVs=perplexity_with_OOVs,
        perplexity_without_OOVs=perplexity_without_OOVs,
        counts_of_OOVs=f"{counts_of_OOVs:,}",
        ngram_counts=f"{ngram_counts:,}",
    )


def train_collect_for_ngram(
    ngram,
    train_dataset,
    test_dataset,
    dataset_name,
    print_to_console=False,
):
    print("processing ngram:", ngram)
    print("#" * 50)
    print("Training")
    train_lm_model(
        ngram=ngram,
        dataset_name=dataset_name,
        train_dataset=train_dataset,
        print_to_console=print_to_console,
    )
    print("collecting results")
    results = get_model_results(
        ngram=ngram,
        dataset_name=dataset_name,
        test_dataset=test_dataset,
    )
    execute_bash(
        f"rm -r {constants.LMs_DIR}/LMs/{dataset_name}",
        print_to_console=print_to_console,
    )
    return results


def training_pipeline(
    dataset,
    dataset_name,
    tokenizer_class,
    is_dotted=True,
    results_file=None,
):
    configure_environment()
    log_content(
        content=f"""
        Some of the Dataset Samples before tokenization:
        {constants.NEW_LINE.join(dataset[:5])}
        """,
        results_file=results_file,
    )

    log_content(
        "Tokenize the dataset",
        results_file=results_file,
    )

    if tokenizer_class == FarasaMorphologicalTokenizer:
        segmenter = FarasaSegmenter(interactive=True)
        dataset = list(
            map(
                lambda item: " ".join(
                    tokenizer_class.split_text(
                        item,
                        segmenter=segmenter,
                        undot_text=not is_dotted,
                    )
                ),
                tqdm(dataset),
            ),
        )
    else:
        dataset = list(
            map(
                lambda item: " ".join(
                    tokenizer_class.split_text(
                        item,
                        undot_text=not is_dotted,
                    )
                ),
                tqdm(dataset),
            ),
        )
    log_content(
        content=f"""
        Some of the Dataset Samples after tokenization:
        {constants.NEW_LINE.join(dataset[:5])}
        """,
        results_file=results_file,
    )
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=0.1,
        shuffle=True,
        random_state=constants.RANDOM_SEED,
    )
    results = {}
    results[dataset_name] = {}
    log_content(
        content="TRAINING STARTED",
        results_file=results_file,
    )
    for index, ngram in enumerate(constants.NGRAMS):
        print_to_console = False
        if index == len(constants.NGRAMS) - 1:
            print_to_console = True
        results[dataset_name][ngram] = train_collect_for_ngram(
            ngram=ngram,
            dataset_name=dataset_name,
            test_dataset=test_dataset,
            train_dataset=train_dataset,
            print_to_console=print_to_console,
        )
    log_content(
        # content=pprint.pformat(results, indent=4),
        content=json.dumps(
            results,
            indent=4,
            # ensure_ascii=False,
        ),
        strip_lines=False,
        results_file=results_file,
    )
    log_content(
        content="TRAINING FINISHED",
        results_file=results_file,
    )
    return results
