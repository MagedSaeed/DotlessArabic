import os
import sys
import json
import click
from pprint import pprint
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import log_content
from dotless_arabic.experiments import constants
from dotless_arabic.tokenizers import TOKENIZERS_MAP
from dotless_arabic.processing import (
    process,
    undot,
    undot_with_letters_random_remapping,
)
from dotless_arabic.datasets.utils import (
    tokens_frequency,
    calculate_entropy,
    tokenize_dataset_for_statistics,
)


@click.command()
@click.option(
    "--dataset",
    help="dataset name to collect statistics for. Note that statistics files will be saved in '{current_dir_of_this_file}'/random_mapping/{dataset}_dataset/",
    required=True,
    type=click.Choice(list(constants.COLLECT_DATASET_FOR_ANALYSIS.keys())),
)
@click.option(
    "--tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the dataset",
    type=click.Choice(list(TOKENIZERS_MAP.keys())),
)
def run(dataset, tokenizer_class):
    dataset_name = dataset + "_dataset"

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/{dataset}/statistics/random_mapping"

    Path(f"{results_dir}").mkdir(parents=True, exist_ok=True)

    tokenizer_class = TOKENIZERS_MAP[tokenizer_class]

    results_file_path = f"{results_dir}/{tokenizer_class.__name__}.json"

    dataset = constants.COLLECT_DATASET_FOR_ANALYSIS[dataset]()[
        :1_000_000
    ]  # get maximum 1M samples, for computing limitations

    log_content(
        content=f"""
            Process the Dataset
            """,
    )

    dataset = list(
        map(
            process,
            tqdm(dataset),
        ),
    )

    log_content(
        content=f"""
            Tokenize the Dataset with {tokenizer_class.__name__}
            """,
    )

    dataset = tokenize_dataset_for_statistics(
        dataset=tuple(dataset),
        tokenizer_class=tokenizer_class,
    )

    log_content(
        content=f"""
        Some of the Dataset Samples before collecting statistics:
        {constants.NEW_LINE.join(dataset[:5])}
        """,
    )

    ################################################
    ###### Dotted Dataset statistics ###############
    ################################################

    log_content(
        content=f"""
            Dotted Statistics Analysis Started at {datetime.now()} for dataset {dataset_name} tokenized by {tokenizer_class.__name__}
            """,
    )

    log_content(
        content=f"""
        Samples Count: {len(dataset):,}
        """,
    )

    counter = tokens_frequency(dataset=tuple(dataset))
    results = {
        "dotted": {
            "vocab": len(counter.keys()),
            "tokens": sum(
                counter.values(),
            ),
        }
    }

    log_content(
        content=f"""
        Unique Vocabulary Count: {results['dotted']['vocab']:,}
        """,
    )

    log_content(
        content=f"""
        All Tokens Count: {results['dotted']['tokens']:,}
        """,
    )

    entropy = calculate_entropy(tokens_frequency=counter)
    results["dotted"]["entropy"] = entropy

    log_content(
        content=f"""
        Tokens Entropy: {entropy:,.4f}
        """,
    )

    results["dotted"]["average_tokens_length"] = sum(map(len, counter.keys())) / len(
        counter.keys()
    )

    threshold = 0.1
    truncated_counter = dict(list(counter.items())[: int(threshold * len(counter))])

    results["dotted"][f"top_{threshold}_average_tokens_length"] = sum(
        map(len, truncated_counter.keys())
    ) / len(truncated_counter.keys())

    log_content(
        content=f"""
        Top {threshold}% average tokens length: {results['dotted'][f'top_{threshold}_average_tokens_length']}
        """,
    )

    log_content(
        content=f"""
        Dotted Statistics Analysis Finished for dataset {dataset_name} tokenized by {tokenizer_class.__name__} at {datetime.now()}
        """,
    )

    ################################################
    ###### Undotted Dataset statistics #############
    ################################################

    log_content(
        content=f"""
        Undotted Statistics Analysis Started at {datetime.now()} for dataset {dataset_name} tokenized by {tokenizer_class.__name__}
        """,
        # results_file=results_file_path,
    )
    log_content(
        content=f"""
            Undotting Dataset
            """,
        # results_file=results_file_path,
    )

    undotted_dataset = list(
        map(
            undot,
            tqdm(dataset),
        )
    )
    log_content(
        content=f"""
            Create an undotted tokens frequency mapping
            """,
        # results_file=results_file_path,
    )

    log_content(
        content=f"""
        Some of the Dataset Samples after undotting:
        {constants.NEW_LINE.join(map(undot,undotted_dataset[:5]))}
        """,
        # results_file=results_file_path,
    )

    undotted_counter = tokens_frequency(dataset=tuple(undotted_dataset))

    log_content(
        content=f"""
        Unique Vocabulary Count: {len(undotted_counter.keys()):,}
        """,
        # results_file=results_file_path,
    )

    log_content(
        content=f"""
        All Undotted Tokens Count: {sum(undotted_counter.values()):,}
        """,
        # results_file=results_file_path,
    )

    log_content(
        content=f"""
        undotted vocab/undotted tokens: {len(undotted_counter.keys())/sum(undotted_counter.values()):,.4f}
        """,
        # results_file=results_file_path,
    )

    undotted_entropy = calculate_entropy(tokens_frequency=undotted_counter)

    log_content(
        content=f"""
        Undotted Tokens Entropy: {undotted_entropy:.4f}
        """,
        # results_file=results_file_path,
    )

    log_content(
        content=f"""
        Average tokens length: {sum(map(len,undotted_counter.keys()))/len(undotted_counter.keys()):,.4f}
        """,
        # results_file=results_file_path,
    )

    threshold = 0.1
    truncated_undotted_counter = dict(
        list(undotted_counter.items())[: int(threshold * len(undotted_counter))]
    )

    log_content(
        content=f"""
        Top {threshold}% average tokens length: {sum(map(len,truncated_undotted_counter.keys()))/len(truncated_undotted_counter.keys()):,.4f}
        """,
        # results_file=results_file_path,
    )

    log_content(
        content=f"""
        dotted voacb - undotted vocab: {len(counter.keys())-len(undotted_counter.keys()):,}
        """,
        # results_file=results_file_path,
    )

    log_content(
        content=f"""
        Undotted Statistics Analysis Finished for dataset {dataset_name} tokenized by {tokenizer_class.__name__} at {datetime.now()}
        """,
        # results_file=results_file_path,
    )
    results["dotless"] = {
        "mapping": constants.LETTERS_MAPPING,
        "vocab": len(undotted_counter.keys()),
        "tokens": sum(undotted_counter.values()),
        "entropy": undotted_entropy,
        "average_tokens_length": sum(map(len, undotted_counter.keys()))
        / len(undotted_counter.keys()),
        f"top_{threshold}_average_tokens_length": sum(
            map(len, truncated_undotted_counter.keys())
        )
        / len(truncated_undotted_counter.keys()),
    }

    ##########################################################################
    ###### Undotted with letters random remapping Dataset statistics #########
    ##########################################################################

    results["random_dotless"] = []
    log_content(
        content=f"""
    Undotted with random letters remapping Statistics Analysis Started at {datetime.now()} for dataset {dataset_name} tokenized by {tokenizer_class.__name__}
    """,
    )
    max_number_of_seeds = 500
    for seed in tqdm(range(max_number_of_seeds), leave=max_number_of_seeds):
        log_content(
            content=f"""
                Undotting Dataset
                """,
            # results_file=results_file_path,
        )

        undotted_dataset = list()

        undotted_first_sample, letters_mapping = undot_with_letters_random_remapping(
            text=dataset[0],
            seed=seed,
            process_first=False,
            return_mapping=True,
        )

        log_content(
            content=f"""
                Undotted first sample
                {undotted_first_sample}
                """,
            # results_file=results_file_path,
        )

        log_content(content=f"mapping:")
        pprint(letters_mapping)

        undotted_dataset += list(
            map(
                lambda text: undot_with_letters_random_remapping(text=text, seed=seed),
                tqdm(dataset[1:]),
            )
        )
        tokens_frequency.cache_clear()
        undotted_counter = tokens_frequency(dataset=tuple(undotted_dataset))

        log_content(
            content=f"""
            Some of the Dataset Samples after undotting:
            {constants.NEW_LINE.join(map(undot,undotted_dataset[:5]))}
            """,
            # results_file=results_file_path,
        )

        undotted_entropy = calculate_entropy(tokens_frequency=undotted_counter)
        average_vocab_count = sum(map(len, undotted_counter.keys())) / len(
            undotted_counter.keys()
        )

        threshold = 0.1
        truncated_undotted_counter = dict(
            list(undotted_counter.items())[: int(threshold * len(undotted_counter))]
        )
        average_top10_vocab_count = sum(
            map(len, truncated_undotted_counter.keys())
        ) / len(truncated_undotted_counter.keys())
        results["random_dotless"].append(
            {
                "seed": seed,
                "mapping": letters_mapping,
                "vocab": len(undotted_counter.keys()),
                "tokens": sum(undotted_counter.values()),
                "entropy": undotted_entropy,
                "average_tokens_length": average_vocab_count,
                "average_top10_tokens_length": average_top10_vocab_count,
            }
        )

    results["random_dotless"].sort(key=lambda x: x["entropy"], reverse=True)

    json.dump(
        results,
        open(results_file_path, "w"),
        indent=4,
        # sort_keys=True,
        ensure_ascii=False,
    )

    log_content(
        content=f"""
        Undotted Statistics Analysis Finished for dataset {dataset_name} tokenized by {tokenizer_class.__name__} at {datetime.now()}
        """,
        # results_file=results_file_path,
    )


if __name__ == "__main__":
    run()
