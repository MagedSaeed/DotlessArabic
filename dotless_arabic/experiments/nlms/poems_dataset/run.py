import sys
import click
from pathlib import Path

if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.poems_dataset.collect import collect_dataset
from dotless_arabic.experiments.nlms.run_experiment import run

current_dir = Path(__file__).resolve().parent

dataset = collect_dataset()
dataset_name = "poems_dataset"


@click.command()
@click.option(
    "--tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the dataset",
    type=click.Choice(
        [
            "CharacterTokenizer",
            "DisjointLetterTokenizer",
            "FarasaMorphologicalTokenizer",
            "WordTokenizer",
        ]
    ),
)
@click.option(
    "--vocab_coverage",
    default=constants.DEFAULT_VOCAB_COVERAGE,
    help="Vocab coverage to consider, the tokenizer will consider vocabs that covers this percentage of the running text",
)
def run_experiment(tokenizer_class, vocab_coverage):
    run(
        dataset=dataset,
        sequence_length=18,
        results_dir=current_dir,
        dataset_name=dataset_name,
        vocab_coverage=vocab_coverage,
        tokenizer_class=tokenizer_class,
    )


if __name__ == "__main__":
    run_experiment()
