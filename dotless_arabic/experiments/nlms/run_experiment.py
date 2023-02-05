import sys
import click
import tkseem as tk
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import log_content
from dotless_arabic.tokenizers import TOKENIZERS_MAP
from dotless_arabic.processing import process, undot
from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.training_pipeline import training_pipeline


@click.command()
@click.option(
    "--tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the dataset",
    type=click.Choice(list(TOKENIZERS_MAP.keys())),
)
@click.option(
    "--vocab_coverage",
    default=constants.DEFAULT_VOCAB_COVERAGE,
    help="Vocab coverage to consider, the tokenizer will consider vocabs that covers this percentage of the running text",
)
@click.option(
    "--dataset",
    help="dataset name to train. Note that results files will be saved in '{current_dir_of_this_file}'/{dataset}_dataset/",
    required=True,
    type=click.Choice(list(constants.COLLECT_DATASET.keys())),
)
@click.option(
    "--sequence_length",
    help="sequence length to consider when tokenizing dataset samples",
    type=int,
    default=None,
)
@click.option(
    "--gpu_devices",
    help="GPU devices indexes to consider if the machine has GPUs. Expected input to be index,index...",
    type=str,
    default=str(constants.GPU_DEVICES),
)
@click.option(
    "--cpu_devices",
    help="CPU devices (processes) to consider if the code will run on CPU",
    type=int,
    default=constants.CPU_DEVICES,
)
@click.option(
    "--batch_size",
    help="Batch size to consider in various data setups",
    type=int,
    default=constants.DEFAULT_BATCH_SIZE,
)
@click.option(
    "--seqlen_percentile",
    help="Sequence Length Percentile. That is, you would be sure that this percentile of your samples are completely covered",
    type=float,
    default=constants.SEQUENCE_LENGTH_PERCENTILE,
)
def run(
    dataset,
    vocab_coverage,
    tokenizer_class,
    gpu_devices,
    cpu_devices,
    batch_size,
    sequence_length=None,
    seqlen_percentile=constants.SEQUENCE_LENGTH_PERCENTILE,
):

    dataset_name = dataset + "_dataset"
    sequence_length_percentile = seqlen_percentile

    gpu_devices = list(map(int, gpu_devices.split(",")))

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/results/{dataset_name}"

    # create results dir if not exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_class = TOKENIZERS_MAP[tokenizer_class]

    dotted_results_file_path = f"{results_dir}/results_dotted_tokenizer_{tokenizer_class.__name__}_vocab_coverage_{vocab_coverage}.txt"

    dataset = constants.COLLECT_DATASET[dataset](results_file=dotted_results_file_path)

    # delete the current logging file, if exists

    Path(dotted_results_file_path).unlink(missing_ok=True)

    log_content(
        content=f"""
            Dotted Training Started at {datetime.now()} for tokenizer: {tokenizer_class.__name__}
            """,
        results_file=dotted_results_file_path,
    )

    dataset = list(
        map(
            process,
            tqdm(dataset),
        ),
    )

    log_content(
        content=f"""
        Some of the Dataset Samples before training:
        {constants.NEW_LINE.join(dataset[:5])}
        """,
        results_file=dotted_results_file_path,
    )

    ################################################
    ###### Dotted Dataset Training #################
    ################################################

    dataset_id = f"dotted-{dataset_name}".upper()

    training_pipeline(
        is_dotted=True,
        dataset=dataset,
        batch_size=batch_size,
        dataset_id=dataset_id,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        vocab_coverage=vocab_coverage,
        dataset_name=dataset_id.lower(),
        tokenizer_class=tokenizer_class,
        sequence_length=sequence_length,
        results_file=dotted_results_file_path,
        sequence_length_percentile=sequence_length_percentile,
    )

    log_content(
        content=f"""
        Dotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=dotted_results_file_path,
    )

    ################################################
    ###### Undotted Dataset Training ###############
    ################################################

    undotted_results_file_path = f"{results_dir}/results_undotted_tokenizer_{tokenizer_class.__name__}_vocab_coverage_{vocab_coverage}.txt"

    # delete the current logging file, if exists

    Path(undotted_results_file_path).unlink(missing_ok=True)

    log_content(
        content=f"""
        Undotted Training Started at {datetime.now()} for tokenizer: {tokenizer_class.__name__}
        """,
        results_file=undotted_results_file_path,
    )

    dataset_id = f"undotted-{dataset_name}".upper()

    log_content(
        content=f"""
        Some of the Dataset Samples after undotting:
        {constants.NEW_LINE.join(dataset[:5])}
        """,
        results_file=undotted_results_file_path,
    )

    training_pipeline(
        is_dotted=False,
        dataset=dataset,
        dataset_id=dataset_id,
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        vocab_coverage=vocab_coverage,
        dataset_name=dataset_id.lower(),
        tokenizer_class=tokenizer_class,
        sequence_length=sequence_length,
        results_file=undotted_results_file_path,
        sequence_length_percentile=sequence_length_percentile,
    )

    log_content(
        content=f"""
        Undotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=undotted_results_file_path,
    )


if __name__ == "__main__":
    run()
