import random
import sys
import click
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.datasets.poems.collect import (
    collect_dataset_for_meter_classification,
)

from dotless_arabic.utils import log_content
from dotless_arabic.tokenizers import TOKENIZERS_MAP
from dotless_arabic.processing import process, undot
from dotless_arabic.experiments.meter_classification.src import constants
from dotless_arabic.experiments.meter_classification.src.training_pipeline import (
    training_pipeline,
)
from dotless_arabic.experiments.meter_classification.src.utils import cap_baits_dataset


@click.command()
@click.option(
    "--tokenizer_class",
    help="Tokenizer class to tokenize the dataset",
    type=click.Choice(list(TOKENIZERS_MAP.keys())),
    default=constants.DEFAULT_TOKENIZER_CLASS.__name__,
)
@click.option(
    "--vocab_coverage",
    default=constants.DEFAULT_VOCAB_COVERAGE,
    help="Vocab coverage to consider, the tokenizer will consider vocabs that covers this percentage of the running text",
)
# @click.option(
#     "--dataset",
#     help="dataset name to train. Note that results files will be saved in '{current_dir_of_this_file}'/{dataset}_dataset/",
#     required=True,
#     type=click.Choice(["poems"]),
# )
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
    "--undot_text",
    help="Undot the text. This is False by default",
    type=bool,
    default=False,
)
@click.option(
    "--baits_size",
    help="Number of biats to consider. If left empty, all the dataset will be considered",
    type=int,
    default=0,
)
@click.option(
    "--dropout_prob",
    help="Model dropout. This parameter is sensitive for dotted/dotless text. Defaulted to 0.333",
    type=float,
    default=0.333,
)
def run(
    vocab_coverage,
    tokenizer_class,
    gpu_devices,
    cpu_devices,
    batch_size,
    undot_text,
    baits_size,
    dropout_prob,
):
    dataset_name = "ashaar"
    dataset = collect_dataset_for_meter_classification()

    cap_threshold = "all" if baits_size == 0 else baits_size

    gpu_devices = list(map(int, gpu_devices.split(",")))

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/results/{dataset_name}/{tokenizer_class}/{'dotted' if not undot_text else 'undotted'}"

    # create results dir if not exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_class = TOKENIZERS_MAP[tokenizer_class]

    results_file_path = (
        f"{results_dir}/dropout_{dropout_prob}_baits_size_{cap_threshold}.txt"
    )

    # delete the current logging file, if exists

    Path(results_file_path).unlink(missing_ok=True)

    dataset = collect_dataset_for_meter_classification(results_file=results_file_path)

    log_content(
        content=f"""
        dataset's classes:{set(dataset.values())}
        """,
        results_file=results_file_path,
    )

    log_content(
        content="""filter out empty baits or baits with one verse only.""",
        results_file=results_file_path,
    )

    dataset = {
        bait: meter_class
        for bait, meter_class in tqdm(dataset.items())
        if process(bait[0]).strip() and process(bait[1]).strip()
    }

    log_content(
        content=f"""
        baits in dataset after the last filteration: {len(dataset)}
        """,
        results_file=results_file_path,
    )

    if baits_size:
        dataset = cap_baits_dataset(dataset=dataset, threshold=baits_size)

        log_content(
            content=f"""
            dataset length after capping: {len(dataset)}
            """,
            results_file=results_file_path,
        )

    log_content(
        content="processing the dataset",
        results_file=results_file_path,
    )

    baits = list(
        map(
            lambda bait: f"{process(bait[0])} # {process(bait[1])}",
            tqdm(dataset.keys()),
        )
    )

    if undot_text:
        log_content(
            content="""
            undotting the text
            """,
        )
        baits = list(
            map(
                lambda bait: undot(bait),
                tqdm(baits),
            )
        )

    meters = list(dataset.values())

    log_content(
        content="shuffling the dataset",
        results_file=results_file_path,
    )

    zipped_dataset = list(zip(baits, meters))
    random.Random(constants.RANDOM_SEED).shuffle(zipped_dataset)
    baits, meters = list(map(list, zip(*zipped_dataset)))

    log_content(
        content="split to train/val/test",
        results_file=results_file_path,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        baits,
        meters,
        test_size=0.05,
        random_state=42,
        shuffle=True,
        stratify=meters,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.05,
        random_state=42,
        shuffle=True,
        stratify=y_train,
    )

    log_content(
        content=f"""
    x_train size:{len(x_train)}
    y_train size:{len(y_train)}

    x_val size:{len(x_val)}
    y_val size:{len(y_val)}

    x_test size:{len(x_test)}
    y_test size:{len(y_test)}
    """,
        results_file=results_file_path,
    )

    assert (
        len(x_train) == len(y_train)
        and len(x_test) == len(y_test)
        and len(x_val) == len(y_val)
    ), f"one of the splits of train/val/test are not equal in size. sources(baits) != targets(meters)!"

    log_content(
        content=f"""
            Training Started at {datetime.now()} for tokenizer: {tokenizer_class.__name__}
            """,
        results_file=results_file_path,
    )

    log_content(
        content=f"""
        Some of the Dataset Samples before training:
        {constants.NEW_LINE.join(x_train[:5])}
        """,
        results_file=results_file_path,
    )

    training_pipeline(
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        x_train=x_train,
        y_train=y_train,
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        is_dotted=not undot_text,
        dataset_name=dataset_name,
        dropout_prob=dropout_prob,
        cap_threshold=cap_threshold,
        vocab_coverage=vocab_coverage,
        results_file=results_file_path,
        tokenizer_class=tokenizer_class,
    )

    log_content(
        content=f"""
        Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=results_file_path,
    )


if __name__ == "__main__":
    run()
