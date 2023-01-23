import sys
from pathlib import Path

import datasets
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.training_pipeline import training_pipeline
from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.processing import process, undot

################################################
############# Dataset Preparation ##############
################################################

current_dir = Path(__file__).resolve().parent

dotted_results_file_path = f"{current_dir}/results_dotted.txt"

# delete the current logging file, if exists

Path(dotted_results_file_path).unlink(missing_ok=True)

log_to_file(
    text=f"""
    Dotted Training Started
    """,
    results_file=dotted_results_file_path,
)

ashaar = datasets.load_dataset("arbml/ashaar", split="train")

baits = list()

for poem in tqdm(ashaar["poem verses"]):
    index = 1
    for shatr in poem:
        if index % 2 == 0:
            baits.append(f"{prev_shatr} {shatr}")
        else:
            prev_shatr = shatr
        index += 1

log_to_file(
    text=f"""
    Sample of datasets samples:
    {constants.NEW_LINE.join(baits[:5])}
    """,
    results_file=dotted_results_file_path,
)

log_to_file(
    text=f"""
    Number of Baits:
    {len(baits):,}
    """,
    results_file=dotted_results_file_path,
)

baits = list(
    filter(
        lambda bait: 50 > len(bait.replace(" ", "")) >= 10,
        baits,
    )
)

log_to_file(
    text=f"""
    Number of baits after deleting 25>bait>5 chars:
    {len(baits):,}
    """,
    results_file=dotted_results_file_path,
)

dataset_name = "poems_dataset"

dataset = baits

dataset = list(
    map(
        process,
        tqdm(dataset),
    ),
)


log_to_file(
    text=f"""
    Some of the Dataset Samples after processing:
    {constants.NEW_LINE.join(dataset[:5])}
    """,
    results_file=dotted_results_file_path,
)

################################################
###### Dotted Dataset Training #################
################################################

dataset_id = f"dotted-{dataset_name}".upper()

training_pipeline(
    dataset=dataset,
    dataset_name=dataset_id.lower(),
    dataset_id=dataset_id,
    is_dotted=True,
    results_file=dotted_results_file_path,
)

log_to_file(
    text=f"""
    Dotted Training Finished
    """,
    results_file=dotted_results_file_path,
)

################################################
###### Undotted Dataset Training ###############
################################################

undotted_results_file_path = f"{current_dir}/results_undotted.txt"

# delete the current logging file, if exists

Path(undotted_results_file_path).unlink(missing_ok=True)


log_to_file(
    text=f"""
    Undotted Training Started
    """,
    results_file=undotted_results_file_path,
)

dataset_id = f"undotted-{dataset_name}".upper()

undotted_dataset = list(
    map(
        undot,
        dataset,
    )
)


log_to_file(
    text=f"""
    Some of the Dataset Samples after undotting:
    {constants.NEW_LINE.join(undotted_dataset[:5])}
    """,
    results_file=undotted_results_file_path,
)


training_pipeline(
    dataset=undotted_dataset,
    dataset_name=dataset_id.lower(),
    dataset_id=dataset_id,
    is_dotted=False,
    results_file=undotted_results_file_path,
)

log_to_file(
    text=f"""
    Undotted Training Finished
    """,
    results_file=undotted_results_file_path,
)
