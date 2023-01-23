import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.training_pipeline import training_pipeline
from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.processing import dataset_dot_transform, process, undot
from dotless_arabic.utils import download_file

################################################
############# Dataset Preparation ##############
################################################

current_dir = Path(__file__).resolve().parent

if "sanadset.csv" not in os.listdir(current_dir):
    download_file(
        file_url="https://data.mendeley.com/public-files/datasets/5xth87zwb5/files/fe0857fa-73af-4873-b652-60eb7347b811/file_downloaded",
        file_name="sanadset.csv",
        output_dir=current_dir,
    )

sanadset_hadeeth_dataset = list(
    pd.read_csv(f"{current_dir}/sanadset.csv")["Matn"].dropna()
)
sanadset_hadeeth_dataset = sorted(list(set(sanadset_hadeeth_dataset)))


dotted_results_file_path = f"{current_dir}/results_dotted.txt"

# delete the current logging file, if exists

Path(dotted_results_file_path).unlink(missing_ok=True)

log_to_file(
    text=f"""
    Dotted Training Started
    """,
    results_file=dotted_results_file_path,
)

log_to_file(
    text=f"""
    Sample of datasets documents:
    {constants.NEW_LINE.join(sanadset_hadeeth_dataset[:5])}
    """,
    results_file=dotted_results_file_path,
)
log_to_file(
    text=f"""
    Original Number of Samples:
    {len(sanadset_hadeeth_dataset):,}
    """,
    results_file=dotted_results_file_path,
)

dataset_name = "sanadset_hadeeth_dataset"

dataset = list(set(sanadset_hadeeth_dataset))

log_to_file(
    text=f"""
    Number of Samples after dropping duplicates:
    {len(dataset):,}
    """,
    results_file=dotted_results_file_path,
)

dataset = dataset_dot_transform(tqdm(sanadset_hadeeth_dataset))


log_to_file(
    text=f"""
    Number of Samples after splitting on dots:
    {len(dataset):,}
    """,
    results_file=dotted_results_file_path,
)

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
