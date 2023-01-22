from pathlib import Path

from tqdm.auto import tqdm

from dotless_arabic.experiments.nlms.src.training_pipeline import training_pipeline
from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.processing import process, undot

################################################
############# Dataset Preparation ##############
################################################

quran_dataset_path = (
    # f"{PROJECT_ROOT_DIR}/DatasetsAndTokenizers/Quran/Arabic-Original.csv"
    f"dotless_arabic/experiments/nlms/quran_dataset/Arabic-Original.csv"
)
quran_dataset = [
    line.split("|")[-1] for line in open(quran_dataset_path).read().splitlines()
]
# split on بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ as it appears on every begining of surah
basmala = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
no_basmala_quran_dataset = []
for ayiah in quran_dataset:
    if ayiah.startswith(basmala) and len(ayiah) > len(basmala):
        ayiah = ayiah[len(basmala) :].strip()
        if ayiah:
            no_basmala_quran_dataset.append(ayiah)
    else:
        no_basmala_quran_dataset.append(ayiah.strip())
quran_dataset = no_basmala_quran_dataset

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

dataset_name = "quran_dataset"
dataset_id = f"dotted-{dataset_name}".upper()

dataset = list(
    map(
        process,
        tqdm(quran_dataset),
    ),
)

newline = "\n"

log_to_file(
    text=f"""
    Dataset Sample:
    {newline.join(dataset[:5])}
    """,
    results_file=dotted_results_file_path,
)

################################################
###### Dotted Dataset Training #################
################################################

training_pipeline(
    dataset=dataset,
    dataset_name="dotted-quran_dataset",
    dataset_id="dotted-qurand_dataset".upper(),
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

undotted_dataset = list(
    map(
        undot,
        tqdm(dataset),
    ),
)

log_to_file(
    text=f"""
    Undotted Dataset Sample:
    {newline.join(undotted_dataset[:5])}
    """,
    results_file=undotted_results_file_path,
)

training_pipeline(
    dataset=undotted_dataset,
    dataset_name="undotted-quran_dataset",
    dataset_id="undotted-qurand_dataset".upper(),
    is_dotted=False,
    results_file=undotted_results_file_path,
)

log_to_file(
    text=f"""
    Undotted Training Finished
    """,
    results_file=undotted_results_file_path,
)
