import sys
from pathlib import Path


if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.experiments.nlms.src.utils import log_to_file


def collect_dataset(log_steps=True):

    if log_steps:
        current_dir = Path(__file__).resolve().parent

        dotted_results_file_path = f"{current_dir}/results_dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

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

    if log_steps:
        log_to_file(
            text=f"""
            Number of samples:
            {len(quran_dataset)}
            """,
            results_file=dotted_results_file_path,
        )

    return quran_dataset
