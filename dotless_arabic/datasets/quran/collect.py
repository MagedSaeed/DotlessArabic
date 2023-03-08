import sys, os
from pathlib import Path

if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.utils import log_content


def collect_dataset_for_analysis(results_file=None):
    current_dir = Path(__file__).resolve().parent
    if "Arabic-Original.csv" not in os.listdir(current_dir):
        raise FileNotFoundError("Dataset file 'Arabic-Original.csv' does not exist!!")
    quran_dataset_path = (
        # f"{PROJECT_ROOT_DIR}/DatasetsAndTokenizers/Quran/Arabic-Original.csv"
        f"{current_dir}/Arabic-Original.csv"
    )
    quran_dataset = [
        line.split("|")[-1] for line in open(quran_dataset_path).read().splitlines()
    ]
    # split on بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ as it appears on every beginning of surah
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
    return quran_dataset


def collect_dataset_for_language_modeling(results_file=None):
    quran_dataset = collect_dataset_for_analysis()
    log_content(
        content=f"""
        Number of samples:
        {len(quran_dataset)}
        """,
        results_file=results_file,
    )

    return quran_dataset
