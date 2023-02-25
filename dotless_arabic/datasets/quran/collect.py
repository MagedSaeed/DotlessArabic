import sys


if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.utils import log_content


def collect_raw_dataset():
    quran_dataset_path = (
        # f"{PROJECT_ROOT_DIR}/DatasetsAndTokenizers/Quran/Arabic-Original.csv"
        f"dotless_arabic/datasets/quran/Arabic-Original.csv"
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
    quran_dataset = collect_raw_dataset()
    log_content(
        content=f"""
        Number of samples:
        {len(quran_dataset)}
        """,
        results_file=results_file,
    )

    return quran_dataset
