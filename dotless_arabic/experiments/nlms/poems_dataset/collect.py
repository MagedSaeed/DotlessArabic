import sys
from pathlib import Path

import datasets
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.utils import log_to_file


def collect_dataset(log_steps=True):

    if log_steps:
        current_dir = Path(__file__).resolve().parent

        dotted_results_file_path = f"{current_dir}/results_dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

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

    if log_steps:
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
            lambda bait: 60 > len(bait.replace(" ", "")) >= 5,
            baits,
        )
    )

    if log_steps:
        log_to_file(
            text=f"""
            Number of baits after deleting 60>len(bait)>=5 chars:
            {len(baits):,}
            """,
            results_file=dotted_results_file_path,
        )

    return baits
