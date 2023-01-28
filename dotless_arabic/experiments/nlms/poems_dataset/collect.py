import sys
from pathlib import Path

import datasets
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.processing import process
from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.utils import log_to_file


def collect_dataset(log_steps=True):

    if log_steps:
        current_dir = Path(__file__).resolve().parent

        dotted_results_file_path = f"{current_dir}/results_dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

    ashaar = datasets.load_dataset("arbml/ashaar", split="train")

    non_accepted_meters = [
        "التفعيله",
        "الحداء",
        "الدوبيت",
        "السلسلة",
        "الصخري",
        "الكان كان",
        "اللويحاني",
        "المسحوب",
        "المواليا",
        "الموشح",
        "الهجيني",
        "بحر التفعيله",
        "بحر الدوبيت",
        "بحر السلسلة",
        "بحر القوما",
        "بحر المواليا",
        "بحر تفعيلة الرجز",
        "بحر تفعيلة الرمل",
        "بحر تفعيلة الكامل",
        "بحر تفعيلة المتقارب",
        "بحر مجزوء الدوبيت",
        "بحر مجزوء المواليا",
        "بحر مخلع موشح",
        "زجل",
        "شعر التفعيلة",
        "شعر حر",
        "عدة أبحر",
    ]
    ashaar = ashaar.filter(
        lambda example: example["poem meter"] not in non_accepted_meters
    )

    poems = list()

    for poem in tqdm(ashaar["poem verses"]):
        clean_poem = list()
        for shatr in poem:
            processed_shatr = process(shatr)
            if (
                len(processed_shatr.replace(" ", "")) < 10
                or len(processed_shatr.replace(" ", "")) > 30
            ):
                continue
            if len(processed_shatr) > 0:
                clean_poem.append(processed_shatr)
        if clean_poem:
            poems.append(clean_poem)

    if log_steps:
        log_to_file(
            text=f"""
            Sample of datasets samples:
            {constants.NEW_LINE.join(poems[0][:5])}
            """,
            results_file=dotted_results_file_path,
        )

    log_to_file(
        text=f"""
        Number of Poems:
        {len(poems):,}
        """,
        results_file=dotted_results_file_path,
    )

    baits = list()
    for poem in tqdm(poems):
        bait = ""
        for i, shatr in enumerate(poem, start=1):
            bait += f"{shatr} "
            if i % 2 == 0:
                baits.append(bait.strip())
                bait = ""

    if log_steps:
        log_to_file(
            text=f"""
            Number of baits after deleting 30>= len(shatr) chars >= 10 chars:
            {len(baits):,}
            """,
            results_file=dotted_results_file_path,
        )

    return baits
