import sys
import datasets
from tqdm.auto import tqdm


if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic import constants
from dotless_arabic.utils import log_content
from dotless_arabic.processing import process

NON_CLASSICAL_METERS = [
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
    "عامي",
    # None,
]


def collect_dataset_for_analysis(results_file=None):
    ashaar = datasets.load_dataset("arbml/ashaar", split="train")
    verses = list()
    for poem in ashaar["poem verses"]:
        verses.extend(poem)
    return verses


def collect_dataset_for_language_modeling(results_file=None):

    ashaar = datasets.load_dataset("arbml/ashaar", split="train")

    log_content(
        content=f"""
        Number datasets samples:
        {len(ashaar)}
        """,
        results_file=results_file,
    )

    ashaar = ashaar.filter(
        lambda example: example["poem meter"] not in NON_CLASSICAL_METERS
    )

    log_content(
        content=f"""
        Number datasets samples after filtering non accepted meters:
        {len(ashaar)}
        """,
        results_file=results_file,
    )

    baits = list()

    for poem in tqdm(ashaar["poem verses"]):
        index = 1
        for shatr in poem:
            if index % 2 == 0:
                baits.append(f"{prev_shatr} {shatr}")
            else:
                prev_shatr = shatr
            index += 1

    log_content(
        content=f"""
        Sample of datasets samples:
        {constants.NEW_LINE.join(baits[:5])}
        """,
        results_file=results_file,
    )

    log_content(
        content=f"""
        Number of Baits:
        {len(baits):,}
        """,
        results_file=results_file,
    )

    baits = list(
        filter(
            lambda bait: 60 >= len(process(bait).replace(" ", "")) >= 30,
            tqdm(baits),
        )
    )

    log_content(
        content=f"""
        Number of baits after deleting 60>= len(bait) chars >= 30 chars:
        {len(baits):,}
        """,
        results_file=results_file,
    )

    return baits


METER_CLASSES = []


def collect_dataset_for_meter_classification(results_file=None):

    ashaar = datasets.load_dataset("arbml/ashaar", split="train")

    log_content(
        content=f"""
        Number datasets samples:
        {len(ashaar)}
        """,
        results_file=results_file,
    )

    ashaar = ashaar.filter(
        lambda example: example["poem meter"] not in NON_CLASSICAL_METERS + [None]
    )

    log_content(
        content=f"""
        Number datasets samples after filtering non accepted meters:
        {len(ashaar)}
        """,
        results_file=results_file,
    )

    baits_with_classes = dict()

    for poem, meter in tqdm(zip(ashaar["poem verses"], ashaar["poem meter"])):
        index = 1
        for shatr in poem:
            if index % 2 == 0:
                baits_with_classes[f"{prev_shatr} {shatr}"] = meter
            else:
                prev_shatr = shatr
            index += 1

    log_content(
        content=f"""
        Sample of datasets samples:
        {constants.NEW_LINE.join(list(baits_with_classes.keys())[:5])}
        """,
        results_file=results_file,
    )

    log_content(
        content=f"""
        Number of Baits:
        {len(baits_with_classes.keys()):,}
        """,
        results_file=results_file,
    )

    baits_with_classes = {
        bait: meter
        for bait, meter in tqdm(baits_with_classes.items())
        if 60 >= len(process(bait).replace(" ", "")) >= 30
    }

    log_content(
        content=f"""
        Number of baits after deleting 60>= len(bait) chars >= 30 chars:
        {len(baits_with_classes.keys()):,}
        """,
        results_file=results_file,
    )

    return baits_with_classes
