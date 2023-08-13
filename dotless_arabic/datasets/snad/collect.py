import datasets

from dotless_arabic import constants
from dotless_arabic.utils import log_content
from dotless_arabic.processing import process


def collect_dataset_for_analysis(results_file=None):
    ashaar = datasets.load_dataset("arbml/ashaar", split="train")
    verses = list()
    for poem in ashaar["poem verses"]:
        verses.extend(poem)
    return verses




def collect_dataset_for_meter_classification(results_file=None):

    snad = datasets.load_dataset("arbml/snad", split="train")

    log_content(
        content=f"""
        Number datasets samples:
        {len(snad)}
        """,
        results_file=results_file,
    )

    snad = snad.filter(lambda item:len(item['title_details']) > 0)

    log_content(
        content=f"""
        Dataset length after filtering out empty items:
        {len(snad)}
        """,
        results_file=results_file,
    )

    dataset = {
        item['title_details']:item['title_category']
        for item in snad
    }

    log_content(
        content=f"""
        Sample of datasets samples:
        {constants.NEW_LINE.join(document for document in list(dataset.keys())[:2])}
        """,
        results_file=results_file,
    )

    log_content(
        content=f"""
        Number of Baits:
        {len(baits_with_meter_names.keys()):,}
        """,
        results_file=results_file,
    )

#     baits_with_meter_names = {
#         bait: meter
#         for bait, meter in tqdm(baits_with_meter_names.items())
#         if 60 >= len(process(bait).replace(" ", "")) >= 20
#     }

#     log_content(
#         content=f"""
#         Number of baits after deleting 60>= len(bait) chars >= 20 chars:
#         {len(baits_with_meter_names.keys()):,}
#         """,
#         results_file=results_file,
#     )

    log_content(
        content=f"""
        Map meter names to classes:
        """,
        results_file=results_file,
    )

    baits_with_meter_classes = map_meter_names_to_classes(
        baits_dict=baits_with_meter_names
    )

    return baits_with_meter_classes