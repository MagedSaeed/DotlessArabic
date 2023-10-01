import os
import tempfile
import zipfile
import requests
from tqdm.auto import tqdm

import datasets

from dotless_arabic.utils import log_content

"""
This script downloads and prepare arbml (https://github.com/ARBML/ARBML/tree/master/datasets/Translation) subset of open_subtitle. 
"""

DEFAULT_CACHE_DIR = tempfile.gettempdir()
DATASET_URL = "https://raw.githubusercontent.com/ARBML/ARBML/master/datasets/Translation/translation.zip"

DATASET_FOLDER = DEFAULT_CACHE_DIR + "/dotless-arabic"


def download_and_cache_dataset(
    unzip=True,
    log_file=None,
    url=DATASET_URL,
    cache_dir=DEFAULT_CACHE_DIR,
    dataset_folder=DATASET_FOLDER,
):
    try:
        # Get the system's default temporary directory

        # Extract the filename from the URL
        file_name = "open_subtitles.zip"
        file_path = os.path.join(cache_dir, file_name)

        # Check if the file already exists in the cache
        print(file_path)
        if os.path.exists(file_path):
            log_content(
                content=f"Retrieving Dataset ZIP file from cache",
                results_file=log_file,
            )
        else:
            # Download the file
            log_content(
                content=f"Downloading Dataset ZIP File from {url}",
                results_file=log_file,
            )
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 KB

                # Create a tqdm progress bar
                progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

                # Save the file to the cache directory while updating the progress bar
                with open(file_path, "wb") as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        progress_bar.update(len(data))

                # Close the progress bar
                progress_bar.close()
            else:
                print(
                    f"Failed to download file from {url} (status code: {response.status_code})."
                )
                return
        if unzip:
            if not os.path.exists(dataset_folder):
                log_content(content=f"UNZIPPING", results_file=log_file)
                with zipfile.ZipFile(file_path, "r") as zip_file:
                    zip_file.extractall(dataset_folder)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def collect_parallel_dataset_for_translation(split="train"):
    download_and_cache_dataset()
    ar_sentences_path = DATASET_FOLDER + "/translation/" + f"ar_data_{split}.txt"
    en_sentences_path = DATASET_FOLDER + "/translation/" + f"en_data_{split}.txt"
    with open(ar_sentences_path) as file:
        ar_sentences = file.read().splitlines()
    with open(en_sentences_path) as file:
        en_sentences = file.read().splitlines()
    parallel_dataset = datasets.Dataset.from_dict(
        dict(
            ar=ar_sentences,
            en=en_sentences,
        )
    )
    return parallel_dataset


def collect_parallel_train_dataset_for_translation():
    return collect_parallel_dataset_for_translation(split="train")


def collect_parallel_test_dataset_for_translation():
    return collect_parallel_dataset_for_translation(split="test")
