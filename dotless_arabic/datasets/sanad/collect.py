from collections import defaultdict
import os
import zipfile
import requests
import tempfile
from tqdm import tqdm
from dotless_arabic.utils import log_content

DEFAULT_CACHE_DIR = tempfile.gettempdir()
BALANCED_DATASET_URL = "https://data.mendeley.com/public-files/datasets/57zpx667y9/files/bb58cb62-e41c-46c7-9744-59c22a2cadb1/file_downloaded"

DATASET_FOLDER = DEFAULT_CACHE_DIR + "/sanad"


def download_and_cache_dataset(
    url=BALANCED_DATASET_URL,
    cache_dir=DEFAULT_CACHE_DIR,
    dataset_folder=DATASET_FOLDER,
    unzip=True,
    log_file=None,
):
    try:
        # Get the system's default temporary directory

        # Extract the filename from the URL
        file_name = "balanced_sanad.zip"
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


def get_dataset_split(
    dataset_folder=DATASET_FOLDER,
    split="Train",
    newspapers=("khaleej",),
):
    accepted_newspapers = ("khaleej", "arabiya", "akhbarona")
    assert set(newspapers).issubset(
        set(accepted_newspapers)
    ), f"newspapers should be part of {accepted_newspapers}"
    download_and_cache_dataset()
    dataset = dict()
    for newspaper in newspapers:
        articles_classes_path = dataset_folder + "/" + newspaper + "/" + split
        for article_class_index, article_class in enumerate(
            os.listdir(articles_classes_path)
        ):
            for file in os.listdir(articles_classes_path + "/" + article_class):
                with open(
                    articles_classes_path + "/" + article_class + "/" + file,
                    "r",
                ) as article_file:
                    dataset[article_file.read()] = article_class_index
    return dict(dataset)


def collect_train_dataset_for_topic_modeling(newspapers=("khaleej",)):
    return get_dataset_split(split="Train", newspapers=newspapers)


def collect_test_dataset_for_topic_modeling(newspapers=("khaleej",)):
    return get_dataset_split(split="Test", newspapers=newspapers)
