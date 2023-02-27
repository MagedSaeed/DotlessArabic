from dotless_arabic.constants import *

from dotless_arabic.datasets.quran.collect import (
    collect_dataset_for_language_modeling as collect_quran_dataset,
    collect_dataset_for_analysis as collect_quran_dataset_for_analysis,
)
from dotless_arabic.datasets.sanadset_hadeeth.collect import (
    collect_dataset_for_language_modeling as collect_sanadset_hadeeth_dataset,
    collect_dataset_for_analysis as collect_sanadset_hadeeth_dataset_for_analysis,
)
from dotless_arabic.datasets.poems.collect import (
    collect_dataset_for_language_modeling as collect_poems_dataset,
    collect_dataset_for_analysis as collect_poems_dataset_for_analysis,
)
from dotless_arabic.datasets.news.collect import (
    collect_dataset_for_language_modeling as collect_news_dataset,
    collect_dataset_for_analysis as collect_news_dataset_for_analysis,
)
from dotless_arabic.datasets.wikipedia.collect import (
    collect_dataset_for_language_modeling as collect_wikipedia_dataset,
    collect_dataset_for_analysis as collect_wikipedia_dataset_for_analysis,
)

from dotless_arabic.datasets.aggregated.collect import (
    collect_dataset as collect_aggregated_dataset,
    collect_dataset_for_analysis as collect_aggregated_dataset_for_analysis,
)


COLLECT_DATASET_FOR_LANGUAGE_MODELLING = {
    "quran": collect_quran_dataset,
    "sanadset_hadeeth": collect_sanadset_hadeeth_dataset,
    "poems": collect_poems_dataset,
    "news": collect_news_dataset,
    "wikipedia": collect_wikipedia_dataset,
    "aggregated": collect_aggregated_dataset,
}


COLLECT_DATASET_FOR_ANALYSIS = {
    "quran": collect_quran_dataset_for_analysis,
    "sanadset_hadeeth": collect_sanadset_hadeeth_dataset_for_analysis,
    "poems": collect_poems_dataset_for_analysis,
    "news": collect_news_dataset_for_analysis,
    "wikipedia": collect_wikipedia_dataset_for_analysis,
    "aggregated": collect_aggregated_dataset_for_analysis,
}


DEFAULT_TOKENIZER_CLASS = "WordTokenizer"
