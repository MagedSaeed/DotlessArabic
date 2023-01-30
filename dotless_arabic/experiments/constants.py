from dotless_arabic.constants import *

from dotless_arabic.datasets.quran.collect import (
    collect_dataset as collect_quran_dataset,
)
from dotless_arabic.datasets.sanadset_hadeeth.collect import (
    collect_dataset as collect_sanadset_hadeeth_dataset,
)
from dotless_arabic.datasets.poems.collect import (
    collect_dataset as collect_poems_dataset,
)
from dotless_arabic.datasets.news.collect import (
    collect_dataset as collect_news_dataset,
)
from dotless_arabic.datasets.wikipedia.collect import (
    collect_dataset as collect_wikipedia_dataset,
)
from dotless_arabic.datasets.aggregated.collect import (
    collect_dataset as collect_aggregated_dataset,
)


COLLECT_DATASET = {
    "quran": collect_quran_dataset,
    "sanadset_hadeeth": collect_sanadset_hadeeth_dataset,
    "poems": collect_poems_dataset,
    "news": collect_news_dataset,
    "wikipedia": collect_wikipedia_dataset,
    "aggregated": collect_aggregated_dataset,
}

DEFAULT_TOKENIZER_CLASS = "WordTokenizer"
