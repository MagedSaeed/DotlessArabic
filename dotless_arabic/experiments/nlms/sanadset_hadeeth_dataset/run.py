import sys
from pathlib import Path

import tkseem as tk

if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.experiments.nlms.run_experiment import run
from dotless_arabic.experiments.nlms.sanadset_hadeeth_dataset.collect import (
    collect_dataset,
)

current_dir = Path(__file__).resolve().parent

dataset = collect_dataset()
dataset_name = "sanadset_hadeeth_dataset"
tokenizer_class = tk.WordTokenizer

run(
    dataset=dataset,
    results_dir=current_dir,
    dataset_name=dataset_name,
    tokenizer_class=tokenizer_class,
)
