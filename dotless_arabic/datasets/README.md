to collect statistics about datasets:

```bash
python dotless_arabic/datasets/collect_statistics.py --dataset quran;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth;
python dotless_arabic/datasets/collect_statistics.py --dataset poems;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia;
python dotless_arabic/datasets/collect_statistics.py --dataset news;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated;
```

The above commands will collect statistics for word tokenizer as it is the default when the tokenizer_class option is not passed.

You can use the flag `--redo_counts` to redo the calculations of tokens frequencies. Usually, these frequencies are calculated then saved as json files in `tokens_count` folder in the same level of the dataset's `collect.py` file. However, with this flag, they are recalculated again. Useful if one of the functionalities changed in the pipeline, i.e. `process` function changed or `tokenization split_text` method changed.

To collect statistics for other tokenizers, starting by farasa tokenizer:

```bash
python dotless_arabic/datasets/collect_statistics.py --dataset=quran --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset poems --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset news --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated --tokenizer_class=FarasaMorphologicalTokenizer;
```

disjoint tokenizer:

```bash
python dotless_arabic/datasets/collect_statistics.py --dataset=quran --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset poems --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset news --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated --tokenizer_class=DisjointLetterTokenizer;
```

characters tokenizer:

```bash
python dotless_arabic/datasets/collect_statistics.py --dataset=quran --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset poems --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset news --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated --tokenizer_class=CharacterTokenizer;
```

to run them all in one shot:

```bash
python dotless_arabic/datasets/collect_statistics.py --dataset quran;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth;
python dotless_arabic/datasets/collect_statistics.py --dataset poems;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia;
python dotless_arabic/datasets/collect_statistics.py --dataset news;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated;
python dotless_arabic/datasets/collect_statistics.py --dataset=quran --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset poems --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset news --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset=quran --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset poems --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset news --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated --tokenizer_class=CharacterTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset=quran --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset sanadset_hadeeth --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset poems --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset wikipedia --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset news --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/datasets/collect_statistics.py --dataset aggregated --tokenizer_class=FarasaMorphologicalTokenizer;
```
