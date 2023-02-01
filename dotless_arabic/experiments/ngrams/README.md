to run ngrams experiments:

```bash
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=quran --tokenizer_class=WordTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=WordTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=poems --tokenizer_class=WordTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=news --tokenizer_class=WordTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=wikipedia --tokenizer_class=WordTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=aggregated --tokenizer_class=WordTokenizer;


python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=quran --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=poems --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=news --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=wikipedia --tokenizer_class=DisjointLetterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=aggregated --tokenizer_class=DisjointLetterTokenizer;


python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=quran --tokenizer_class=CharacterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=CharacterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=poems --tokenizer_class=CharacterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=news --tokenizer_class=CharacterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=wikipedia --tokenizer_class=CharacterTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=aggregated --tokenizer_class=CharacterTokenizer;


python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=quran --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=poems --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=news --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=wikipedia --tokenizer_class=FarasaMorphologicalTokenizer;
python dotless_arabic/experiments/ngrams/run_experiment.py --dataset=aggregated --tokenizer_class=FarasaMorphologicalTokenizer;
```