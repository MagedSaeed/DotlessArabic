to run any of these experiments and log the terminal output to a log file:

```bash
sccript -c "python3 dotless_arabic/experiments/nlms/{dataset}/run.py" dotless_arabic/experiments/nlms/{dataset}/run.log
```

Note that this will run the dotted and undotted experiments for the selected dataset

### Sample run with different GPUs Devices:

```bash
# on one tmux session:
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=WordTokenizer --gpu_devices=0;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=WordTokenizer --gpu_devices=0 --sequence_length=20;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=WordTokenizer --gpu_devices=0;

# on another tmux session
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=WordTokenizer --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=WordTokenizer --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=aggregated --tokenizer_class=WordTokenizer --gpu_devices=1;
```

The following are the commands for running DisjointLetterTokenizer

```bash
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
```
The following are the  commands for running FarasaMorphologicalTokenizer

```bash
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=FarasaMorphologicalTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=FarasaMorphologicalTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=FarasaMorphologicalTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=FarasaMorphologicalTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=FarasaMorphologicalTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975;
```


The following are the commands for running CharacterTokenizer

```bash
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=CharacterTokenizer --vocab_coverage=1 --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=CharacterTokenizer --vocab_coverage=1 --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=CharacterTokenizer --vocab_coverage=1 --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=CharacterTokenizer --vocab_coverage=1 --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=CharacterTokenizer --vocab_coverage=1 --seqlen_percentile=0.95 --gpu_devices=1;
```

## Commands used to run the final experiments:

```bash
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=WordTokenizer --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=WordTokenizer --gpu_devices=1;
# for poems experiments, we do not need to add sequence length as we are running on poems level, not bait level.
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=WordTokenizer --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=WordTokenizer --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=WordTokenizer --gpu_devices=1;

python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=DisjointLetterTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=DisjointLetterTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=DisjointLetterTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=DisjointLetterTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=DisjointLetterTokenizer --seqlen_percentile=0.975 --gpu_devices=1;

python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=FarasaMorphologicalTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=FarasaMorphologicalTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=FarasaMorphologicalTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=FarasaMorphologicalTokenizer --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=FarasaMorphologicalTokenizer --seqlen_percentile=0.975 --gpu_devices=1;

python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
```

## If you want to clear tuning data while running bulk experiments, you can use a similar bash script to the following depending on the location of ray tuning data (seems to be ~/ray_results for linux):
```bash
rm -r ~/ray_results/tune_lm/
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
rm -r ~/ray_results/tune_lm/
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
rm -r ~/ray_results/tune_lm/
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
rm -r ~/ray_results/tune_lm/
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
rm -r ~/ray_results/tune_lm/
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=CharacterTokenizer --seqlen_percentile=0.95 --gpu_devices=1;
rm -r ~/ray_results/tune_lm/
```


## Commands used to run the final experiments for transformers:

```bash
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=WordTokenizer --gpu_devices=1 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=WordTokenizer --gpu_devices=1 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=WordTokenizer --gpu_devices=1 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=WordTokenizer --gpu_devices=1 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=WordTokenizer --gpu_devices=1 --model_type=transformer;


python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=DisjointLetterTokenizer --gpu_devices=0 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=DisjointLetterTokenizer --gpu_devices=0 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=DisjointLetterTokenizer --gpu_devices=0 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=DisjointLetterTokenizer --gpu_devices=0 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=DisjointLetterTokenizer --gpu_devices=0 --seqlen_percentile=0.975 --model_type=transformer;

python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=FarasaMorphologicalTokenizer --gpu_devices=1 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=FarasaMorphologicalTokenizer --gpu_devices=1 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=FarasaMorphologicalTokenizer --gpu_devices=1 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=FarasaMorphologicalTokenizer --gpu_devices=1 --seqlen_percentile=0.975 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=FarasaMorphologicalTokenizer --gpu_devices=1 --seqlen_percentile=0.975 --model_type=transformer;

python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=CharacterTokenizer --gpu_devices=0 --seqlen_percentile=0.95 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=CharacterTokenizer --gpu_devices=0 --seqlen_percentile=0.95 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=CharacterTokenizer --gpu_devices=0 --seqlen_percentile=0.95 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=CharacterTokenizer --gpu_devices=0 --seqlen_percentile=0.95 --model_type=transformer;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=CharacterTokenizer --gpu_devices=0 --seqlen_percentile=0.95 --model_type=transformer;
```