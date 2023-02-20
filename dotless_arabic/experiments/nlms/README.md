to run any of these experiments and log the terminal output to a log file:

```bash
sccript -c "python3 dotless_arabic/experiments/nlms/{dataset}/run.py" dotless_arabic/experiments/nlms/{dataset}/run.log
```

Note that this will run the dotted and undotted experiments for the selected dataset

### Sample run with different GPUs Devices:

```
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
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=quran --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=sanadset_hadeeth --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=poems --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=wikipedia --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
python dotless_arabic/experiments/nlms/run_experiment.py --dataset=news --tokenizer_class=DisjointLetterTokenizer --vocab_coverage=0.975 --seqlen_percentile=0.975 --gpu_devices=1;
```