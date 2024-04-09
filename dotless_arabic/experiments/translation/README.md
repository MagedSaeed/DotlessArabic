```python
python dotless_arabic/experiments/translation/run_experiment.py --gpu_devices=0 --source_lang=ar --target_lang=en --source_tokenizer_class=SentencePieceTokenizer --target_tokenizer_class=SentencePieceTokenizer;
python dotless_arabic/experiments/translation/run_experiment.py --gpu_devices=0 --source_lang=en --target_lang=ar --source_tokenizer_class=SentencePieceTokenizer --target_tokenizer_class=SentencePieceTokenizer;

python dotless_arabic/experiments/translation/run_experiment.py --gpu_devices=0 --source_lang=en --target_lang=ar --source_tokenizer_class=WordTokenizer --target_tokenizer_class=WordTokenizer;
python dotless_arabic/experiments/translation/run_experiment.py --gpu_devices=0 --source_lang=ar --target_lang=en --source_tokenizer_class=WordTokenizer --target_tokenizer_class=WordTokenizer;
```