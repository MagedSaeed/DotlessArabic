from functools import lru_cache
import torch
import numpy as np
from torchmetrics.functional import word_error_rate, char_error_rate

from dotless_arabic.tokenizers import CharacterTokenizer
from dotless_arabic.experiments.dots_retrieval.src.processing import prepare
from dotless_arabic.experiments.dots_retrieval.src.models import LitBiLSTMModel


MODEL_CHECKPOINT_PATH = (
    "dotless_arabic/experiments/dots_retrieval/bin/dots_retrieval_model.ckpt"
)
SOURCE_TOKENIZER_MODEL_PATH = (
    "dotless_arabic/experiments/dots_retrieval/bin/source_tokenizer.model"
)
TARGET_TOKENIZER_MODEL_PATH = (
    "dotless_arabic/experiments/dots_retrieval/bin/target_tokenizer.model"
)


def create_features_from_text_list(text, tokenizer, seq_len=500):
    encoded_text = [tokenizer.token_to_id(c) for c in text]
    encoded_text = encoded_text[:seq_len]
    return np.array(encoded_text)


@lru_cache()
def get_source_tokenizer():
    # load source tokenizer
    source_tokenizer = CharacterTokenizer()
    source_tokenizer.load_model(SOURCE_TOKENIZER_MODEL_PATH)
    source_tokenizer.vocab_size = len(source_tokenizer.vocab)
    return source_tokenizer


@lru_cache()
def get_target_tokenizer():
    # load target tokenizer
    target_tokenizer = CharacterTokenizer()
    target_tokenizer.load_model(TARGET_TOKENIZER_MODEL_PATH)
    target_tokenizer.vocab_size = len(target_tokenizer.vocab)
    return target_tokenizer


@lru_cache
def load_model(vocab_size, output_size):
    model = LitBiLSTMModel(
        vocab_size=vocab_size,
        output_size=output_size,
    ).load_from_checkpoint(checkpoint_path=MODEL_CHECKPOINT_PATH)
    # call the model and predict output
    model.eval()
    return model


def add_dots_to_undotted_text(undotted_text):
    if not undotted_text:
        return undotted_text
    # load tokenizers
    source_tokenizer = get_source_tokenizer()
    target_tokenizer = get_target_tokenizer()
    # prepare the text and encode it
    undotted_text = prepare(undotted_text)
    encoded_input = create_features_from_text_list(
        text=undotted_text,
        tokenizer=source_tokenizer,
    )
    # define the model and load it from checkpoint
    model = load_model(
        vocab_size=source_tokenizer.vocab_size,
        output_size=target_tokenizer.vocab_size,
    )
    with torch.no_grad():
        predictions = model(
            torch.tensor(
                [encoded_input],
                device=model.device,
            ).to(dtype=torch.long)
        )
    # convert to tokens:
    # predictions = predictions.cpu().tolist()
    predictions = torch.argmax(predictions, dim=-1)
    predictions = predictions.squeeze()
    predictions = predictions.tolist()
    predicted_text = "".join(target_tokenizer.decode(predictions))
    predicted_text = (
        predicted_text.replace("<PAD>", "")[: len(undotted_text)]
        .strip()
        .replace("▁", " ")
    )
    return predicted_text.strip()


def calculate_text_metrics(predictions, labels, target_tokenizer, print_text=False):
    # drop pads, those pads are not necessary pad tokens!!
    # last_pad = predictions[-1]
    # for i,pad in reversed(list(enumerate(predictions))):
    #   if pad == last_pad:
    #     predictions.pop(i)
    #   else:
    #     break

    true_text = "".join(target_tokenizer.decode(labels))
    true_text = true_text.replace("<PAD>", "").strip().replace("▁", " ")
    # true_text = re.sub(' +',' ',true_text)

    predicted_text = "".join(target_tokenizer.decode(predictions))
    predicted_text = (
        predicted_text.replace("<PAD>", "")[: len(true_text)].strip().replace("▁", " ")
    )
    # predicted_text = re.sub(' +',' ',predicted_text)

    if print_text:
        print(predicted_text)
        print(true_text)

    wer = word_error_rate(preds=predicted_text, target=true_text)
    cer = char_error_rate(preds=predicted_text, target=true_text)

    return wer, cer
