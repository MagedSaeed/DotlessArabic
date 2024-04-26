import math
import os
import re
from pathlib import Path
import shutil
import numpy as np

import torch

from tqdm.auto import tqdm

from sacrebleu.metrics import BLEU

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    LearningRateMonitor,
    StochasticWeightAveraging,
)

from dotless_arabic.experiments.translation.src.processing import (
    process_en,
    process_ar,
)
from dotless_arabic.processing.processing import undot

from dotless_arabic.experiments.translation.src import constants
from dotless_arabic.experiments.dots_retrieval.src.utils import (
    add_dots_to_undotted_text,
)
from sacremoses import MosesDetokenizer, MosesTokenizer
from dotless_arabic.tokenizers import WordTokenizer, SentencePieceTokenizer


def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)


def train_translator(
    translator,
    wandb_logger,
    train_dataloader,
    val_dataloader,
    gpu_devices,
    source_lang,
    target_lang,
    source_tokenizer_class,
    target_tokenizer_class,
    callbacks=None,
    one_run=False,
    text_type="dotted",
    validate_and_fit=True,
    use_rich_progressbar=True,
    max_epochs=constants.MAX_EPOCHS,
):
    # remove any previous checkpoints
    checkpoints_dir = f"NMT/{source_lang}_to_{target_lang}/{source_tokenizer_class.__name__}_to_{target_tokenizer_class.__name__}/{text_type}/checkpoints"
    checkpoints_path = Path(checkpoints_dir)
    if callbacks is None:
        callbacks = []
    if validate_and_fit:
        shutil.rmtree(checkpoints_path, ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        mode="min",
        # save_top_k=10,
        save_top_k=5,
        verbose=False,
        save_last=True,
        monitor="val_loss",
        dirpath=checkpoints_dir,
        save_weights_only=False,
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        filename="{epoch}-{val_loss:.3f}-{step}",
    )
    callbacks.append(checkpoint_callback)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        # min_delta=0.025,
        # min_delta=0,
        min_delta=0.0001,
        patience=3,  # 3 epochs
        # patience=3,
        check_finite=True,
    )
    callbacks.append(early_stopping_callback)
    if use_rich_progressbar:
        callbacks.append(RichProgressBar())
    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )
    callbacks.append(lr_monitor)
    xavier_init(model=translator)
    trainer = Trainer(
        deterministic=True,
        callbacks=callbacks,
        # gradient_clip_val=1,
        gradient_clip_val=0,
        max_epochs=max_epochs,
        # val_check_interval=0.5,
        check_val_every_n_epoch=1,
        accelerator="auto",
        logger=wandb_logger,
        fast_dev_run=one_run,
        devices=list(map(int, gpu_devices.split(","))),
        # log_every_n_steps=max(len(train_dataloader) // 25, 1),
        log_every_n_steps=max(len(train_dataloader) // 25, 1),
    )
    if validate_and_fit:
        trainer.validate(
            model=translator,
            dataloaders=val_dataloader,
        )
        trainer.fit(
            translator,
            train_dataloader,
            val_dataloader,
        )
    return trainer


def get_oovs_rate(
    dataloader,
    unk_token_id=0,
    pad_token_id=1,
):
    dataset = dataloader.dataset
    oovs = 0
    tokens_sum = 0
    for document in dataset.encoded_dataset:
        for token_id in document:
            if token_id == unk_token_id:
                oovs += 1
            elif token_id in [pad_token_id]:
                continue
            tokens_sum += 1
    return f"{(oovs / tokens_sum) * 100:.2f}"


def create_features_from_text_list(
    text_list,
    tokenizer,
    sequence_length=None,
    undot_text=False,
    use_tqdm=True,
    add_eos=True,
    add_bos=True,
):
    encoded = list()
    text_list = tqdm(text_list) if use_tqdm else text_list
    for doc in text_list:
        if undot_text:
            doc = undot(doc)
        encoded_doc = [
            tokenizer.token_to_id(token) for token in tokenizer.split_text(doc)
        ]
        if add_bos:
            encoded_doc.insert(0, tokenizer.token_to_id("<bos>"))
        if add_eos:
            encoded_doc += [tokenizer.token_to_id("<eos>")]
        if sequence_length is not None:
            encoded_doc = tokenizer.pad(encoded_doc, length=sequence_length)
            encoded_doc = encoded_doc[:sequence_length]
        if add_eos:
            if encoded_doc[-1] != tokenizer.token_to_id(tokenizer.pad_token):
                encoded_doc[-1] = tokenizer.token_to_id("<eos>")
        encoded.append(np.array(encoded_doc))
    return np.array(encoded)


def get_source_tokenizer(
    train_dataset,
    source_language_code,
    undot_text=False,
    tokenizer_class=WordTokenizer,
):
    if tokenizer_class == SentencePieceTokenizer:
        tokenizer = tokenizer_class(
            # vocab_size=35_000,
            vocab_size=4_000,
            special_tokens=["<bos>", "<eos>"],
        )
    else:
        tokenizer = tokenizer_class(
            vocab_size=10**10,
            special_tokens=["<bos>", "<eos>"],
        )
    if undot_text:
        tokenizer.train(
            text="\n".join(tqdm(map(undot, train_dataset[source_language_code])))
        )
    else:
        tokenizer.train(text="\n".join(tqdm(train_dataset[source_language_code])))
    if tokenizer_class != SentencePieceTokenizer:
        vocab = {
            vocab: freq
            for vocab, freq in tokenizer.vocab.items()
            if freq > 1 or freq < 0
        }
        tokenizer.vocab = vocab
        tokenizer.vocab_size = len(vocab)
    return tokenizer


def get_target_tokenizer(
    train_dataset,
    target_language_code,
    undot_text=False,
    tokenizer_class=WordTokenizer,
):
    if tokenizer_class == SentencePieceTokenizer:
        tokenizer = tokenizer_class(
            # vocab_size=10_000,
            vocab_size=4_000,
            special_tokens=["<bos>", "<eos>"],
        )
    else:
        tokenizer = tokenizer_class(
            vocab_size=10**10,
            special_tokens=["<bos>", "<eos>"],
        )
    if undot_text:
        tokenizer.train(
            text="\n".join(tqdm(map(undot, train_dataset[target_language_code])))
        )
    else:
        tokenizer.train(text="\n".join(tqdm(train_dataset[target_language_code])))
    # delete 1-freq words if tokenizer class is not Sentencepiece
    if tokenizer_class != SentencePieceTokenizer:
        vocab = {
            vocab: freq
            for vocab, freq in tokenizer.vocab.items()
            if freq > 1 or freq < 0
        }
        tokenizer.vocab = vocab
        tokenizer.vocab_size = len(vocab)
    return tokenizer


def get_sequence_length(
    dataset,
    use_tqdm=True,
    percentile=constants.SEQUENCE_LENGTH_PERCENTILE,
):
    """
    The dataset is expected to be tokenized.
    """
    lengths = []
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        lengths.append(len(document))
    lengths = sorted(lengths)
    lengths_count = len(lengths)
    percentile_index = (
        math.floor(lengths_count * percentile) if 0 <= percentile < 1 else -1
    )
    length = lengths[percentile_index]
    return length


def get_best_checkpoint(
    source_language_code,
    target_language_code,
    source_tokenizer_class,
    target_tokenizer_class,
    is_dotted=True,
    checkpoints_base_path="NMT",
):
    text_type = "dotted"
    if not is_dotted:
        text_type = "undotted"
    checkpoints_path = f"{checkpoints_base_path}/{source_language_code}_to_{target_language_code}/{source_tokenizer_class.__name__}_to_{target_tokenizer_class.__name__}/{text_type}/checkpoints"
    sorted_models_path = []
    for filename in os.listdir(checkpoints_path):
        if filename.startswith("epoch"):
            sorted_models_path.append(filename)
    sorted_models_path = sorted(
        sorted_models_path,
        key=lambda filename: "".join(
            c for c in filename.split("=")[2] if c.isdigit() or c == "."
        ),
    )
    # for file_name in os.listdir(checkpoints_path):
    #     if file_name.startswith("epoch"):
    #         print(f"checkpiont {file_name} found.")
    #         return f"{checkpoints_path}/{file_name}"
    best_path = f"{checkpoints_path}/{sorted_models_path[0]}"
    print("best ckpt:", best_path)
    return best_path
    # print("Could NOT find a checkpoint!!")


def get_averaged_model(
    model_instance,
    source_language_code,
    target_language_code,
    source_tokenizer_class,
    target_tokenizer_class,
    is_dotted=True,
    checkpoints_base_dir="NMT",
):
    text_type = "dotted"
    if not is_dotted:
        text_type = "undotted"
    checkpoints_dir = f"{checkpoints_base_dir}/{source_language_code}_to_{target_language_code}/{source_tokenizer_class.__name__}_to_{target_tokenizer_class.__name__}/{text_type}/checkpoints"

    checkpoints_paths = list(
        map(
            lambda item: f"{checkpoints_dir}/{item}",
            os.listdir(checkpoints_dir),
        )
    )

    # Load checkpoints and extract model weights
    loaded_checkpoints = [torch.load(path) for path in checkpoints_paths]
    model_weights = [checkpoint["state_dict"] for checkpoint in loaded_checkpoints]

    # Average weights
    avg_weights = {}
    num_checkpoints = len(checkpoints_paths)
    for key in model_weights[0].keys():
        avg_weights[key] = (
            sum([model_weights[i][key] for i in range(num_checkpoints)])
            / num_checkpoints
        )

    # Create a new model with averaged weights
    averaged_model = model_instance
    averaged_model.load_state_dict(avg_weights)
    averaged_model = averaged_model.to(constants.DEVICE)

    return averaged_model


def get_blue_score(
    model,
    source_sentences,
    target_sentences,
    source_tokenizer,
    target_tokenizer,
    max_sequence_length,
    source_language_code,
    target_language_code,
    use_tqdm=True,
    is_dotted=True,
    detokenize=True,
    predictions=None,
    show_translations_for=100,
    decode_with_beam_search=False,
    add_dots_to_predictions=False,
    save_predictions_and_targets=True,
):
    targets = [f"{sentence}" for sentence in target_sentences]
    if use_tqdm:
        source_sentences = tqdm(source_sentences)
        target_sentences = tqdm(target_sentences)
    if not predictions:
        if not decode_with_beam_search:
            predictions = [
                model.translate(
                    input_sentence=sentence,
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer,
                    max_sequence_length=max_sequence_length,
                )
                for sentence in source_sentences
            ]
        else:
            predictions = [
                model.translate_with_beam_search(
                    input_sentence=sentence,
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer,
                    max_sequence_length=max_sequence_length,
                )
                for sentence in source_sentences
            ]
    predictions = list(
        map(
            lambda text: text.replace("<eos>", "").replace("<bos>", "").strip(),
            predictions,
        )
    )

    if add_dots_to_predictions:
        predictions = list(map(add_dots_to_undotted_text, predictions))

    # def replace_dash(text):
    #     # Use a regular expression to find '-' surrounded by non-whitespace characters
    #     pattern = r"(\S)-(\S)"
    #     replaced_text = re.sub(pattern, r"\1 ##AT##-##AT## \2", text)
    #     return replaced_text

    # predictions = list(map(replace_dash, predictions))
    # targets = list(map(replace_dash, targets))

    if detokenize:
        # detokenization
        moses_detokenizer = MosesDetokenizer()
        source_sentences = list(
            map(
                lambda text: moses_detokenizer.detokenize(text.split()),
                source_sentences,
            )
        )  # detokenize sources to be conveneint when shown on console
        predictions = list(
            map(
                lambda text: moses_detokenizer.detokenize(text.split()),
                predictions,
            )
        )
        targets = list(
            map(
                lambda text: moses_detokenizer.detokenize(text.split()),
                targets,
            )
        )

    for i, (s, p, t) in enumerate(
        zip(
            source_sentences,
            predictions,
            targets,
        )
    ):
        if i >= show_translations_for:
            break
        print(f"Source: {s}")
        print(f"Prediction: {p}")
        print(f"Target: {t}")
        print("*" * 80)
    text_type = "dotted"
    if not is_dotted:
        text_type = "undotted"
    decode_method = "beam_search_decode"
    if not decode_with_beam_search:
        decode_method = "greedy_decode"
    if save_predictions_and_targets:
        predictions_file_name = f"dotless_arabic/experiments/translation/results/{source_language_code}_to_{target_language_code}/{source_tokenizer.__class__.__name__}_to_{target_tokenizer.__class__.__name__}/{text_type}_{decode_method}"
        if add_dots_to_predictions:
            predictions_file_name += "_with_dots"
        predictions_file_name += "_predictions.txt"
        with open(predictions_file_name, "w") as f:
            f.write("\n".join(pred for pred in predictions))
        with open(
            f"dotless_arabic/experiments/translation/results/{source_language_code}_to_{target_language_code}/{source_tokenizer.__class__.__name__}_to_{target_tokenizer.__class__.__name__}/{text_type}_{decode_method}_targets.txt",
            "w",
        ) as f:
            if (
                not is_dotted and add_dots_to_predictions
            ):  # catche the case of undotted text and add_dots_to_predictions is True
                f.write("\n".join(undot(target) for target in targets))

    # detokenize:

    bleu = BLEU(lowercase=True)

    bleu_score = bleu.corpus_score(predictions, [targets])
    print("sacre bleu", round(bleu_score.score, 3))
    print("sacre bleu signature:", bleu.get_signature())
    return round(bleu_score.score, 3)
