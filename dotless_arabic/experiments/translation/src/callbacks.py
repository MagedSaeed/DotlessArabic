import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from dotless_arabic.experiments.translation.src.utils import get_blue_score


class BleuDuringTrainingCallback(Callback):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        source_tokenizer,
        target_tokenizer,
        max_sequence_length,
        source_language_code,
        target_language_code,
        *args,
        show_translation_for=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.seqlen = max_sequence_length
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code
        self.show_translation_for = show_translation_for

    def on_train_epoch_end(self, trainer, pl_module):
        # def on_train_end(self, trainer, pl_module):
        train_bleu_score = get_blue_score(
            use_tqdm=False,
            model=pl_module,
            decode_with_beam_search=False,
            max_sequence_length=self.seqlen,
            save_predictions_and_targets=False,
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            show_translations_for=self.show_translation_for,
            source_sentences=self.train_dataset[self.source_language_code],
            target_sentences=self.train_dataset[self.target_language_code],
        )
        val_bleu_score = get_blue_score(
            use_tqdm=False,
            model=pl_module,
            decode_with_beam_search=False,
            max_sequence_length=self.seqlen,
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            show_translations_for=self.show_translation_for,
            source_sentences=self.val_dataset[self.source_language_code],
            target_sentences=self.val_dataset[self.target_language_code],
        )
        pl_module.log(
            "train_bleu_score",
            train_bleu_score,
            prog_bar=True,
            on_step=False,
        )
        pl_module.log(
            "val_bleu_score",
            val_bleu_score,
            prog_bar=True,
            on_step=False,
        )
