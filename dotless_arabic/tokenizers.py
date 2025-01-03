import io
import re
import tempfile
import warnings
import tkseem as tk
import sentencepiece as spm
from functools import lru_cache
from collections import defaultdict
from farasa.segmenter import FarasaSegmenter

from dotless_arabic.processing import undot


def tokenize_words_vocab(vocab, tokenizer):
    tokenized_vocab = defaultdict(int)
    for vocab, frequency in vocab.items():
        if vocab in tokenizer.special_tokens + [
            tokenizer.pad_token,
            tokenizer.unk_token,
        ]:
            tokenized_vocab[vocab] = frequency
            continue
        if isinstance(tokenizer, FarasaMorphologicalTokenizer):
            subwords = tokenizer.split_text(vocab, segmenter=tokenizer.segmenter)
        else:
            subwords = tokenizer.split_text(vocab)
        for subword in subwords:
            tokenized_vocab[subword] += frequency
    return dict(tokenized_vocab)


class WordTokenizer(tk.WordTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text, undot_text=False):
        if undot_text:
            return undot(text).split()
        return text.split()

    def train(self, text=None, file_path=None):
        """Train using words' frequency

        Args:
            file_path (str): file to train
        """

        print("Training WordTokenizer ...")
        self.vocab = self._truncate_dict(
            self._get_tokens_frequency(text=text, file_path=file_path)
        )
        self.vocab_size = len(self.vocab)

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = " ".join(tokens)
        return detokenized


class CharacterTokenizer(tk.CharacterTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text, undot_text=False):
        if undot_text:
            text = undot(text)
        splitted_text = []
        for character in list(text):
            if character.isspace():
                splitted_text.append("<##>")
            else:
                splitted_text.append(character)
        return splitted_text

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("<##>", " ")
        return detokenized

    def train(self, text=None, file_path=None):
        print("Training CharacterTokenizer ...")

        assert (
            file_path is not None or text is not None
        ), "either file_path or text should be provided."

        if not text:
            text = open(file_path, "r").read()

        tokens_frequency = defaultdict(int)
        for word in WordTokenizer.split_text(text):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab = tokenize_words_vocab(vocab=self.vocab, tokenizer=self)
        self.vocab_size = len(self.vocab)


class DisjointLetterTokenizer(tk.DisjointLetterTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text, undot_text=False):
        rx = re.compile(r"([اأإآءؤﻵﻹﻷدذرزو])")
        text = rx.sub(r"\1 ", text.replace(" ", " <##> "))
        if undot_text:
            return undot(text).split()
        return text.split()

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("<##>", " ")
        return detokenized

    def train(self, text=None, file_path=None):
        """Train data using disjoint letters
        Args:
            file_path (str): file to train
        """
        print("Training DisjointLetterTokenizer ...")

        if not text:
            text = open(file_path, "r").read()

        tokens_frequency = defaultdict(int)

        for word in WordTokenizer.split_text(text):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab = tokenize_words_vocab(vocab=self.vocab, tokenizer=self)
        self.vocab_size = len(self.vocab)


DEFAULT_SEGMENTER = FarasaSegmenter(interactive=True)


class FarasaMorphologicalTokenizer(tk.FarasaMorphologicalTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(
        cls,
        text,
        segmenter=None,
        undot_text=False,
    ):
        if segmenter is None:
            segmenter = DEFAULT_SEGMENTER
        assert isinstance(
            segmenter,
            FarasaSegmenter,
        ), "segmenter should be an instance of FarasaSegmenter"

        text = segmenter.segment(text).replace(" ", " <##> ").replace("+", " ")
        if undot_text:
            return undot(text).split()
        return text.split()

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("<##>", " ")
        return detokenized

    def train(self, text=None, file_path=None):
        """Train data using farasa
        Args:
            file_path (str): file to train
        """

        print("Training FarasaMorphologicalTokenizer...")

        if not text:
            with open(file_path, "r") as f:
                text = f.read()

        tokens_frequency = defaultdict(int)
        for word in WordTokenizer.split_text(text):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab = tokenize_words_vocab(vocab=self.vocab, tokenizer=self)
        self.vocab_size = len(self.vocab)

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("<##>", " ")
        return detokenized


class SentencePieceTokenizer(tk.SentencePieceTokenizer):
    """Sentencepiece based tokenization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self, unk_token="<UNK>", pad_token="<PAD>", vocab_size=10000, special_tokens=[],
        special_tokens = kwargs.get("special_tokens")
        self.bos_token = "<s>"  # sentencepiece defaults
        self.bos_id = -1
        self.eos_token = "</s>"  # sentencepiece defaults
        self.eos_id = -1
        if special_tokens:
            if "<bos>" in special_tokens:
                self.bos_token = "<bos>"
                self.bos_id = 2
            if "<eos>" in special_tokens:
                self.eos_token = "<eos>"
                self.eos_id = 3

    def train(
        self,
        text=None,
        file_path=None,
        model_type="bpe",
        **kwargs,
    ):
        """Train using sentence piece

        Args:
            file_path (str): file to train
            model_type (str, optional): train using sp. Defaults to "bpe".
        """
        assert (
            text is not None or file_path is not None
        ), "file_path or text should be provided"

        if file_path:
            text = open(file_path, "r").read().splitlines()

        text_file = tempfile.NamedTemporaryFile()

        with open(text_file.name, "w") as file:
            file.write(text)

        print("Training SentencePiece ...")
        self.model = io.BytesIO()

        """
        This was written to catch a special case for sentencepiece.
        If higher vocab size than the total dataset vocabulary is given, it gives an error. Hence this custom training procedure.
        """

        def _train(vocab_size):
            # https://github.com/google/sentencepiece/blob/master/doc/options.md
            spm.SentencePieceTrainer.train(
                input=text_file.name,
                model_writer=self.model,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=kwargs.get("character_coverage", 1.0),
                unk_id=0,
                pad_id=1,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                add_dummy_prefix=False,
                pad_piece=self.pad_token,
                unk_piece=self.unk_token,
                bos_piece=self.bos_token,
                eos_piece=self.eos_token,
                normalization_rule_name="identity",
                user_defined_symbols=self.special_tokens,
                minloglevel=1,  # to suppress train logs, https://github.com/speechbrain/speechbrain/pull/206#issuecomment-669260984
            )

        try:
            _train(vocab_size=self.vocab_size)
        except RuntimeError as e:
            error_message = str(e)
            if "Please set it to a value" in error_message:
                vocab_size = int(
                    "".join(c for c in error_message.split("<=")[-1] if c.isdigit())
                )
                warnings.warn(
                    f"the given vocab_size ({self.vocab_size}) is more than the total dataset vocabulary. This cause an error for SentencePieceTokenizer. Reducing it to max vocab_size of dataset {vocab_size} and retraining the model."
                )
                self.vocab_size = vocab_size
                _train(vocab_size=self.vocab_size)
            else:
                raise e

        model_file = tempfile.NamedTemporaryFile()
        self.save_model(model_file.name)
        self.sp = spm.SentencePieceProcessor(model_file=model_file.name)
        self.vocab_size = self.sp.vocab_size()
        self.vocab = {
            self.sp.id_to_piece(id): id for id in range(self.sp.get_piece_size())
        }

    def split_text(self, text):
        """
        For this tokenizer, split text is the same as tokenize
        """
        warnings.warn(
            "sentencepiece tokenizer cannot split text unless with PBE mode. It needs to be trained first!"
        )
        return self.tokenize(text)

    def tokenize_from_splits(self, text):
        """
        For this tokenizer, split text is the same as tokenize
        """
        warnings.warn(
            "sentencepiece tokenizer cannot split text unless with PBE mode. It needs to be trained first!"
        )
        return self.tokenize(text)


TOKENIZERS_MAP = {
    tokenizer_class.__name__: tokenizer_class
    for tokenizer_class in [
        WordTokenizer,
        FarasaMorphologicalTokenizer,
        SentencePieceTokenizer,
        DisjointLetterTokenizer,
        CharacterTokenizer,
    ]
}


def get_tokenizer(
    train_dataset,
    tokenizer_class,
    undot_text=False,
    vocab_size=10_000,
):
    if undot_text:
        text = "\n".join(undot(item) for item in train_dataset if item.strip())
    else:
        text = "\n".join(item for item in train_dataset if item.strip())
    tokenizer = tokenizer_class(
        vocab_size=vocab_size,
        special_tokens=["#", "<##>"],
    )
    tokenizer.train(text=text)
    return tokenizer
