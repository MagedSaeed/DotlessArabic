import re
import tkseem as tk
from functools import lru_cache
from collections import defaultdict
from farasa.segmenter import FarasaSegmenter

from dotless_arabic.processing import undot


def tokenize_vocab(vocab, tokenizer):
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

    def train(self, file_path):
        print("Training CharacterTokenizer ...")

        text = open(file_path, "r").read()

        tokens_frequency = defaultdict(int)
        for word in WordTokenizer.split_text(text):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab = tokenize_vocab(vocab=self.vocab, tokenizer=self)
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

    def train(self, file_path):
        """Train data using disjoint letters
        Args:
            file_path (str): file to train
        """
        print("Training DisjointLetterTokenizer ...")

        text = open(file_path, "r").read()

        tokens_frequency = defaultdict(int)

        for word in WordTokenizer.split_text(text):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab = tokenize_vocab(vocab=self.vocab, tokenizer=self)
        self.vocab_size = len(self.vocab)


class FarasaMorphologicalTokenizer(tk.FarasaMorphologicalTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(
        cls,
        text,
        interactive_segmentation=True,
        segmenter=None,
        undot_text=False,
    ):
        if segmenter is None:
            segmenter = FarasaSegmenter(interactive=interactive_segmentation)
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

    def train(self, file_path):
        """Train data using farasa
        Args:
            file_path (str): file to train
        """

        print("Training FarasaMorphologicalTokenizer...")

        with open(file_path, "r") as f:
            text = f.read()

        tokens_frequency = defaultdict(int)
        for word in WordTokenizer.split_text(text):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab = tokenize_vocab(vocab=self.vocab, tokenizer=self)
        self.vocab_size = len(self.vocab)


TOKENIZERS_MAP = {
    tokenizer_class.__name__: tokenizer_class
    for tokenizer_class in [
        WordTokenizer,
        FarasaMorphologicalTokenizer,
        DisjointLetterTokenizer,
        CharacterTokenizer,
    ]
}
