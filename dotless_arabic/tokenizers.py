import re
import tkseem as tk
from functools import lru_cache
from farasa.segmenter import FarasaSegmenter

from dotless_arabic.processing import undot


class CharacterTokenizer(tk.CharacterTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text, undot_text=False):
        if undot_text:
            text = undot(text)
        return list(
            map(
                lambda c: "<##>" if c.isspace() else c,
                list(text),
            )
        )

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("<##>", " ")
        return detokenized


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


class WordTokenizer(tk.WordTokenizer):
    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text, undot_text=False):
        if undot_text:
            return undot(text).split()
        return text.split()


TOKENIZERS_MAP = {
    tokenizer_class.__name__: tokenizer_class
    for tokenizer_class in [
        WordTokenizer,
        FarasaMorphologicalTokenizer,
        DisjointLetterTokenizer,
        CharacterTokenizer,
    ]
}
