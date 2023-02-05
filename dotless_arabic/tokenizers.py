from collections import defaultdict
import re
import tkseem as tk
from functools import lru_cache
from farasa.segmenter import FarasaSegmenter

from dotless_arabic.processing import undot


class CharacterTokenizer(tk.CharacterTokenizer):
    def __init__(
        self,
        unk_token="<UNK>",
        pad_token="<PAD>",
        vocab_size=10000,
        special_tokens=[],
    ):
        special_tokens.append("<##>")  # this is a placeholder for space character
        super().__init__(unk_token, pad_token, vocab_size, special_tokens)

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


class DisjointLetterTokenizer(tk.DisjointLetterTokenizer):
    def __init__(
        self,
        unk_token="<UNK>",
        pad_token="<PAD>",
        vocab_size=10000,
        special_tokens=[],
    ):
        special_tokens.append("<##>")
        super().__init__(unk_token, pad_token, vocab_size, special_tokens)

    @classmethod
    @lru_cache(maxsize=10_000)
    def split_text(cls, text, undot_text=False):
        rx = re.compile(r"([اأإآءؤﻵﻹﻷدذرزو])")
        text = rx.sub(r"\1 ", text.replace(" ", " <##> "))
        if undot_text:
            return undot(text).split()
        return text.split()


class FarasaMorphologicalTokenizer(tk.FarasaMorphologicalTokenizer):
    def __init__(
        self,
        unk_token="<UNK>",
        pad_token="<PAD>",
        vocab_size=10000,
        special_tokens=[],
        interactive_segmentation=True,
    ):
        special_tokens.append("<##>")
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            interactive_segmentation=interactive_segmentation,
        )

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
