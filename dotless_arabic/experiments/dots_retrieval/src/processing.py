import re
from pyarabic import araby
from dotless_arabic import constants
from dotless_arabic.processing import process


def rm_extra_whitespaces(text):
    return re.sub(r" +", " ", text)


# def clean_pipeline(text):
#     # text = rm_link(text)
#     # text = rm_html(text)
#     # text = space_bt_punct(text)
#     # text = rm_punct2(text)
#     # text = rm_number(text)
#     # text = rm_nonascii(text)
#     # text = rm_emoji(text)
#     """arabic specific"""
#     text = araby.strip_diacritics(text)
#     text = araby.strip_tatweel(text)
#     text = araby.normalize_alef(text)
#     text = araby.normalize_hamza(text)
#     text = araby.normalize_ligature(text)
#     # remove any non arabic character
#     text = "".join([c for c in text if c in constants.ARABIC_LETTERS or c.isspace()])
#     text = rm_extra_whitespaces(text)
#     # text = spell_correction(text)
#     return process(text).strip()


def prepare(text):
    return text.replace(" ", "▁")
