import re
import string

from pyarabic import araby

from dotless_arabic.processing import process


def process_en(text):
    clean_text = text.replace("quot", "")
    clean_text = clean_text.replace("amp", "")
    strip_chars = string.punctuation
    clean_text = "".join(c for c in text if c not in strip_chars)
    clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    clean_text = re.sub(r"([?.!,¿])", r" \1 ", clean_text)
    clean_text = "".join(c for c in clean_text if not c.isdigit())
    clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    # return re.sub(r"[^a-zA-Z ]+", "", text).lower()
    return clean_text.lower()


def process_ar(text):
    text = araby.strip_diacritics(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    text = araby.normalize_ligature(text)
    return process(text)
    # strip_chars = string.punctuation
    # strip_chars = strip_chars.replace("<", "").replace(">", "")
    # clean_text = "".join(c for c in text if c not in strip_chars)
    # clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    # clean_text = re.sub(r"([?.!,¿])", r" \1 ", clean_text)
    # clean_text = araby.strip_diacritics(text=clean_text)
    # clean_text = araby.strip_tatweel(text=clean_text)
    # return clean_text.lower()
