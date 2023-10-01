import re
import string


def process_source(text):
    strip_chars = string.punctuation
    clean_text = "".join(c for c in text if c not in strip_chars)
    clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    return clean_text.lower()


def process_target(text):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("<", "").replace(">", "")
    clean_text = "".join(c for c in text if c not in strip_chars)
    clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    return clean_text
