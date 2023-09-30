import string


def process_source(text):
    strip_chars = string.punctuation
    clean_text = "".join(c for c in text if c not in strip_chars)
    return clean_text.lower()


def process_target(text):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("<", "").replace(">", "")
    return "".join(c for c in text if c not in strip_chars)
