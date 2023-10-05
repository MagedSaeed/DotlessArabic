import re
import string

from pyarabic import araby

from dotless_arabic.processing import process
from dotless_arabic import constants


def process_source(text):
    clean_text = text.replace("quot", "")
    clean_text = clean_text.replace("amp", "")
    strip_chars = string.punctuation
    clean_text = "".join(c for c in text if c not in strip_chars)
    clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    return clean_text.lower()


def process_target(text):
    # return process(text)
    # strip_chars = string.punctuation
    # strip_chars = strip_chars.replace("<", "").replace(">", "")
    # clean_text = "".join(c for c in text if c not in strip_chars)
    # clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    # return clean_text
    strip_chars = string.punctuation
    text = "".join(c for c in text if c not in strip_chars)
    text = text.replace("quot", "")
    # replace less known arabic unicode characters with their mapping from the well known arabic characters
    text = text.translate(str.maketrans(constants.UNICODE_LETTERS_MAPPING))

    # add spaces between punctuations, if there is not
    text = re.sub(
        r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>«»`؛=+\-\*\&\^\%\$\#\@\!:|…123456789a-zA-Z;؟–−])""",
        r" \1 ",
        text,
    )
    text = re.sub("\s{2,}", " ", text).strip()  # remove multiple spaces
    """
  interestingly, there is a difference betwen re.sub('\s+',' ',s) and re.sub('\s{2,}',' ',s)
  the first one remove newlines while the second does not.
  """
    """ remove specific characters found in the un dataset """
    text = text.replace("\xa0", "")
    text = text.replace("\x85", "")
    text = text.replace("\x96", "")
    text = text.replace("\u200a", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u3000", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\u2002", " ")
    text = text.replace("\u2003", " ")

    # replace one char ﻻ and its variants
    text = text.replace("ﻷ", "لا")
    text = text.replace("ﻹ", "لا")
    text = text.replace("ﻵ", "لا")
    text = text.replace("ﻻ", "لا")
    """these are included already in the first step"""
    # text = text.replace('\u200f','').strip() # remove a wired char,
    text = araby.strip_tashkeel(text)  # remove tashkeel
    text = araby.strip_tatweel(text)  # remove tatweel
    text = araby.strip_diacritics(text)  # remove diacritics and special symbols
    """"""
    # text = text.replace('أ','ا').replace('إ','ا').replace('ؤ','و').replace('ئ','ى').replace('ء','').replace('آ','ا') # remove hamza symbol completely
    # issues with hamza complete removal:
    # - words maybe broken إنشاءات
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ؤ", "و")
        .replace("ئ", "ى")
        .replace("آ", "ا")
    )  # remove hamza symbol attached to letters
    return text.strip()
