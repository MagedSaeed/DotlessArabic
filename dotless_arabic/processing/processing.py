import re

from tqdm.auto import tqdm

from dotless_arabic import constants


def process(text, separators=[".", ":", "،", ",", "؛", "؟", "!"]):
    # replace less known arabic unicode characters with their mapping from the well known arabic characters
    text = text.translate(str.maketrans(constants.UNICODE_LETTERS_MAPPING))

    # add spaces between punctuations, if there is not
    text = re.sub(
        r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>«»`؛=+\-\*\&\^\%\$\#\@\!:|…123456789a-zA-Z;؟–−])""",
        r" \1 ",
        text,
    )
    # remove any non arabic character
    text = "".join(
        [c for c in text if c in constants.ARABIC_LETTERS or c.isspace()]
    )  # keep only arabic chars and spaces
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
    # text = araby.strip_tashkeel(text) # remove tashkeel
    # text = araby.strip_tatweel(text) # remove tatweel
    # text = araby.strip_diacritics(text) # remove diacritics and special symbols
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


def undot(text, process_first=False):
    """
    this function does not add a new line.
    make sure to append a newline afterwards
    """
    # TODO: Should yaa be mapped to baa shape in the beginning and the middle?
    if process_first:
        text = process(text)
    undotted_text = ""
    # for word in text.split():
    for word in re.split("[ \t]+", text):  # match every white space except new lines
        if word.endswith("ن"):
            word = word[:-1] + constants.NOON_SHAPE
        if word.endswith("ق"):
            word = word[:-1] + constants.QAF_SHAPE
        if word.startswith("ي") and len(word) > 1:
            word = constants.BAA_SHAPE + word[1:]
        word = word.translate(
            word.maketrans(
                "".join(constants.LETTERS_MAPPING.keys()),
                "".join(constants.LETTERS_MAPPING.values()),
            )
        )
        undotted_text += word
        if not word.isspace():
            undotted_text += " "
    return undotted_text.strip()


def dataset_dot_transform(dataset, use_tqdm=True):
    # split on dots
    transformed_dataset = list()
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        transformed_dataset.extend(
            paragraph.strip() for paragraph in document.split(".") if paragraph.strip()
        )
    return transformed_dataset


def dataset_newline_transform(dataset, use_tqdm=True):
    # split on newlines
    transformed_dataset = list()
    dataset = tqdm(dataset) if use_tqdm else dataset
    for document in dataset:
        paragraphs = document.splitlines()
        transformed_dataset.extend(
            paragraph.strip() for paragraph in paragraphs if paragraph.strip()
        )
    return transformed_dataset
