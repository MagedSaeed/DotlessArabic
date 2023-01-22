import os

import pyarabic.araby as araby

RANDOM_SEED = 42

CPU_COUNT = os.cpu_count()

ARABIC_LETTERS = "".join(araby.LETTERS) + "".join(araby.LIGUATURES)


ALEF_SHAPE = "ا"
BAA_SHAPE = "\u066E"
JEEM_SHAPE = "ح"
DAL_SHAPE = "د"
RAA_SHAPE = "ر"
SEEN_SHAPE = "س"
SAAD_SHAPE = "ص"
TAA_SHAPE = "ط"
AIN_SHAPE = "ع"
FAA_SHAPE = "\u06A1"
QAF_SHAPE = "\u066F"
KAF_SHAPE = "ك"
LAM_SHAPE = "ل"
MEEM_SHAPE = "م"
# NOON_SHAPE =u'\u06BA' same as baa_shape?
NOON_SHAPE = "\u06BA"
HAA_SHAPE = "ه"
WAW_SHAPE = "و"
YAA_SHAPE = "ى"
HAMZA_SHAPE = "ء"

LETTERS_MAPPING = {
    "ا": ALEF_SHAPE,
    "أ": ALEF_SHAPE,
    "إ": ALEF_SHAPE,
    "ب": BAA_SHAPE,
    "ت": BAA_SHAPE,
    "ث": BAA_SHAPE,
    "ج": JEEM_SHAPE,
    "ح": JEEM_SHAPE,
    "خ": JEEM_SHAPE,
    "د": DAL_SHAPE,
    "ذ": DAL_SHAPE,
    "ر": RAA_SHAPE,
    "ز": RAA_SHAPE,
    "س": SEEN_SHAPE,
    "ش": SEEN_SHAPE,
    "ص": SAAD_SHAPE,
    "ض": SAAD_SHAPE,
    "ط": TAA_SHAPE,
    "ظ": TAA_SHAPE,
    "ع": AIN_SHAPE,
    "غ": AIN_SHAPE,
    "ف": FAA_SHAPE,
    "ق": FAA_SHAPE,  # same as faa?
    "ك": KAF_SHAPE,
    "ل": LAM_SHAPE,
    "م": MEEM_SHAPE,
    "ن": BAA_SHAPE,
    "ه": HAA_SHAPE,
    "ة": HAA_SHAPE,
    "و": WAW_SHAPE,
    "ؤ": WAW_SHAPE,
    "ي": YAA_SHAPE,
    "ى": YAA_SHAPE,
    "ئ": YAA_SHAPE,
    "ء": HAMZA_SHAPE,
}


NEW_LINE = "\n"
