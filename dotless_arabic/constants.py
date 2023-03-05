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


UNICODE_LETTERS_MAPPING = {
    "ﻲ": "ي",
    "ﻟ": "ل",
    "ﻓ": "و",
    "ﺎ": "ا",
    "ﺍ": "ا",
    "ﻭ": "و",
    "ﺮ": "ر",
    "ﻣ": "م",
    "ک": "ك",
    "ﻴ": "ي",
    "ﻮ": "و",
    "ﻳ": "ي",
    "ﺪ": "د",
    "ﻻ": "لا",
    "ﻠ": "ل",
    "ﺑ": "ب",
    "ﺭ": "ر",
    "ﻤ": "م",
    "ﻬ": "ه",
    "ﻨ": "ن",
    "ﻋ": "ء",
    "ﻰ": "ى",
    "ﺗ": "ت",
    "ﻧ": "ن",
    "ﻢ": "م",
    "ﻪ": "ه",
    "ﺃ": "أ",
    "ﺘ": "ت",
    "ﻌ": "ع",
    "ﻦ": "ن",
    "ﺒ": "ب",
    "ﺣ": "ح",
    "ﺇ": "إ",
    "ﺴ": "س",
    "گ": "ك",
    "ﻗ": "و",
    "ﺤ": "ح",
    "ﺳ": "س",
    "ﺩ": "د",
    "ٱ": "ا",
    "ﺖ": "ت",
    "ﻔ": "ف",
    "ﻷ": "لأ",
    "ﻘ": "ق",
    "ﺟ": "ج",
    "ﻞ": "ل",
    "ﻛ": "ك",
    "ﻫ": "ه",
    "ﻜ": "ك",
    "ﻥ": "ن",
    "ﺠ": "ج",
    "ﺸ": "ش",
    "ﺼ": "ص",
    "ﻚ": "ك",
    "ﻝ": "ل",
    "ﺀ": "ء",
    "ﻡ": "م",
    "|": "ا",
    "ﺔ": "ة",
    "ﺛ": "ث",
    "ﺷ": "ش",
    "ﺰ": "ز",
    "ﺬ": "ذ",
    "ﺧ": "خ",
    "ﺐ": "ب",
    "ﺻ": "ص",
    "ﻱ": "ي",
    "ﻩ": "ه",
    "ﻄ": "ط",
    "ﻼ": "لا",
    "ﺫ": "ذ",
    "ﺕ": "ت",
    "ﺏ": "ب",
    "ی": "ى",
    "ﻯ": "ى",
    "ﻏ": "غ",
    "ﺨ": "خ",
    "ﻊ": "ع",
    "ﺓ": "ة",
    "ﻀ": "ض",
    "ﺂ": "آ",
    "ﻙ": "ك",
    "ﻐ": "غ",
    "ﻕ": "ق",
    "ﻃ": "ط",
    "ﻒ": "ف",
    "ﺯ": "ز",
    "ﻑ": "ف",
    "ھ": "ه",
    "ﻖ": "ق",
    "ٲ": "ا",
    "ﺿ": "ض",
    "ﺄ": "أ",
    "ﺜ": "ث",
    "ﺡ": "ح",
    "ﻉ": "ع",
    "ﺢ": "ح",
    "ﺱ": "س",
    "ﺋ": "ئ",
    "ﻈ": "ظ",
    "ﭐ": "ا",
    "ﻵ": "لآ",
    "ﻹ": "لإ",
    "ﺲ": "س",
    "ﺌ": "ئ",
    "ﺝ": "ج",
    "ﻂ": "ط",
    "ﻇ": "ظ",
    "ﺶ": "ش",
    "ﺽ": "ض",
    "ﭑ": "ا",
    "ﺾ": "ض",
    "ﺞ": "ج",
    "ﺅ": "ؤ",
    "ﺚ": "ث",
    "ﺺ": "ص",
    "ﻸ": "ﻷ",
    "ﺈ": "إ",
    "ﺆ": "ؤ",
    "ﻶ": "لآ",
    "ﯾ": "ي",
    "ﻎ": "غ",
    "ﺦ": "خ",
    "ﻍ": "غ",
    "ﺥ": "خ",
    "ﺵ": "ش",
    "ﻺ": "لإ",
    "ٳ": "ا",
    "ﻅ": "ظ",
    "ﺙ": "ث",
    "ﺉ": "ئ",
}
