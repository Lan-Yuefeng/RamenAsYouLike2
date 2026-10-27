"""Class vocabulary for UECFOODPIXCOMPLETE (UEC Food Pix Complete, 102 food classes).

Generated from `<dataset_root>/data/category.txt` (id -> name, ids 1..102) with id 0
added as the background class.

Unlike ADE20K -- where id 0 means "unannotated" and is skipped -- UECFOODPIXCOMPLETE
annotates every pixel, and 0 is the real non-food region (plate, tray, table). It is
kept as a controllable class here, the same choice `dataset/ramen_captions.py` makes
for the ramen table background.

Captions follow the ADE20K/COCO-Stuff convention used by upstream Seg2Any: the
regional caption is the bare class name and there is no global caption, so
`regional_max_sequence_length` can stay small (10 in config/seg2any_uecfoodpix.yaml).
"""

CLASS_ID_TO_NAME = {
    0: "background",
    1: "rice",
    2: "eels on rice",
    3: "pilaf",
    4: "chicken-'n'-egg on rice",
    5: "pork cutlet on rice",
    6: "beef curry",
    7: "sushi",
    8: "chicken rice",
    9: "fried rice",
    10: "tempura bowl",
    11: "bibimbap",
    12: "toast",
    13: "croissant",
    14: "roll bread",
    15: "raisin bread",
    16: "chip butty",
    17: "hamburger",
    18: "pizza",
    19: "sandwiches",
    20: "udon noodle",
    21: "tempura udon",
    22: "soba noodle",
    23: "ramen noodle",
    24: "beef noodle",
    25: "tensin noodle",
    26: "fried noodle",
    27: "spaghetti",
    28: "Japanese-style pancake",
    29: "takoyaki",
    30: "gratin",
    31: "sauteed vegetables",
    32: "croquette",
    33: "grilled eggplant",
    34: "sauteed spinach",
    35: "vegetable tempura",
    36: "miso soup",
    37: "potage",
    38: "sausage",
    39: "oden",
    40: "omelet",
    41: "ganmodoki",
    42: "jiaozi",
    43: "stew",
    44: "teriyaki grilled fish",
    45: "fried fish",
    46: "grilled salmon",
    47: "salmon meuniere ",
    48: "sashimi",
    49: "grilled pacific saury ",
    50: "sukiyaki",
    51: "sweet and sour pork",
    52: "lightly roasted fish",
    53: "steamed egg hotchpotch",
    54: "tempura",
    55: "fried chicken",
    56: "sirloin cutlet ",
    57: "nanbanzuke",
    58: "boiled fish",
    59: "seasoned beef with potatoes",
    60: "hambarg steak",
    61: "beef steak",
    62: "dried fish",
    63: "ginger pork saute",
    64: "spicy chili-flavored tofu",
    65: "yakitori",
    66: "cabbage roll",
    67: "rolled omelet",
    68: "egg sunny-side up",
    69: "fermented soybeans",
    70: "cold tofu",
    71: "egg roll",
    72: "chilled noodle",
    73: "stir-fried beef and peppers",
    74: "simmered pork",
    75: "boiled chicken and vegetables",
    76: "sashimi bowl",
    77: "sushi bowl",
    78: "fish-shaped pancake with bean jam",
    79: "shrimp with chill source",
    80: "roast chicken",
    81: "steamed meat dumpling",
    82: "omelet with fried rice",
    83: "cutlet curry",
    84: "spaghetti meat sauce",
    85: "fried shrimp",
    86: "potato salad",
    87: "green salad",
    88: "macaroni salad",
    89: "Japanese tofu and vegetable chowder",
    90: "pork miso soup",
    91: "chinese soup",
    92: "beef bowl",
    93: "kinpira-style sauteed burdock",
    94: "rice ball",
    95: "pizza toast",
    96: "dipping noodles",
    97: "hot dog",
    98: "french fries",
    99: "mixed rice",
    100: "goya chanpuru",
    101: "others",
    102: "beverage",
}

NUM_CLASSES = len(CLASS_ID_TO_NAME)  # 103 including background


# Caption overrides. `category.txt` id 101 is the catch-all bucket for food outside the
# 100 named categories, so its literal name "others" is a semantically empty prompt --
# and it is the most frequent food class in the set (2334/9000 train images, 3.1% of all
# pixels). "food" keeps the region, its contour and its attention routing intact while
# giving T5 a word it has a real prior for, and doubles as the inference-time fallback
# for dishes the 100-class vocabulary does not cover.
#
# The regions genuinely contain food, so folding them into "background" would teach the
# anchor class (57.8% of pixels) that background can look like a meal. Dropping them
# instead would leave those image tokens attending to no text at all, because
# global_caption is None here and encode_all_prompt then sets the global text length to 0.
#
# To go back to the raw label, empty this dict.
REGIONAL_NAME_OVERRIDE = {
    "others": "food",
}


def regional_caption(class_name):
    """Bare class name, as in ADE20KDataset/COCOStuffDataset."""
    return REGIONAL_NAME_OVERRIDE.get(class_name, class_name)
