"""Shared caption vocabulary for UEC-Ramen555.

Both the training dataset (`dataset/ramen_dataset.py`) and the inference example
builder (`examples_Ramen/make_examples.py`) import from here, so that the captions
the model is trained on and the captions it is prompted with at inference time are
worded identically. Editing a phrase here changes both sides at once.
"""
import csv
import os

# label_list.txt: index -> (name, RGB in Ramen_color_masks)
CLASS_ID_TO_NAME = {
    0: "background",
    1: "bowl",
    2: "soup",
    3: "spoon",
    4: "chopsticks",
    5: "cut_egg",
    6: "egg",
    7: "seaweed",
    8: "meat",
    9: "menma",
    10: "naruto",
}

CLASS_COLORS = {
    "background": (0, 0, 0),
    "bowl": (226, 128, 128),
    "soup": (64, 32, 32),
    "spoon": (255, 0, 0),
    "chopsticks": (255, 191, 0),
    "cut_egg": (127, 255, 0),
    "egg": (0, 255, 63),
    "seaweed": (0, 255, 255),
    "meat": (0, 63, 255),
    "menma": (127, 0, 255),
    "naruto": (255, 0, 191),
}

SOUP_TYPES = ["salt", "soysauce", "miso", "porkbone", "spicy"]

SOUP_TEXT = {
    "salt": "clear pale golden shio (salt) broth",
    "soysauce": "clear brown shoyu (soy sauce) broth",
    "miso": "opaque reddish-brown miso broth",
    "porkbone": "creamy white tonkotsu (pork bone) broth",
    "spicy": "deep red spicy broth with chili oil floating on top",
}

# Regional captions. Kept well under regional_max_sequence_length (50 T5 tokens).
# "soup" is generated per-sample from the soup label, so it is not listed here.
REGION_TEXT = {
    "background": "The table surface behind the ramen bowl, softly out of focus.",
    "bowl": "A ceramic ramen bowl seen from above, its glazed rim catching the light.",
    "spoon": "A Chinese soup spoon resting in the bowl.",
    "chopsticks": "A pair of wooden chopsticks laid across the bowl.",
    "cut_egg": "A halved soft-boiled egg (ajitama), the glossy orange yolk facing up.",
    "egg": "A whole soft-boiled egg with smooth glossy white surface.",
    "seaweed": "A sheet of dried nori seaweed, dark and slightly glossy, standing in the soup.",
    "meat": "A slice of chashu pork with a soft fat rim and browned white surface.",
    "menma": "Strips of menma, pale yellow-brown fermented bamboo shoots.",
    "naruto": "A slice of naruto fish cake, white with a pink spiral pattern.",
}

TOPPING_NAME = {
    "cut_egg": "a halved boiled egg",
    "egg": "a boiled egg",
    "seaweed": "nori seaweed",
    "meat": "sliced chashu pork",
    "menma": "bamboo shoots",
    "naruto": "fish cake",
    "spoon": "a soup spoon",
    "chopsticks": "chopsticks",
}

TOPPING_ORDER = ["cut_egg", "egg", "seaweed", "meat", "menma", "naruto"]
UTENSIL_ORDER = ["spoon", "chopsticks"]


def load_soup_labels(dataset_root):
    """name (e.g. '0001.png') -> soup type string."""
    path = os.path.join(dataset_root, "soup_labels.txt")
    rows = list(csv.reader(open(path), delimiter="\t"))
    hdr = rows[0][1:]
    out = {}
    for r in rows[1:]:
        flags = {h: int(v) for h, v in zip(hdr, r[1:])}
        out[r[0]] = next((k for k, v in flags.items() if v), "salt")
    return out


def regional_caption(class_name, soup_type):
    if class_name == "soup":
        return f"The {SOUP_TEXT[soup_type]}, with wheat noodles just under the surface."
    return REGION_TEXT[class_name]


def global_caption(class_names, soup_type):
    toppings = [TOPPING_NAME[c] for c in TOPPING_ORDER if c in class_names]
    utensils = [TOPPING_NAME[c] for c in UTENSIL_ORDER if c in class_names]

    cap = (
        f"A top-down photo of a bowl of Japanese ramen with {SOUP_TEXT[soup_type]}. "
        f"Wheat noodles are submerged in the broth"
    )
    cap += f", topped with {', '.join(toppings)}. " if toppings else ". "
    if utensils:
        verb = "rest" if len(utensils) > 1 else "rests"
        cap += f"{' and '.join(utensils).capitalize()} {verb} in the bowl. "
    cap += "The bowl sits on a table, shot in warm natural light, appetizing food photography."
    return cap
