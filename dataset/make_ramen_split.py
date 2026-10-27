"""Write the UEC-Ramen555 train/val split (500 / 55), stratified by soup type.

Run once:
    .venv/bin/python -m dataset.make_ramen_split
"""
import os
import random
from collections import Counter, defaultdict

from dataset.ramen_captions import load_soup_labels

ROOT = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"
OUT = "./data/ramen"
NUM_VAL = 55
SEED = 42


def main():
    os.makedirs(OUT, exist_ok=True)
    soup = load_soup_labels(ROOT)
    names = sorted(f for f in os.listdir(os.path.join(ROOT, "Ramen_masks")) if f.endswith(".png"))
    assert len(names) == 555, len(names)

    by_soup = defaultdict(list)
    for n in names:
        by_soup[soup[n]].append(n)

    rng = random.Random(SEED)
    val = []
    # proportional per-soup quota so the 55 held-out images mirror the full distribution
    for s in sorted(by_soup):
        pool = sorted(by_soup[s])
        rng.shuffle(pool)
        k = round(NUM_VAL * len(pool) / len(names))
        val += pool[:k]
    # fix off-by-a-few from rounding
    rest = [n for n in names if n not in set(val)]
    rng.shuffle(rest)
    while len(val) < NUM_VAL:
        val.append(rest.pop())
    while len(val) > NUM_VAL:
        val.pop()

    val_set = set(val)
    train = [n for n in names if n not in val_set]
    assert len(train) == 500 and len(val) == 55
    assert not (set(train) & val_set)

    for split, items in [("train", train), ("val", val)]:
        with open(os.path.join(OUT, f"{split}.txt"), "w") as f:
            f.write("\n".join(sorted(items)) + "\n")
        print(split, len(items), dict(Counter(soup[n] for n in items)))


if __name__ == "__main__":
    main()
