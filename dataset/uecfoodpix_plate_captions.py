"""UECFOODPIXCOMPLETE + UECFOODPIX_Plate (皿マスク) 版のクラス語彙とキャプション。

FLUX.1 版 / 最初の FLUX.2 版 (dataset/uecfoodpix_captions.py) では id 0 を "background" の 1 語にしていた。
その LoRA は背景の見た目をほとんど学習できず、推論で青っぽい無機質な背景が出た (result/uecfoodpix_flux2_test10)。
そこで背景を 2 つに分けて具体的な caption を付け、画像全体の global caption も付ける:

    0   : 皿以外の背景 (テーブルなど)  -> TABLE_CAPTION
    103 : 皿・器・お盆 (UECFOODPIX_Plate の全クラス分の和から料理を除いた領域) -> PLATE_CAPTION
    1..102 : 料理 (名前は uecfoodpix_captions と同じ、"others" -> "food")

マスクは dataset/make_uecfoodpix_plate_masks.py で前処理済みの 1ch PNG (画素値 = 上の id)。
学習 (dataset/uecfoodpix_plate_flux2_dataset.py) と推論 (gen_uecfoodpix_flux2_test.py --plate) の両方がここを import する。
文言を変えたら学習し直すこと。
"""
from dataset.uecfoodpix_captions import CLASS_ID_TO_NAME as FOOD_ID_TO_NAME
from dataset.uecfoodpix_captions import regional_caption as food_caption

TABLE_ID = 0
PLATE_ID = 103
NUM_CLASSES = 104

TABLE_CAPTION = "table surface and surroundings"
PLATE_CAPTION = "plates, bowls and tray"


def regional_caption(class_id):
    if class_id == TABLE_ID:
        return TABLE_CAPTION
    if class_id == PLATE_ID:
        return PLATE_CAPTION
    return food_caption(FOOD_ID_TO_NAME[class_id])


def global_caption(class_ids):
    """例: "A photo of a meal: rice, miso soup and grilled salmon, served on tableware on a table." """
    names = []
    for i in class_ids:
        if i in (TABLE_ID, PLATE_ID):
            continue
        n = food_caption(FOOD_ID_TO_NAME[i]).strip()
        if n == "food":  # id 101 "others"。regional では "food" だが、料理名の列挙の中では浮くので言い換える
            n = "other dishes"
        if n not in names:
            names.append(n)
    if not names:
        dishes = "food"
    elif len(names) == 1:
        dishes = names[0]
    else:
        dishes = ", ".join(names[:-1]) + " and " + names[-1]
    return f"A photo of a meal: {dishes}, served on tableware on a table."
