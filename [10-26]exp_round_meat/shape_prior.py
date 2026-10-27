"""学習データでの具材ごとの形 (円形度・縦横比・面積) の分布。チャーシューを丸く描くと卵になる問題の原因調べ。
円形度 = 4πA/P² (円で 1)。連結成分ごと、面積 200px 未満は捨てる。"""
import os, sys
import numpy as np, cv2
from PIL import Image
D = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_masks"
train = [l.strip() for l in open(os.path.expanduser("~/Seg2Any/data/ramen/train.txt")) if l.strip()]
IDS = {5: "cut_egg", 6: "egg", 8: "meat", 7: "seaweed", 9: "menma", 10: "naruto"}
st = {n: [] for n in IDS.values()}
for name in train:
    name = os.path.basename(name); name = name if name.endswith(".png") else name.split(".")[0] + ".png"
    m = np.array(Image.open(os.path.join(D, name)))
    if m.ndim == 3: m = m[..., 0]
    for k, n in IDS.items():
        num, lab = cv2.connectedComponents((m == k).astype(np.uint8))
        for i in range(1, num):
            c = (lab == i).astype(np.uint8)
            a = c.sum()
            if a < 200: continue
            cs, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            p = sum(cv2.arcLength(x, True) for x in cs)
            (_, _), (w, h), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
            st[n].append((4 * np.pi * a / p ** 2, min(w, h) / max(w, h, 1), a / m.size))
print(f"{'class':8s} {'n':>5s} {'circ med':>8s} {'circ>0.8':>8s} {'aspect med':>10s} {'area% med':>9s}")
for n, v in st.items():
    v = np.array(v)
    print(f"{n:8s} {len(v):5d} {np.median(v[:,0]):8.2f} {(v[:,0]>0.8).mean():8.2f} {np.median(v[:,1]):10.2f} {100*np.median(v[:,2]):9.2f}")
# 円形度>0.8 の成分のうち、どのクラスか
allc = [(n, x[0]) for n, v in st.items() for x in v]
rnd = [n for n, c in allc if c > 0.8]
print("circ>0.8 components by class:", {n: rnd.count(n) for n in st})
