"""SA-1B の tar をストリーミングで読み、SACap-1M が使う画像と json だけを保存する。

tar 全体 (1 個 ~11GB) はディスクに置かない。curl の出力を tarfile のストリームモードで読み、
anno_train.parquet に載っている imagename の .jpg / .json だけを <out_root>/<image_group>/ に書く。

    python scripts/prepare_sacap_images.py --links sa1b_links.txt --anno anno_train.parquet \
        --out_root /export/ssd3/lan-y/SACap-1M/raw --groups sa_000000 sa_000001 ...

完了した group は <out_root>/<group>/.done を置き、再実行時はスキップする。
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # リポジトリ直下 (src/ dataset/ utils/ infer.py) を import できるように
import argparse
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def load_links(path):
    links = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                links[parts[0].replace(".tar", "")] = parts[1]
    return links


def fetch_group(group, url, wanted, out_root):
    out_dir = os.path.join(out_root, group)
    done_flag = os.path.join(out_dir, ".done")
    if os.path.exists(done_flag):
        return group, 0, "skip"
    os.makedirs(out_dir, exist_ok=True)

    # tar のメンバー名 ("./sa_1.jpg") は信用せず、basename が wanted に入っているものだけ自前のパスに書く
    wanted_files = set()
    for name in wanted:
        stem = name[: name.rfind(".")]
        wanted_files.add(stem + ".jpg")
        wanted_files.add(stem + ".json")

    proc = subprocess.Popen(
        ["curl", "-sSfL", "--retry", "5", "--retry-delay", "10", url],
        stdout=subprocess.PIPE,
    )
    n = 0
    try:
        # SA-1B の .tar は実際は gzip なので圧縮形式は自動判定
        with tarfile.open(fileobj=proc.stdout, mode="r|*") as tf:
            for member in tf:
                base = os.path.basename(member.name)
                if not member.isfile() or base not in wanted_files:
                    continue
                data = tf.extractfile(member).read()
                tmp = os.path.join(out_dir, base + ".tmp")
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, os.path.join(out_dir, base))
                n += 1
    finally:
        proc.stdout.close()
        rc = proc.wait()
    if rc != 0:
        return group, n, f"curl exit {rc}"
    if n != len(wanted_files):
        return group, n, f"incomplete: {n}/{len(wanted_files)}"
    open(done_flag, "w").close()
    return group, n, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True)
    ap.add_argument("--anno", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    links = load_links(args.links)
    df = pd.read_parquet(args.anno, columns=["imagename", "image_group"])
    by_group = df.groupby("image_group")["imagename"].apply(list).to_dict()

    tasks = []
    for g in args.groups:
        if g not in links:
            sys.exit(f"no link for {g}")
        tasks.append((g, links[g], by_group.get(g, []), args.out_root))

    with ThreadPoolExecutor(args.jobs) as ex:
        for group, n, status in ex.map(lambda t: fetch_group(*t), tasks):
            print(f"{group}: {n} files, {status}", flush=True)


if __name__ == "__main__":
    main()
