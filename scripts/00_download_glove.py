#!/usr/bin/env python3
"""Tải GloVe 6B (Wikipedia + Gigaword) và extract CHỈ phiên bản 50d.

Nguồn: http://nlp.stanford.edu/data/glove.6B.zip  (~ 822 MB nén, chứa 50d/100d/200d/300d)
Sau khi giải nén, ta chỉ giữ `glove.6B.50d.txt` (~ 170 MB) và xóa zip.

Đích: data/raw/glove/glove.6B.50d.txt

Cách dùng:
    python scripts/00_download_glove.py
    python scripts/00_download_glove.py --keep-zip      # giữ lại file zip
    python scripts/00_download_glove.py --extract-only  # bỏ qua download nếu zip đã có
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
DEFAULT_DEST = Path("data/raw/glove")
TARGET_MEMBER = "glove.6B.50d.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--keep-zip", action="store_true", help="Giữ glove.6B.zip sau khi extract")
    p.add_argument("--extract-only", action="store_true", help="Bỏ qua download nếu đã có zip")
    p.add_argument("--force", action="store_true", help="Tải lại kể cả khi đã có file")
    return p.parse_args()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"[download] → {target}  (~ 822 MB)")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(target, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="downloading"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def extract_member(zip_path: Path, dest: Path, member: str) -> Path:
    print(f"[extract] {member} từ {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / member
    with zipfile.ZipFile(zip_path, "r") as zf:
        if member not in zf.namelist():
            print(f"[lỗi] Zip không chứa {member}. Các member có: {zf.namelist()}")
            sys.exit(1)
        with zf.open(member) as src, open(out_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return out_path


def main() -> None:
    args = parse_args()
    zip_path = args.dest / "glove.6B.zip"
    target = args.dest / TARGET_MEMBER

    if target.is_file() and not args.force:
        size = target.stat().st_size
        print(f"[skip] Đã có {target} ({size:,} bytes). Dùng --force để tải lại.")
        return

    if not args.extract_only:
        if zip_path.is_file() and not args.force:
            print(f"[skip] Đã có {zip_path}. Dùng --force để tải lại.")
        else:
            download(args.url, zip_path)

    if not zip_path.is_file():
        print(f"[lỗi] Không tìm thấy {zip_path}")
        sys.exit(1)

    extracted = extract_member(zip_path, args.dest, TARGET_MEMBER)
    print(f"  ✓ {extracted}  ({extracted.stat().st_size:,} bytes)")

    if not args.keep_zip:
        print(f"[clean] xóa {zip_path} (dùng --keep-zip để giữ)")
        zip_path.unlink()

    print(f"\n[done] GloVe sẵn sàng. Chạy:")
    print(f"  python scripts/04_train_baselines.py --models tfidf bilstm glove")


if __name__ == "__main__":
    main()
