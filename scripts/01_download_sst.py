#!/usr/bin/env python3
"""Tải Stanford Sentiment Treebank về data/raw/sst/.

Nguồn chính thức: https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
(~ 6 MB zip, ~ 50 MB khi giải nén).

Sau khi giải nén, các file sẽ ở:
    data/raw/sst/stanfordSentimentTreebank/
        datasetSentences.txt
        datasetSplit.txt
        dictionary.txt
        sentiment_labels.txt
        SOStr.txt, STree.txt, original_rt_snippets.txt

Cách dùng:
    python scripts/01_download_sst.py
    python scripts/01_download_sst.py --force      # tải lại
    python scripts/01_download_sst.py --url <URL>  # mirror khác
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_URL = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
DEFAULT_DEST = Path("data/raw/sst")

REQUIRED_FILES = [
    "datasetSentences.txt",
    "datasetSplit.txt",
    "dictionary.txt",
    "sentiment_labels.txt",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help=f"URL của zip (default: {DEFAULT_URL})")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"Thư mục đích (default: {DEFAULT_DEST})")
    p.add_argument("--force", action="store_true", help="Tải lại kể cả khi đã có file")
    p.add_argument("--chunk-size", type=int, default=1024 * 64, help="Kích thước chunk khi tải")
    return p.parse_args()


def already_extracted(dest: Path) -> bool:
    """Kiểm tra xem các file cần thiết đã tồn tại trong dest hay chưa."""
    base = dest / "stanfordSentimentTreebank"
    if not base.is_dir():
        return False
    return all((base / name).is_file() for name in REQUIRED_FILES)


def download(url: str, target: Path, chunk_size: int) -> None:
    """Tải file từ URL về target, hiện progress bar."""
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"[download] → {target}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(target, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="downloading"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def extract(zip_path: Path, dest: Path) -> None:
    """Giải nén zip vào dest."""
    print(f"[extract] {zip_path} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def verify(dest: Path) -> None:
    """In trạng thái các file bắt buộc."""
    base = dest / "stanfordSentimentTreebank"
    print(f"\n[verify] files trong {base}:")
    missing = []
    for name in REQUIRED_FILES:
        path = base / name
        if path.is_file():
            size = path.stat().st_size
            print(f"  ✓ {name}  ({size:,} bytes)")
        else:
            print(f"  ✗ {name}  (THIẾU)")
            missing.append(name)
    if missing:
        print(f"\n[verify] LỖI: thiếu {len(missing)} file: {missing}")
        sys.exit(1)
    print("\n[verify] OK — đã sẵn sàng cho scripts/02_prepare_data.py")


def main() -> None:
    args = parse_args()

    if not args.force and already_extracted(args.dest):
        print(f"[skip] Đã có dữ liệu tại {args.dest}/stanfordSentimentTreebank/")
        print("       Dùng --force để tải lại.")
        verify(args.dest)
        return

    zip_path = args.dest / "stanfordSentimentTreebank.zip"
    download(args.url, zip_path, args.chunk_size)
    extract(zip_path, args.dest)
    verify(args.dest)


if __name__ == "__main__":
    main()
