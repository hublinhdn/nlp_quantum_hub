#!/usr/bin/env python3
"""Patch lambeq's dead MODELS_URL.

Background:
    Cambridge Quantum Computing đã sáp nhập với Honeywell thành Quantinuum
    cuối 2021. Domain cũ `qnlp.cambridgequantum.com` hiện đã chết. Nhưng
    lambeq vẫn hardcode URL này trong `model_downloader.py`.

Strategy:
    1. Tìm file model_downloader.py trong môi trường lambeq
    2. Thử các URL thay thế (Quantinuum, Wayback Machine)
    3. Nếu có URL hoạt động → patch file (backup file gốc)
    4. Nếu fail tất cả → in hướng dẫn manual

Cách dùng:
    python scripts/00_fix_lambeq_url.py            # tự động tìm + patch
    python scripts/00_fix_lambeq_url.py --check    # chỉ kiểm tra, không patch
    python scripts/00_fix_lambeq_url.py --restore  # khôi phục backup
    python scripts/00_fix_lambeq_url.py --url URL  # patch với URL chỉ định
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests


ORIGINAL_URL = "https://qnlp.cambridgequantum.com/models"
TEST_VERSION_PATH = "/bobcat/latest/version.txt"

# Các URL thay thế thử theo thứ tự
ALTERNATIVE_URLS = [
    "https://qnlp.quantinuum.com/models",
    # Thêm các mirror khác khi phát hiện
]


# ---------------------------------------------------------------------------
# Locate lambeq file
# ---------------------------------------------------------------------------


def find_downloader_file() -> Path | None:
    try:
        import lambeq
    except ImportError:
        return None
    base = Path(lambeq.__file__).parent
    files = list(base.rglob("model_downloader.py"))
    return files[0] if files else None


def find_current_url(file_path: Path) -> str | None:
    """Tìm URL hiện tại trong MODELS_URL = '...'."""
    content = file_path.read_text()
    m = re.search(r"MODELS_URL\s*=\s*['\"]([^'\"]+)['\"]", content)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Test URLs
# ---------------------------------------------------------------------------


def test_version(base_url: str, timeout: int = 10) -> tuple[bool, str | None]:
    """Test {base_url}/bobcat/latest/version.txt. Trả về (works, version_text)."""
    try:
        r = requests.get(base_url + TEST_VERSION_PATH, timeout=timeout)
        if r.status_code == 200:
            return True, r.text.strip()
    except Exception:
        pass
    return False, None


def test_model_tar(base_url: str, version: str, timeout: int = 15) -> bool:
    """HEAD check file model.tar.gz có truy cập được không."""
    url = f"{base_url}/bobcat/{version}/model.tar.gz"
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def test_full(base_url: str) -> tuple[bool, str | None, str]:
    """Test cả version.txt + model.tar.gz. Trả về (ok, version, reason)."""
    works, version = test_version(base_url)
    if not works:
        return False, None, "version.txt không truy cập được"
    if not version:
        return False, None, "version.txt trả về rỗng"
    if not test_model_tar(base_url, version):
        return False, version, f"version OK={version} nhưng model.tar.gz không có"
    return True, version, "OK"


# ---------------------------------------------------------------------------
# Wayback Machine
# ---------------------------------------------------------------------------


def get_wayback_base(timeout: int = 15) -> str | None:
    """Tìm Wayback snapshot của version.txt. Trả về base URL có id_ modifier."""
    probe = ORIGINAL_URL + TEST_VERSION_PATH
    api = f"http://archive.org/wayback/available?url={probe}"
    try:
        r = requests.get(api, timeout=timeout)
        data = r.json()
    except Exception as e:
        print(f"    Wayback API error: {e}")
        return None

    snap = data.get("archived_snapshots", {}).get("closest", {})
    if not snap.get("available"):
        return None
    ts = snap.get("timestamp")
    if not ts:
        return None
    # id_ trả về raw content (không có Wayback toolbar)
    return f"https://web.archive.org/web/{ts}id_/{ORIGINAL_URL}"


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------


def backup_path_for(file_path: Path) -> Path:
    return file_path.with_suffix(file_path.suffix + ".bak")


def patch_file(file_path: Path, new_url: str, old_url: str = ORIGINAL_URL) -> Path:
    """Replace old_url với new_url trong file. Lưu backup."""
    content = file_path.read_text()
    if old_url not in content:
        sys.exit(f"Không thấy '{old_url}' trong {file_path}")
    backup = backup_path_for(file_path)
    if not backup.exists():
        backup.write_text(content)
    file_path.write_text(content.replace(old_url, new_url))
    return backup


def restore_file(file_path: Path) -> bool:
    backup = backup_path_for(file_path)
    if not backup.is_file():
        return False
    file_path.write_text(backup.read_text())
    backup.unlink()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không patch")
    p.add_argument("--restore", action="store_true", help="Khôi phục file gốc từ .bak")
    p.add_argument("--url", help="Patch với URL chỉ định (skip auto-discovery)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("  lambeq MODELS_URL patcher")
    print("=" * 64)

    file_path = find_downloader_file()
    if not file_path:
        sys.exit("Không tìm thấy lambeq. Cài đặt: pip install lambeq")
    print(f"\n[file] {file_path}")

    current_url = find_current_url(file_path)
    print(f"[current MODELS_URL] {current_url}")

    if args.restore:
        if restore_file(file_path):
            print("[restore] Đã khôi phục file gốc từ .bak")
        else:
            print("[restore] Không thấy backup để khôi phục")
        return

    # User chỉ định URL trực tiếp
    if args.url:
        print(f"\n[user-specified] Test {args.url}")
        ok, version, reason = test_full(args.url)
        if not ok:
            sys.exit(f"  ✗ URL không hoạt động: {reason}")
        print(f"  ✓ {args.url}  (version={version})")
        if args.check:
            return
        backup = patch_file(file_path, args.url, current_url or ORIGINAL_URL)
        print(f"\n[done] Patched. Backup: {backup}")
        return

    # Auto-discovery
    # Step 1: Test URL hiện tại
    print(f"\n[step 1] Test URL hiện tại")
    if current_url:
        ok, version, reason = test_full(current_url)
        status = "✓" if ok else "✗"
        print(f"  {status} {current_url}  ({reason})")
        if ok:
            print(f"\n[done] URL hiện tại hoạt động. Không cần patch.")
            return

    # Step 2: Thử alternatives
    print(f"\n[step 2] Thử các URL alternatives")
    for url in ALTERNATIVE_URLS:
        ok, version, reason = test_full(url)
        status = "✓" if ok else "✗"
        print(f"  {status} {url}  ({reason})")
        if ok:
            if args.check:
                return
            backup = patch_file(file_path, url, current_url or ORIGINAL_URL)
            print(f"\n[done] Patched lambeq → {url}")
            print(f"       Backup: {backup}")
            print(f"\nChạy lại smoke test:")
            print(f"  python scripts/05a_parse_smoke.py")
            return

    # Step 3: Wayback Machine
    print(f"\n[step 3] Tìm Wayback Machine snapshot")
    wayback = get_wayback_base()
    if wayback:
        print(f"  Found: {wayback}")
        ok, version, reason = test_full(wayback)
        status = "✓" if ok else "✗"
        print(f"  {status} Wayback ({reason})")
        if ok:
            if args.check:
                return
            backup = patch_file(file_path, wayback, current_url or ORIGINAL_URL)
            print(f"\n[done] Patched → Wayback Machine")
            print(f"       Backup: {backup}")
            print(f"\nLưu ý: Wayback chậm, lần đầu tải model có thể mất 5–15 phút.")
            print(f"\nChạy lại smoke test:")
            print(f"  python scripts/05a_parse_smoke.py")
            return
    else:
        print(f"  ✗ Không có snapshot")

    # Step 4: Manual fallback
    print(f"\n[FAIL] Tất cả lựa chọn auto đều không hoạt động.")
    print(f"\nThử các bước sau theo thứ tự:")
    print(f"  1. Upgrade lambeq:")
    print(f"     pip install --upgrade lambeq")
    print(f"     # Bản mới có thể đã sửa URL")
    print(f"")
    print(f"  2. Chạy lại script này:")
    print(f"     python scripts/00_fix_lambeq_url.py")
    print(f"")
    print(f"  3. Manual: tự tải bobcat model và truyền local path:")
    print(f"     # Tải model về data/raw/bobcat/")
    print(f"     # Sửa src/discocat/parse.py để truyền model_name_or_path")
    print(f"")
    print(f"  4. Patch URL bằng tay với mirror bạn biết:")
    print(f"     python scripts/00_fix_lambeq_url.py --url <YOUR_MIRROR>")


if __name__ == "__main__":
    main()
