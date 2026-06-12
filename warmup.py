"""
warmup.py — tải embedding model về cho tất cả các folder implementation.

Chạy 1 lần khi còn internet:

    python warmup.py

Sau khi chạy xong, mỗi folder sẽ có sẵn ./models/vietnamese-sbert để chạy offline:

    bell/models/vietnamese-sbert
    minhtran/models/vietnamese-sbert
    Vu/models/vietnamese-sbert
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path


MODEL_NAME = "keepitreal/vietnamese-sbert"
TARGET_FOLDERS = ["bell", "minhtran", "Vu"]

ROOT = Path(__file__).resolve().parent


def ensure_packages() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("[ERROR] sentence-transformers chưa được cài.")
        print("        Chạy trước: pip install -r requirements.txt")
        sys.exit(1)


def download_to(save_dir: Path) -> None:
    from sentence_transformers import SentenceTransformer

    save_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(save_dir))


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    ensure_packages()

    print(f"[*] Tải model: {MODEL_NAME}")
    print(f"[*] Root: {ROOT}\n")

    staging = ROOT / "models" / "vietnamese-sbert"

    if staging.is_dir() and any(staging.iterdir()):
        print(f"[*] Phát hiện model đã có sẵn tại {staging}")
        print("    -> bỏ qua bước tải, chỉ copy sang các folder.")
    else:
        print(f"[*] Đang tải về {staging} ...")
        start = time.time()
        download_to(staging)
        print(f"[OK] Tải xong sau {time.time() - start:.1f}s")

    for folder in TARGET_FOLDERS:
        dst = ROOT / folder / "models" / "vietnamese-sbert"
        if not (ROOT / folder).is_dir():
            print(f"[skip] Không thấy folder {folder}/")
            continue

        print(f"[*] Copy -> {dst}")
        copy_tree(staging, dst)

    print("\n[*] Smoke test (load model từ bell/models/vietnamese-sbert)...")
    try:
        from sentence_transformers import SentenceTransformer

        test_path = ROOT / "bell" / "models" / "vietnamese-sbert"
        if test_path.is_dir():
            m = SentenceTransformer(str(test_path))
            vec = m.encode(["xin chào", "đây là test"], normalize_embeddings=True)
            print(f"[OK] Embedding dim = {vec.shape[1]}, shape = {vec.shape}")
        else:
            print(f"[WARN] Không tìm thấy {test_path} để test.")
    except Exception as exc:
        print(f"[WARN] Smoke test fail: {exc}")

    print("\n[DONE] Warmup hoàn tất. Bây giờ có thể chạy bất kỳ folder nào offline.")


if __name__ == "__main__":
    main()
