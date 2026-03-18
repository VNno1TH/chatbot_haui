"""
_loader.py  (private)
Load và cache các file JSON — chỉ đọc file 1 lần duy nhất.
"""

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

JSON_FILES = {
    "diem_chuan"   : DATA_DIR / "diem_chuan_2023_2024_2025.json",
    "hoc_phi"      : DATA_DIR / "muc_thu_hoc_phi.json",
    "chi_tieu"     : DATA_DIR / "chi_tieu_to_hop_2025.json",
    "chi_tieu_2026": DATA_DIR / "chi_tieu_tuyen_sinh_2026.json",
    "to_hop"       : DATA_DIR / "to_hop_mon_thi.json",
    "diem_uu_tien" : DATA_DIR / "diem_uu_tien.json",
    "diem_quy_doi" : DATA_DIR / "diem_quy_doi.json",
}

_cache: dict[str, Any] = {}


def load(key: str) -> Any:
    """Load JSON theo key, cache lại sau lần đầu đọc."""
    if key not in _cache:
        path = JSON_FILES.get(key)
        if path is None:
            raise KeyError(f"Không tìm thấy key '{key}' trong JSON_FILES.")
        if not path.exists():
            raise FileNotFoundError(f"File không tồn tại: {path}")
        _cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[key]


def clear_cache():
    """Xóa cache — dùng khi cần reload dữ liệu mới."""
    _cache.clear()
