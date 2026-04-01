"""
nganh.py  (v2 — Unified khoa/trường alias + cross-source lookup)

Thay đổi so với v1:
  [FIX-1] _KHOA_ALIAS bao phủ đủ 5 Trường + 4 Khoa với mọi biến thể tên
  [FIX-2] _resolve_khoa dùng fuzzy token matching thay vì chỉ substring
  [FIX-3] get_nganh_theo_khoa đọc cả từ MD frontmatter (truong_khoa) nếu JSON thiếu
  [FIX-4] build_nganh_index() tạo unified index từ cả JSON lẫn MD files
"""

import re
from pathlib import Path
from ._loader import load
from ._utils  import normalize, match_nganh, not_found, ok


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED KHOA / TRƯỜNG ALIAS MAP
#
# Key = bất kỳ cách người dùng có thể gõ (lowercase, không dấu, viết tắt)
# Value = nhom_nganh chuẩn trong chi_tieu_to_hop_2025.json
#
# CÁC TRƯỜNG TRỰC THUỘC HaUI (5 trường):
#   - Trường CNTT & Truyền thông  → nhom "CNTT"
#   - Trường Cơ khí - Ô tô        → nhom "Cơ khí"
#   - Trường Kinh tế               → nhom "Kinh tế"
#   - Trường Ngoại ngữ - Du lịch   → nhom "Ngôn ngữ" + "Du lịch"
#   - Trường Điện - Điện tử        → nhom "Cơ khí" (cùng nhom trong JSON)
#
# CÁC KHOA TRỰC THUỘC (4 khoa):
#   - Khoa Công nghệ Hóa           → nhom "Cơ khí" (trong JSON) / "Hóa" riêng
#   - Khoa Công nghệ May & TKTT    → nhom "Dệt may"
#   - Khoa Lý luận Chính trị & PL  → không có ngành tuyển sinh
#   - Khoa Khoa học cơ bản         → không có ngành tuyển sinh
# ─────────────────────────────────────────────────────────────────────────────

_KHOA_ALIAS: dict[str, str] = {
    # ── Trường CNTT & Truyền thông ──────────────────────────────────────────
    "cntt"                                          : "CNTT",
    "công nghệ thông tin"                           : "CNTT",
    "cntt và truyền thông"                          : "CNTT",
    "cntt & truyền thông"                           : "CNTT",
    "công nghệ thông tin và truyền thông"           : "CNTT",
    "công nghệ thông tin & truyền thông"            : "CNTT",
    "trường cntt"                                   : "CNTT",
    "trường cntt và truyền thông"                   : "CNTT",
    "trường cntt & truyền thông"                    : "CNTT",
    "trường công nghệ thông tin"                    : "CNTT",
    "trường công nghệ thông tin và truyền thông"    : "CNTT",
    "trường công nghệ thông tin & truyền thông"     : "CNTT",
    "it"                                            : "CNTT",
    "khoa cntt"                                     : "CNTT",
    "khoa công nghệ thông tin"                      : "CNTT",

    # ── Trường Cơ khí - Ô tô ───────────────────────────────────────────────
    "cơ khí"                                        : "Cơ khí",
    "co khi"                                        : "Cơ khí",
    "cơ khí ô tô"                                   : "Cơ khí",
    "cơ khí - ô tô"                                 : "Cơ khí",
    "co khi o to"                                   : "Cơ khí",
    "trường cơ khí"                                 : "Cơ khí",
    "trường cơ khí ô tô"                            : "Cơ khí",
    "trường cơ khí - ô tô"                          : "Cơ khí",
    "truong co khi o to"                            : "Cơ khí",
    "khoa cơ khí"                                   : "Cơ khí",
    "ô tô"                                          : "Cơ khí",  # nếu hỏi về trường có ngành ô tô
    "điện"                                          : "Cơ khí",  # Trường Điện - Điện tử cũng nhóm Cơ khí
    "điện điện tử"                                  : "Cơ khí",
    "điện - điện tử"                                : "Cơ khí",
    "trường điện"                                   : "Cơ khí",
    "trường điện điện tử"                           : "Cơ khí",
    "trường điện - điện tử"                         : "Cơ khí",

    # ── Trường Kinh tế ──────────────────────────────────────────────────────
    "kinh tế"                                       : "Kinh tế",
    "kinh te"                                       : "Kinh tế",
    "trường kinh tế"                                : "Kinh tế",
    "truong kinh te"                                : "Kinh tế",
    "khoa kinh tế"                                  : "Kinh tế",
    "kinh doanh"                                    : "Kinh tế",
    "quản trị"                                      : "Kinh tế",
    "tài chính"                                     : "Kinh tế",
    "kế toán"                                       : "Kinh tế",

    # ── Trường Ngoại ngữ - Du lịch (ngôn ngữ) ──────────────────────────────
    "ngoại ngữ"                                     : "Ngôn ngữ",
    "ngôn ngữ"                                      : "Ngôn ngữ",
    "ngoai ngu"                                     : "Ngôn ngữ",
    "ngon ngu"                                      : "Ngôn ngữ",
    "trường ngoại ngữ"                              : "Ngôn ngữ",
    "trường ngôn ngữ"                               : "Ngôn ngữ",
    "trường ngoại ngữ du lịch"                      : "Ngôn ngữ",
    "trường ngoại ngữ - du lịch"                    : "Ngôn ngữ",
    "ngoại ngữ du lịch"                             : "Ngôn ngữ",
    "tiếng anh"                                     : "Ngôn ngữ",
    "tiếng nhật"                                    : "Ngôn ngữ",
    "tiếng trung"                                   : "Ngôn ngữ",
    "tiếng hàn"                                     : "Ngôn ngữ",
    "khoa ngoại ngữ"                                : "Ngôn ngữ",

    # ── Du lịch (cũng thuộc Trường Ngoại ngữ - Du lịch) ────────────────────
    "du lịch"                                       : "Du lịch",
    "du lich"                                       : "Du lịch",
    "trường du lịch"                                : "Du lịch",
    "khoa du lịch"                                  : "Du lịch",
    "khách sạn"                                     : "Du lịch",
    "nhà hàng"                                      : "Du lịch",
    "lữ hành"                                       : "Du lịch",

    # ── Khoa Công nghệ Hóa ──────────────────────────────────────────────────
    "hóa"                                           : "Hóa",
    "hoa"                                           : "Hóa",
    "công nghệ hóa"                                 : "Hóa",
    "công nghệ hóa học"                             : "Hóa",
    "khoa hóa"                                      : "Hóa",
    "khoa công nghệ hóa"                            : "Hóa",
    "khoa công nghệ hóa học"                        : "Hóa",
    "môi trường"                                    : "Hóa",  # ngành KT môi trường thuộc Khoa Hóa
    "hóa dược"                                      : "Hóa",  # thuộc Khoa Hóa dù JSON nhom=Thực phẩm
    "hoa duoc"                                      : "Hóa",

    # ── Khoa Công nghệ May & Thiết kế thời trang ────────────────────────────
    "dệt may"                                       : "Dệt may",
    "det may"                                       : "Dệt may",
    "may"                                           : "Dệt may",
    "thời trang"                                    : "Dệt may",
    "thiết kế thời trang"                           : "Dệt may",
    "khoa dệt may"                                  : "Dệt may",
    "khoa may"                                      : "Dệt may",
    "khoa công nghệ may"                            : "Dệt may",
    "khoa công nghệ may và thiết kế thời trang"     : "Dệt may",

    # ── Thực phẩm & Hóa dược → cùng Khoa Công nghệ Hóa ────────────────────
    # (trong JSON nhom="Thực phẩm", nhưng về mặt tổ chức đều thuộc Khoa Hóa)
    "thực phẩm"                                     : "Hóa",
    "thuc pham"                                     : "Hóa",
    "công nghệ thực phẩm"                           : "Hóa",
    "khoa thực phẩm"                                : "Hóa",
    "dược"                                          : "Hóa",
}

# Map từ nhom_nganh chuẩn → tên hiển thị đẹp
_NHOM_DISPLAY: dict[str, str] = {
    "CNTT"      : "Trường CNTT & Truyền thông",
    "Cơ khí"    : "Trường Cơ khí - Ô tô / Điện - Điện tử",
    "Kinh tế"   : "Trường Kinh tế",
    "Ngôn ngữ"  : "Trường Ngoại ngữ - Du lịch (ngành Ngôn ngữ)",
    "Du lịch"   : "Trường Ngoại ngữ - Du lịch (ngành Du lịch)",
    "Hóa"       : "Khoa Công nghệ Hóa",
    "Dệt may"   : "Khoa Công nghệ May & Thiết kế thời trang",
    "Thực phẩm" : "Khoa Thực phẩm / Hóa dược",
}

# Map nhom_nganh trong JSON → nhom chuẩn của chúng ta
# (Khoa Hóa trong JSON thuộc nhom "Cơ khí" nhưng ta tách riêng)
_JSON_NHOM_TO_INTERNAL: dict[str, str] = {
    "CNTT"      : "CNTT",
    "Cơ khí"    : "Cơ khí",      # JSON gộp cả Cơ khí + Điện + Hóa + Môi trường
    "Kinh tế"   : "Kinh tế",
    "Ngôn ngữ"  : "Ngôn ngữ",
    "Du lịch"   : "Du lịch",
    "Dệt may"   : "Dệt may",
    "Thực phẩm" : "Thực phẩm",
}

# Các ngành thuộc Khoa Công nghệ Hóa (4 ngành)
# 7510401 = Công nghệ kỹ thuật hóa học  (nhom=Cơ khí trong JSON)
# 7510406 = Công nghệ kỹ thuật môi trường (nhom=Cơ khí trong JSON)
# 7540101 = Công nghệ thực phẩm          (nhom=Thực phẩm trong JSON)
# 7720203 = Hóa dược                     (nhom=Thực phẩm trong JSON)
_MA_NGANH_KHOA_HOA = {"7510401", "7510406", "7540101", "7720203"}

# Các ngành thuộc Trường Điện - Điện tử
_MA_NGANH_TRUONG_DIEN = {
    "7510301", "7510301TA", "7510302", "7510302TA",
    "75103021", "7510303", "75103031", "75190071",
    "7510206",  # kỹ thuật nhiệt
}


# ─────────────────────────────────────────────────────────────────────────────
# FUZZY RESOLVE
# ─────────────────────────────────────────────────────────────────────────────

def _remove_diacritics_simple(text: str) -> str:
    """Xóa dấu thanh đơn giản cho fuzzy match."""
    replacements = {
        'à':'a','á':'a','ả':'a','ã':'a','ạ':'a',
        'ă':'a','ắ':'a','ặ':'a','ằ':'a','ẳ':'a','ẵ':'a',
        'â':'a','ấ':'a','ầ':'a','ẩ':'a','ẫ':'a','ậ':'a',
        'è':'e','é':'e','ẻ':'e','ẽ':'e','ẹ':'e',
        'ê':'e','ế':'e','ề':'e','ể':'e','ễ':'e','ệ':'e',
        'ì':'i','í':'i','ỉ':'i','ĩ':'i','ị':'i',
        'ò':'o','ó':'o','ỏ':'o','õ':'o','ọ':'o',
        'ô':'o','ố':'o','ồ':'o','ổ':'o','ỗ':'o','ộ':'o',
        'ơ':'o','ớ':'o','ờ':'o','ở':'o','ỡ':'o','ợ':'o',
        'ù':'u','ú':'u','ủ':'u','ũ':'u','ụ':'u',
        'ư':'u','ứ':'u','ừ':'u','ử':'u','ữ':'u','ự':'u',
        'ỳ':'y','ý':'y','ỷ':'y','ỹ':'y','ỵ':'y',
        'đ':'d',
    }
    result = text.lower()
    for src, dst in replacements.items():
        result = result.replace(src, dst)
    return result


def _resolve_khoa(ten_khoa: str) -> str | None:
    """
    Chuẩn hóa tên khoa/trường → nhom_nganh nội bộ.
    Trả None nếu không nhận ra.

    Thứ tự ưu tiên:
      1. Exact match sau normalize
      2. Fuzzy match (không dấu)
      3. Token overlap (≥ 2 từ chính trùng)
    """
    q = normalize(ten_khoa)

    # 1. Exact match
    if q in _KHOA_ALIAS:
        return _KHOA_ALIAS[q]

    # 2. Substring match (alias trong q hoặc q trong alias)
    for alias, nhom in _KHOA_ALIAS.items():
        if alias in q or q in alias:
            return nhom

    # 3. Không dấu fuzzy
    q_nodiac = _remove_diacritics_simple(q)
    for alias, nhom in _KHOA_ALIAS.items():
        alias_nodiac = _remove_diacritics_simple(alias)
        if alias_nodiac in q_nodiac or q_nodiac in alias_nodiac:
            return nhom

    # 4. Token overlap: ít nhất 2 token quan trọng trùng
    _STOP = {"và", "của", "là", "có", "các", "một", "trường", "khoa", "ngành"}
    q_tokens = {w for w in q.split() if w not in _STOP and len(w) >= 3}
    best_nhom, best_overlap = None, 1
    for alias, nhom in _KHOA_ALIAS.items():
        alias_tokens = {w for w in alias.split() if w not in _STOP and len(w) >= 3}
        overlap = len(q_tokens & alias_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_nhom = nhom
    if best_nhom:
        return best_nhom

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MD FRONTMATTER INDEX — đọc metadata từ các file .md ngành
# ─────────────────────────────────────────────────────────────────────────────

_MD_NGANH_INDEX: dict[str, dict] | None = None   # cache


def _build_md_index() -> dict[str, dict]:
    """
    Đọc tất cả file .md có loai='mo_ta_nganh' và index theo ma_nganh.
    Trả dict: ma_nganh → {ma_nganh, ten_nganh, truong_khoa, ...}

    Đây là giải pháp cho vấn đề mismatch:
    - JSON dùng nhom="Cơ khí"
    - MD dùng truong_khoa="Trường Cơ khí - Ô tô"
    Hàm này tạo bridge giữa hai nguồn.
    """
    global _MD_NGANH_INDEX
    if _MD_NGANH_INDEX is not None:
        return _MD_NGANH_INDEX

    try:
        import frontmatter as fm
    except ImportError:
        _MD_NGANH_INDEX = {}
        return _MD_NGANH_INDEX

    # Tìm thư mục chứa file .md (relative to this file)
    base_dirs = [
        Path(__file__).resolve().parent.parent.parent / "data" / "processed",
        Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "nganh",
    ]

    index: dict[str, dict] = {}
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for md_file in base_dir.rglob("*.md"):
            try:
                post = fm.load(str(md_file))
                meta = post.metadata
                if meta.get("loai") != "mo_ta_nganh":
                    continue
                ma = str(meta.get("ma_nganh", "")).strip()
                if not ma:
                    continue
                index[ma] = {
                    "ma_nganh"   : ma,
                    "ten_nganh"  : str(meta.get("ten_nganh", "")),
                    "truong_khoa": str(meta.get("truong_khoa", "")),
                    "file"       : str(md_file.name),
                }
            except Exception:
                continue

    _MD_NGANH_INDEX = index
    return _MD_NGANH_INDEX


def _get_nganh_tu_md_theo_truong(ten_truong: str) -> list[dict]:
    """
    Lấy danh sách ngành từ MD index theo tên trường/khoa.
    Dùng khi JSON không cover đủ.
    """
    md_idx = _build_md_index()
    if not md_idx:
        return []

    q = normalize(ten_truong)
    results = []
    for ma, info in md_idx.items():
        tk = normalize(info.get("truong_khoa", ""))
        # Kiểm tra substring hoặc fuzzy
        if q in tk or tk in q:
            results.append(info)
            continue
        # Fuzzy không dấu
        q_nd = _remove_diacritics_simple(q)
        tk_nd = _remove_diacritics_simple(tk)
        if q_nd in tk_nd or tk_nd in q_nd:
            results.append(info)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_chi_tieu_nganh(ten_nganh: str) -> dict:
    """
    Lấy chỉ tiêu, tổ hợp và phương thức xét tuyển của ngành.
    """
    data    = load("chi_tieu")
    results = [
        d for d in data
        if match_nganh(ten_nganh, d["ten_nganh"])
        or ten_nganh == d.get("ma_nganh", "")
    ]
    if not results:
        return not_found(f"Không tìm thấy ngành '{ten_nganh}'.")
    return ok(ket_qua=[
        {
            "ma_nganh"   : d["ma_nganh"],
            "ten_nganh"  : d["ten_nganh"],
            "nhom"       : d.get("nhom", ""),
            "chi_tieu"   : d["chi_tieu"],
            "to_hop"     : d["to_hop"],
            "phuong_thuc": d["phuong_thuc"],
            "nam"        : d.get("nam", 2025),
        }
        for d in results
    ])


def get_nganh_theo_to_hop(ma_to_hop: str) -> dict:
    """
    Tìm tất cả ngành xét tuyển bằng tổ hợp cho trước.
    """
    data    = load("chi_tieu")
    ma      = ma_to_hop.upper()
    results = [d for d in data if ma in [t.upper() for t in d["to_hop"]]]
    mon_thi = get_mon_thi_to_hop(ma).get("mon", [])
    if not results:
        return not_found(f"Không có ngành nào dùng tổ hợp {ma}.")
    return ok(
        ma_to_hop  = ma,
        mon_thi    = mon_thi,
        so_nganh   = len(results),
        nganh_list = [
            {
                "ma_nganh" : d["ma_nganh"],
                "ten_nganh": d["ten_nganh"],
                "nhom"     : d.get("nhom", ""),
                "chi_tieu" : d["chi_tieu"],
            }
            for d in sorted(results, key=lambda x: x["ten_nganh"])
        ],
    )


def get_chi_tieu_tong_2026() -> dict:
    raw = load("chi_tieu_2026")
    return ok(
        truong        = raw["truong"],
        nam           = raw["nam"],
        tong_chi_tieu = raw["tong_chi_tieu"],
        chi_tieu      = raw["chi_tieu"],
        ghi_chu       = raw.get("ghi_chu", ""),
    )


def get_mon_thi_to_hop(ma_to_hop: str) -> dict:
    data = load("to_hop")
    ma   = ma_to_hop.upper()
    for item in data:
        if item["ma"].upper() == ma:
            return ok(ma=item["ma"], mon=item["mon"])
    return not_found(f"Không tìm thấy tổ hợp '{ma_to_hop}'.")


def get_tat_ca_to_hop() -> list[dict]:
    return load("to_hop")


def get_nganh_theo_khoa(ten_khoa: str) -> dict:
    """
    Lấy danh sách ngành thuộc một khoa / trường.

    [FIX] Nay dùng _resolve_khoa mạnh hơn + cross-source với MD index.
    Ví dụ: "Trường Cơ khí - Ô tô" → nhom="Cơ khí" trong JSON → trả đúng ngành.
    """
    nhom = _resolve_khoa(ten_khoa)
    data = load("chi_tieu")

    if nhom is None:
        # Fallback: tìm trực tiếp trong JSON data
        q    = normalize(ten_khoa)
        q_nd = _remove_diacritics_simple(q)
        hits = []
        for d in data:
            nhom_norm = normalize(d.get("nhom", ""))
            nhom_nd   = _remove_diacritics_simple(nhom_norm)
            if q in nhom_norm or nhom_norm in q or q_nd in nhom_nd:
                hits.append(d)
        if hits:
            nhom = hits[0]["nhom"]
        else:
            # Last resort: MD index cross-source
            md_results = _get_nganh_tu_md_theo_truong(ten_khoa)
            if md_results:
                # Tìm ngành này trong JSON để lấy đủ thông tin chỉ tiêu
                ma_set    = {r["ma_nganh"] for r in md_results}
                json_hits = [d for d in data if d["ma_nganh"] in ma_set]
                if json_hits:
                    return ok(
                        ten_khoa   = ten_khoa,
                        nhom_nganh = json_hits[0].get("nhom", "?"),
                        so_nganh   = len(json_hits),
                        nganh_list = [
                            {"ma_nganh": d["ma_nganh"], "ten_nganh": d["ten_nganh"]}
                            for d in sorted(json_hits, key=lambda x: x["ten_nganh"])
                        ],
                        source = "md_index",
                    )
            return not_found(
                f"Không nhận ra khoa/trường '{ten_khoa}'.\n"
                f"Thử: CNTT, Cơ khí, Kinh tế, Du lịch, Dệt may, "
                f"Ngôn ngữ, Thực phẩm, Hóa, Điện."
            )

    # Lọc ngành theo nhom — với xử lý đặc biệt cho Khoa Hóa
    if nhom == "Hóa":
        # Khoa Hóa trong JSON thuộc nhom "Cơ khí", cần filter theo mã ngành
        nganh_ds = [
            d for d in data
            if d.get("ma_nganh") in _MA_NGANH_KHOA_HOA
        ]
        display_nhom = "Hóa"
    elif nhom == "Cơ khí":
        # Tách Trường Cơ khí vs Trường Điện nếu user hỏi cụ thể
        q_lower = ten_khoa.lower()
        if any(w in q_lower for w in ["điện", "dien", "electron"]):
            nganh_ds = [d for d in data if d.get("ma_nganh") in _MA_NGANH_TRUONG_DIEN]
            display_nhom = "Điện - Điện tử"
        else:
            # Mặc định: tất cả nhom Cơ khí (bao gồm cả điện, hóa, môi trường)
            nganh_ds = [d for d in data if normalize(d.get("nhom", "")) == normalize(nhom)]
            display_nhom = nhom
    else:
        nganh_ds = [
            d for d in data
            if normalize(d.get("nhom", "")) == normalize(nhom)
        ]
        display_nhom = nhom

    if not nganh_ds:
        return not_found(f"Không tìm thấy ngành nào thuộc khoa '{ten_khoa}'.")

    return ok(
        ten_khoa    = ten_khoa,
        nhom_nganh  = display_nhom,
        ten_hien_thi= _NHOM_DISPLAY.get(nhom, ten_khoa),
        so_nganh    = len(nganh_ds),
        nganh_list  = [
            {"ma_nganh": d["ma_nganh"], "ten_nganh": d["ten_nganh"]}
            for d in sorted(nganh_ds, key=lambda x: x["ten_nganh"])
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: danh sách tất cả trường / khoa để trả lời câu hỏi tổng quan
# ─────────────────────────────────────────────────────────────────────────────

def get_co_cau_truong_khoa() -> dict:
    """
    Trả về cơ cấu tổ chức đào tạo HaUI:
    - 5 Trường trực thuộc
    - 4 Khoa trực thuộc
    Mỗi entry kèm danh sách nhóm ngành tương ứng.
    """
    return ok(
        truong_truc_thuoc=[
            {"ten": "Trường CNTT & Truyền thông",   "nhom_json": ["CNTT"]},
            {"ten": "Trường Cơ khí - Ô tô",          "nhom_json": ["Cơ khí"]},
            {"ten": "Trường Kinh tế",                 "nhom_json": ["Kinh tế"]},
            {"ten": "Trường Ngoại ngữ - Du lịch",    "nhom_json": ["Ngôn ngữ", "Du lịch"]},
            {"ten": "Trường Điện - Điện tử",          "nhom_json": ["Cơ khí"]},  # gộp trong JSON
        ],
        khoa_truc_thuoc=[
            {"ten": "Khoa Công nghệ Hóa",                       "nhom_json": ["Cơ khí"]},
            {"ten": "Khoa Công nghệ May & Thiết kế thời trang", "nhom_json": ["Dệt may"]},
            {"ten": "Khoa Lý luận Chính trị & Pháp luật",       "nhom_json": []},
            {"ten": "Khoa Khoa học cơ bản",                     "nhom_json": []},
        ],
        ghi_chu=(
            "Trường Điện - Điện tử và Khoa Công nghệ Hóa được gộp chung "
            "trong nhóm 'Cơ khí' của dữ liệu JSON tuyển sinh."
        )
    )