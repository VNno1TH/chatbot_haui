"""
diem_xet_tuyen.py
Tất cả logic tính toán liên quan đến điểm xét tuyển:
  1. Quy đổi điểm (HSA / TSA / KQHB)
  2. Tính điểm ưu tiên (khu vực + đối tượng)
  3. Kiểm tra đậu/trượt so với điểm chuẩn

Nguồn dữ liệu:
  - diem_quy_doi.json
  - diem_uu_tien.json
"""

from ._loader    import load
from ._utils     import not_found, ok
from .diem_chuan import get_diem_chuan


# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUY ĐỔI ĐIỂM
# ═══════════════════════════════════════════════════════════════════════════════

def _tra_bang(diem: float, bang: list[dict]) -> float | None:
    """Tra bảng quy đổi theo khoảng [tu, den]."""
    for row in bang:
        if row["tu"] <= diem <= row["den"]:
            return row["diem_quy_doi"]
    return None


def quy_doi_HSA(diem_hsa: float) -> dict:
    """
    Quy đổi điểm đánh giá năng lực ĐHQG Hà Nội (HSA, thang 150) → thang 30.

    Args:
        diem_hsa: Điểm HSA (75–150)
    """
    bang    = load("diem_quy_doi")["quy_doi_HSA"]["bang"]
    ket_qua = _tra_bang(diem_hsa, bang)

    if ket_qua is None:
        return not_found(
            f"Điểm HSA {diem_hsa} nằm ngoài bảng quy đổi (75–150)."
        )
    return ok(
        loai          = "HSA",
        ten           = "Đánh giá năng lực ĐHQG Hà Nội",
        diem_goc      = diem_hsa,
        thang_goc     = 150,
        diem_quy_doi  = ket_qua,
        thang_quy_doi = 30,
    )


def quy_doi_TSA(diem_tsa: float) -> dict:
    """
    Quy đổi điểm đánh giá tư duy ĐHBK Hà Nội (TSA, thang 100) → thang 30.

    Args:
        diem_tsa: Điểm TSA (50–100)
    """
    bang    = load("diem_quy_doi")["quy_doi_TSA"]["bang"]
    ket_qua = _tra_bang(diem_tsa, bang)

    if ket_qua is None:
        return not_found(
            f"Điểm TSA {diem_tsa} nằm ngoài bảng quy đổi (50–100)."
        )
    return ok(
        loai          = "TSA",
        ten           = "Đánh giá tư duy ĐHBK Hà Nội",
        diem_goc      = diem_tsa,
        thang_goc     = 100,
        diem_quy_doi  = ket_qua,
        thang_quy_doi = 30,
    )


def quy_doi_KQHB(diem_hb: float) -> dict:
    """
    Quy đổi điểm kết quả học bạ (thang 10) → thang 10 tương đương THPT.

    Args:
        diem_hb: Điểm trung bình học bạ môn (7.0–10.0)
    """
    bang    = load("diem_quy_doi")["quy_doi_KQHB"]["bang"]
    ket_qua = _tra_bang(diem_hb, bang)

    if ket_qua is None:
        return not_found(
            f"Điểm học bạ {diem_hb} nằm ngoài bảng quy đổi (7.0–10.0)."
        )
    return ok(
        loai          = "KQHB",
        ten           = "Kết quả học bạ THPT",
        diem_goc      = diem_hb,
        thang_goc     = 10,
        diem_quy_doi  = ket_qua,
        thang_quy_doi = 10,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ĐIỂM ƯU TIÊN
# ═══════════════════════════════════════════════════════════════════════════════

def get_diem_uu_tien_khu_vuc(ma_kv: str) -> dict:
    """
    Lấy mức điểm ưu tiên khu vực.

    Args:
        ma_kv: "KV1", "KV2-NT", "KV2", "KV3"
    """
    data = load("diem_uu_tien")
    ma   = ma_kv.upper()
    for item in data["uu_tien_khu_vuc"]:
        if item["ma"].upper() == ma:
            return ok(ma=item["ma"], ten=item["ten"], diem=item["diem"])
    return not_found(
        f"Không tìm thấy khu vực '{ma_kv}'. Mã hợp lệ: KV1, KV2-NT, KV2, KV3."
    )


def get_diem_uu_tien_doi_tuong(ma_doi_tuong: str) -> dict:
    """
    Lấy mức điểm ưu tiên đối tượng.

    Args:
        ma_doi_tuong: "01" → "06"
    """
    data = load("diem_uu_tien")
    for nhom in data["uu_tien_doi_tuong"]:
        if ma_doi_tuong in nhom["doi_tuong"]:
            return ok(
                nhom      = nhom["nhom"],
                doi_tuong = nhom["doi_tuong"],
                diem      = nhom["diem"],
            )
    return not_found(
        f"Không tìm thấy đối tượng '{ma_doi_tuong}'. Mã hợp lệ: 01–06."
    )


def tinh_diem_uu_tien(
    tong_diem  : float,
    khu_vuc    : str,
    doi_tuong  : str | None = None,
) -> dict:
    """
    Tính điểm ưu tiên và điểm xét tuyển cuối cùng.

    Công thức:
      - Nếu tổng_điểm < 22.5 → cộng thẳng điểm ưu tiên
      - Nếu tổng_điểm >= 22.5 → điểm ưu tiên = [(30 - tổng) / 7.5] × mức

    Args:
        tong_diem : Tổng điểm 3 môn thang 30
        khu_vuc   : "KV1", "KV2-NT", "KV2", "KV3"
        doi_tuong : "01"–"06" hoặc None

    Returns:
        {
            "tong_diem_goc", "diem_uu_tien_kv", "diem_uu_tien_dt",
            "tong_uu_tien", "diem_uu_tien_thuc", "diem_xet_tuyen", "ghi_chu"
        }
    """
    data   = load("diem_uu_tien")
    nguong = data["cong_thuc_giam_dan"]["nguong_ap_dung"]   # 22.5

    # Điểm ưu tiên khu vực
    kv = get_diem_uu_tien_khu_vuc(khu_vuc)
    if not kv["found"]:
        return kv
    diem_kv = kv["diem"]

    # Điểm ưu tiên đối tượng
    diem_dt = 0.0
    if doi_tuong:
        dt = get_diem_uu_tien_doi_tuong(doi_tuong)
        if dt["found"]:
            diem_dt = dt["diem"]

    tong_uu_tien = diem_kv + diem_dt

    # Áp dụng công thức giảm dần
    if tong_diem >= nguong and tong_uu_tien > 0:
        diem_uu_tien_thuc = round(((30 - tong_diem) / 7.5) * tong_uu_tien, 2)
        ghi_chu = (
            f"Tổng điểm {tong_diem} ≥ {nguong} → áp dụng công thức giảm dần: "
            f"[(30 - {tong_diem}) / 7.5] × {tong_uu_tien} = {diem_uu_tien_thuc}"
        )
    else:
        diem_uu_tien_thuc = tong_uu_tien
        ghi_chu = (
            f"Tổng điểm {tong_diem} < {nguong} → "
            f"cộng thẳng điểm ưu tiên {tong_uu_tien}"
        )

    return ok(
        tong_diem_goc     = tong_diem,
        khu_vuc           = khu_vuc,
        doi_tuong         = doi_tuong,
        diem_uu_tien_kv   = diem_kv,
        diem_uu_tien_dt   = diem_dt,
        tong_uu_tien      = tong_uu_tien,
        diem_uu_tien_thuc = diem_uu_tien_thuc,
        diem_xet_tuyen    = round(tong_diem + diem_uu_tien_thuc, 2),
        ghi_chu           = ghi_chu,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KIỂM TRA ĐẬU / TRƯỢT
# ═══════════════════════════════════════════════════════════════════════════════

def kiem_tra_dau_truot(
    ten_nganh   : str,
    diem_xet    : float,
    nam         : int = 2024,   # 2024 có đủ PT2/PT3/PT4/PT5, 2025 chỉ có "chung"
    phuong_thuc : str | None = None,
) -> dict:
    """
    So sánh điểm xét tuyển với điểm chuẩn → kết luận đậu/trượt.

    Args:
        ten_nganh   : Tên hoặc mã ngành
        diem_xet    : Điểm xét tuyển (đã bao gồm ưu tiên nếu có)
        nam         : Năm so sánh (mặc định 2025)
        phuong_thuc : PT cụ thể. None = so sánh với tất cả PT có dữ liệu

    Returns:
        {
            "found", "ten_nganh", "ma_nganh", "nam", "diem_xet",
            "ket_qua": [{
                "phuong_thuc", "phuong_thuc_ten",
                "diem_chuan", "chenh_lech", "nhan_xet"
            }]
        }
    """
    dc = get_diem_chuan(ten_nganh, nam=nam, phuong_thuc=phuong_thuc)
    if not dc["found"]:
        return dc

    ket_qua = []
    for item in dc["ket_qua"]:
        dc_val = item["diem_chuan"]
        diff   = round(diem_xet - dc_val, 2)

        if diff > 0.5:
            nhan_xet = "✅ Đậu"
        elif diff >= 0:
            nhan_xet = "⚠️ Sát nút — đậu nhưng rất gần điểm chuẩn"
        else:
            nhan_xet = f"❌ Trượt — thiếu {abs(diff)} điểm"

        ket_qua.append({
            "phuong_thuc"    : item["phuong_thuc"],
            "phuong_thuc_ten": item["phuong_thuc_ten"],
            "diem_chuan"     : dc_val,
            "chenh_lech"     : diff,
            "nhan_xet"       : nhan_xet,
        })

    return ok(
        ten_nganh = dc["ten_nganh"],
        ma_nganh  = dc["ma_nganh"],
        nam       = nam,
        diem_xet  = diem_xet,
        ket_qua   = ket_qua,
    )