"""
diem_chuan.py
Tra cứu điểm chuẩn theo ngành, năm, phương thức.

Nguồn dữ liệu: diem_chuan_2023_2024_2025.json
Cấu trúc mỗi record:
  {
    "ma_nganh", "ten_nganh", "nhom_nganh",
    "nam", "phuong_thuc_code", "phuong_thuc_ten",
    "diem_chuan", "thang_diem",
    "cac_phuong_thuc_ap_dung"  (chỉ có ở năm 2025)
  }
"""

from ._loader import load
from ._utils  import match_nganh, not_found, ok

# Map tên hiển thị cho từng phương thức
PT_NAME = {
    "PT2"  : "Học bạ / Chứng chỉ quốc tế",
    "PT3"  : "Thi THPT",
    "PT4"  : "Đánh giá năng lực (HSA - ĐHQG HN)",
    "PT5"  : "Đánh giá tư duy (TSA - ĐHBK HN)",
    "PT6"  : "Đánh giá tư duy (TSA cũ)",
    "chung": "Chung (áp dụng cho nhiều phương thức)",
}


def get_diem_chuan(
    ten_nganh   : str,
    nam         : int | None = None,
    phuong_thuc : str | None = None,
) -> dict:
    """
    Lấy điểm chuẩn của ngành.

    Args:
        ten_nganh   : Tên hoặc mã ngành.
        nam         : Năm xét tuyển (2023/2024/2025). None = tất cả.
        phuong_thuc : Mã PT ("PT3", "PT5"...). None = tất cả.

    Returns:
        {
            "found"    : True,
            "ten_nganh": str,
            "ma_nganh" : str,
            "ket_qua"  : [{
                "nam", "phuong_thuc", "phuong_thuc_ten",
                "diem_chuan", "thang_diem",
                "ap_dung_cho" (optional)
            }]
        }
    """
    data = load("diem_chuan")

    # Lọc theo tên / mã ngành
    results = [
        d for d in data
        if match_nganh(ten_nganh, d["ten_nganh"])
        or ten_nganh == d.get("ma_nganh", "")
    ]
    if not results:
        return not_found(f"Không tìm thấy ngành '{ten_nganh}' trong dữ liệu điểm chuẩn.")

    # Lọc theo năm
    if nam is not None:
        results = [d for d in results if d["nam"] == nam]
        if not results:
            return not_found(
                f"Không có dữ liệu điểm chuẩn năm {nam} cho ngành '{ten_nganh}'."
            )

    # Lọc theo phương thức
    if phuong_thuc:
        pt = phuong_thuc.upper()
        filtered = [
            d for d in results
            if d["phuong_thuc_code"].upper() == pt
            or pt in [p.upper() for p in d.get("cac_phuong_thuc_ap_dung", [])]
        ]
        if filtered:
            results = filtered

    # Format kết quả
    ten = results[0]["ten_nganh"]
    ma  = results[0]["ma_nganh"]

    ket_qua = []
    for d in sorted(results, key=lambda x: (x["nam"], x["phuong_thuc_code"])):
        item = {
            "nam"            : d["nam"],
            "phuong_thuc"    : d["phuong_thuc_code"],
            "phuong_thuc_ten": PT_NAME.get(d["phuong_thuc_code"], d["phuong_thuc_ten"]),
            "diem_chuan"     : d["diem_chuan"],
            "thang_diem"     : d.get("thang_diem", 30),
        }
        if "cac_phuong_thuc_ap_dung" in d:
            item["ap_dung_cho"] = d["cac_phuong_thuc_ap_dung"]
        ket_qua.append(item)

    return ok(ten_nganh=ten, ma_nganh=ma, ket_qua=ket_qua)


def get_diem_chuan_moi_nhat(ten_nganh: str) -> dict:
    """
    Lấy điểm chuẩn năm gần nhất hiện có của ngành.

    Args:
        ten_nganh: Tên hoặc mã ngành.
    """
    data = load("diem_chuan")
    results = [
        d for d in data
        if match_nganh(ten_nganh, d["ten_nganh"])
        or ten_nganh == d.get("ma_nganh", "")
    ]
    if not results:
        return not_found(f"Không tìm thấy ngành '{ten_nganh}'.")

    nam_max = max(d["nam"] for d in results)
    return get_diem_chuan(ten_nganh, nam=nam_max)


def get_lich_su_diem_chuan(ten_nganh: str) -> dict:
    """
    Lấy toàn bộ lịch sử điểm chuẩn các năm của ngành.
    Tiện dùng để phân tích xu hướng tăng/giảm.
    """
    return get_diem_chuan(ten_nganh)
