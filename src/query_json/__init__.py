"""
src/query_json/__init__.py
Public API của package query_json.
Import từ đây thay vì import trực tiếp từng module con.

Cách dùng:
    from src.query_json import get_diem_chuan, fmt_diem_chuan
    from src.query_json import tinh_diem_uu_tien, fmt_tinh_diem_uu_tien
"""

# Thông tin ngành
from .nganh import (
    get_chi_tieu_nganh,
    get_nganh_theo_to_hop,
    get_chi_tieu_tong_2026,
    get_mon_thi_to_hop,
    get_tat_ca_to_hop,
)

# Điểm chuẩn
from .diem_chuan import (
    get_diem_chuan,
    get_diem_chuan_moi_nhat,
    get_lich_su_diem_chuan,
)

# Học phí
from .hoc_phi import (
    get_hoc_phi,
    get_hoc_phi_dai_hoc,
    get_hoc_phi_sau_dai_hoc,
)

# Điểm xét tuyển (quy đổi + ưu tiên + đậu/trượt)
from .diem_xet_tuyen import (
    quy_doi_HSA,
    quy_doi_TSA,
    quy_doi_KQHB,
    get_diem_uu_tien_khu_vuc,
    get_diem_uu_tien_doi_tuong,
    tinh_diem_uu_tien,
    kiem_tra_dau_truot,
)

# Formatter — stringify cho LLM
from .formatter import (
    fmt_diem_chuan,
    fmt_hoc_phi,
    fmt_chi_tieu_nganh,
    fmt_nganh_theo_to_hop,
    fmt_chi_tieu_2026,
    fmt_mon_thi_to_hop,
    fmt_tinh_diem_uu_tien,
    fmt_quy_doi,
    fmt_kiem_tra_dau_truot,
)

__all__ = [
    "get_chi_tieu_nganh", "get_nganh_theo_to_hop",
    "get_chi_tieu_tong_2026", "get_mon_thi_to_hop", "get_tat_ca_to_hop",
    "get_diem_chuan", "get_diem_chuan_moi_nhat", "get_lich_su_diem_chuan",
    "get_hoc_phi", "get_hoc_phi_dai_hoc", "get_hoc_phi_sau_dai_hoc",
    "quy_doi_HSA", "quy_doi_TSA", "quy_doi_KQHB",
    "get_diem_uu_tien_khu_vuc", "get_diem_uu_tien_doi_tuong",
    "tinh_diem_uu_tien", "kiem_tra_dau_truot",
    "fmt_diem_chuan", "fmt_hoc_phi", "fmt_chi_tieu_nganh",
    "fmt_nganh_theo_to_hop", "fmt_chi_tieu_2026", "fmt_mon_thi_to_hop",
    "fmt_tinh_diem_uu_tien", "fmt_quy_doi", "fmt_kiem_tra_dau_truot",
]
