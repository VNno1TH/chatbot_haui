"""
formatter.py
Chuyển kết quả từ các module query_json thành chuỗi văn bản
để đưa vào context cho LLM.

Nguyên tắc: tất cả hàm nhận dict (output của query functions)
và trả về str. Không gọi trực tiếp JSON hay logic tính toán.
"""


# ── Điểm chuẩn ───────────────────────────────────────────────────────────────

def fmt_diem_chuan(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]

    ket_qua = result["ket_qua"]
    lines = [
        f"Điểm chuẩn ngành {result['ten_nganh']} "
        f"(mã {result['ma_nganh']}):"
    ]
    for r in ket_qua:
        line = (
            f"  - Năm {r['nam']} | {r['phuong_thuc_ten']}: "
            f"{r['diem_chuan']} điểm (thang {r['thang_diem']})"
        )
        if "ap_dung_cho" in r:
            line += f" — áp dụng cho {', '.join(r['ap_dung_cho'])}"
        lines.append(line)

    # Thêm phân tích xu hướng nếu có nhiều năm
    nam_list = sorted(set(r["nam"] for r in ket_qua))
    if len(nam_list) >= 2:
        lines.append("")
        lines.append("Xu hướng (PT3 - thi THPT):")
        pt3_data = sorted(
            [r for r in ket_qua if r["phuong_thuc"] in ("PT3", "chung")],
            key=lambda x: x["nam"]
        )
        if len(pt3_data) >= 2:
            for i in range(1, len(pt3_data)):
                prev, curr = pt3_data[i-1], pt3_data[i]
                diff = curr["diem_chuan"] - prev["diem_chuan"]
                arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
                lines.append(
                    f"  {prev['nam']} → {curr['nam']}: "
                    f"{prev['diem_chuan']} → {curr['diem_chuan']} "
                    f"({arrow} {abs(diff):.2f} điểm)"
                )
            # Nhận xét tổng thể
            first, last = pt3_data[0]["diem_chuan"], pt3_data[-1]["diem_chuan"]
            total = last - first
            if total > 0.5:
                lines.append(f"  → Xu hướng TĂNG tổng {total:.2f} điểm qua {len(pt3_data)} năm")
            elif total < -0.5:
                lines.append(f"  → Xu hướng GIẢM tổng {abs(total):.2f} điểm qua {len(pt3_data)} năm")
            else:
                lines.append(f"  → Điểm chuẩn ỔN ĐỊNH (dao động < 0.5 điểm)")
            # Ghi chú nếu 2025 dùng điểm chung
            if any(r["phuong_thuc"] == "chung" for r in pt3_data):
                lines.append(
                    "  ⚠ Lưu ý: Năm 2025 dùng điểm sàn chung (áp dụng cho nhiều PT), "
                    "không phải riêng PT3 — so sánh mang tính tham khảo."
                )

    return "\n".join(lines)


# ── Học phí ───────────────────────────────────────────────────────────────────

def fmt_hoc_phi(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]

    lines   = [f"Học phí HaUI năm học {result['nam_hoc']}:"]
    nhom_cur = ""
    for r in result["ket_qua"]:
        if r["nhom"] != nhom_cur:
            nhom_cur = r["nhom"]
            lines.append(f"\n  [{nhom_cur}]")
        lines.append(f"    - {r['chuong_trinh']}: {r['hien_thi']}")
    return "\n".join(lines)


# ── Chỉ tiêu ngành ────────────────────────────────────────────────────────────

def fmt_chi_tieu_nganh(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]

    blocks = []
    for r in result["ket_qua"]:
        blocks.append(
            f"Ngành {r['ten_nganh']} (mã {r['ma_nganh']}):\n"
            f"  - Chỉ tiêu        : {r['chi_tieu']} sinh viên\n"
            f"  - Tổ hợp xét tuyển: {', '.join(r['to_hop'])}\n"
            f"  - Phương thức     : {', '.join(r['phuong_thuc'])}"
        )
    return "\n\n".join(blocks)


# ── Ngành theo tổ hợp ─────────────────────────────────────────────────────────

def fmt_nganh_theo_to_hop(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]

    mon  = ", ".join(result["mon_thi"]) if result["mon_thi"] else "không rõ"
    lines = [
        f"Tổ hợp {result['ma_to_hop']} gồm môn: {mon}",
        f"Có {result['so_nganh']} ngành xét tuyển bằng tổ hợp này:",
    ]
    for i, n in enumerate(result["nganh_list"], 1):
        lines.append(
            f"  {i:2}. {n['ten_nganh']} (mã {n['ma_nganh']}) "
            f"— chỉ tiêu: {n['chi_tieu']}"
        )
    return "\n".join(lines)


# ── Chỉ tiêu 2026 ─────────────────────────────────────────────────────────────

def fmt_chi_tieu_2026(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]

    lines = [
        f"Chỉ tiêu tuyển sinh HaUI năm {result['nam']}:",
        f"  Tổng: {result['tong_chi_tieu']} sinh viên",
    ]
    for c in result["chi_tieu"]:
        lines.append(f"  - {c['he_dao_tao']}: {c['chi_tieu']}")
    if result.get("ghi_chu"):
        lines.append(f"\n  Lưu ý: {result['ghi_chu']}")
    return "\n".join(lines)


# ── Tổ hợp môn thi ────────────────────────────────────────────────────────────

def fmt_mon_thi_to_hop(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]
    return f"Tổ hợp {result['ma']} gồm: {', '.join(result['mon'])}."


# ── Điểm ưu tiên ─────────────────────────────────────────────────────────────

def fmt_tinh_diem_uu_tien(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]

    dt_str    = f", đối tượng {result['doi_tuong']}" if result["doi_tuong"] else ""
    diem_thuc = result["diem_uu_tien_thuc"]
    diem_xet  = result["diem_xet_tuyen"]
    tong_goc  = result["tong_diem_goc"]

    # Khi điểm ưu tiên thực = 0 (công thức giảm dần với điểm cao),
    # ghi rõ ràng để LLM không tự tính lại sai từ diem_uu_tien_kv
    if diem_thuc == 0:
        return (
            f"Tính điểm ưu tiên (Khu vực {result['khu_vuc']}{dt_str}):\n"
            f"  - Tổng điểm gốc               : {tong_goc}\n"
            f"  - Mức ưu tiên KV (danh nghĩa)  : +{result['diem_uu_tien_kv']} "
            f"— KHÔNG cộng, áp dụng công thức giảm dần\n"
            f"  - Điểm ưu tiên thực tế         : +0 điểm\n"
            f"  - Điểm xét tuyển               : {diem_xet} điểm\n"
            f"  ⚠ Điểm ưu tiên = 0 vì tổng điểm {tong_goc} đã ở mức tối đa (≥ 22.5).\n"
            f"  ({result['ghi_chu']})"
        )

    return (
        f"Tính điểm ưu tiên (Khu vực {result['khu_vuc']}{dt_str}):\n"
        f"  - Tổng điểm gốc    : {tong_goc}\n"
        f"  - Điểm ưu tiên KV  : +{result['diem_uu_tien_kv']}\n"
        f"  - Điểm ưu tiên ĐT  : +{result['diem_uu_tien_dt']}\n"
        f"  - Điểm ưu tiên thực: +{diem_thuc} (đã áp dụng công thức giảm dần)\n"
        f"  - Điểm xét tuyển   : {diem_xet} điểm\n"
        f"  ({result['ghi_chu']})"
    )


# ── Quy đổi điểm ─────────────────────────────────────────────────────────────

def fmt_quy_doi(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]
    return (
        f"Quy đổi điểm {result['ten']}:\n"
        f"  - Điểm gốc    : {result['diem_goc']} / {result['thang_goc']}\n"
        f"  - Điểm quy đổi: {result['diem_quy_doi']} / {result['thang_quy_doi']}"
    )


# ── Đậu / trượt ───────────────────────────────────────────────────────────────

def fmt_kiem_tra_dau_truot(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]

    lines = [
        f"Kiểm tra xét tuyển ngành {result['ten_nganh']} "
        f"(mã {result['ma_nganh']}) — năm {result['nam']}:",
        f"  Điểm xét tuyển của bạn: {result['diem_xet']}",
    ]
    for k in result["ket_qua"]:
        lines.append(
            f"  - {k['phuong_thuc']} ({k['phuong_thuc_ten']}): "
            f"điểm chuẩn {k['diem_chuan']} → {k['nhan_xet']}"
        )
    return "\n".join(lines)