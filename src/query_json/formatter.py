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

    lines   = [
        f"Học phí HaUI năm học {result['nam_hoc']}:",
        f"  ⚠ LƯU Ý: Đơn vị học phí là đồng/tín chỉ (KHÔNG phải /năm hay /học kỳ).",
        f"  Mỗi năm học khoảng 30-35 tín chỉ → học phí thực tế = (số tín chỉ) × (mức/tín chỉ).",
        "",
    ]
    nhom_cur = ""
    for r in result["ket_qua"]:
        if r["nhom"] != nhom_cur:
            nhom_cur = r["nhom"]
            lines.append(f"  [{nhom_cur}]")
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

# ── Ngành theo khoa ───────────────────────────────────────────────────────────

def fmt_nganh_theo_khoa_v2(result: dict, diem_xet: float | None = None) -> str:
    """
    Format danh sách ngành theo khoa, kèm so sánh điểm chuẩn nếu có diem_xet.
    Dùng khi đã có dữ liệu điểm chuẩn kèm theo từng ngành.

    Args:
        result   : Output của get_nganh_theo_khoa()
        diem_xet : Điểm của thí sinh (tùy chọn) để so sánh trực tiếp
    """
    if not result["found"]:
        return result["thong_bao"]

    lines = [
        f"Trường/Khoa {result['ten_khoa']} có {result['so_nganh']} ngành:",
    ]
    if diem_xet is not None:
        lines[0] += f" (so sánh với điểm của bạn: {diem_xet})"

    lines.append("")

    for n in result["nganh_list"]:
        # Lấy điểm chuẩn tốt nhất để hiển thị (ưu tiên PT3 2025, fallback chung 2025, rồi 2024)
        dc_2025 = n.get("diem_chuan_pt3_2025") or n.get("diem_chuan_chung_2025")
        dc_2024 = n.get("diem_chuan_pt3_2024")

        dc_str = ""
        nhan_xet = ""
        if dc_2025 is not None:
            dc_str = f"{dc_2025} điểm (2025)"
            if diem_xet is not None:
                diff = round(diem_xet - dc_2025, 2)
                if diff > 0.5:
                    nhan_xet = f" → Điểm bạn cao hơn {diff} điểm"
                elif diff >= 0:
                    nhan_xet = f" → Sát nút (+{diff})"
                else:
                    nhan_xet = f" → Thiếu {abs(diff)} điểm"
        elif dc_2024 is not None:
            dc_str = f"{dc_2024} điểm (2024, chưa có 2025)"

        ct_str = f", chỉ tiêu {n['chi_tieu']}" if n.get("chi_tieu") else ""
        to_hop_str = f", tổ hợp: {', '.join(n['to_hop'][:3])}" if n.get("to_hop") else ""

        lines.append(
            f"  • {n['ten_nganh']} (mã {n['ma_nganh']}){ct_str}"
        )
        if dc_str:
            lines.append(f"    Điểm chuẩn PT3: {dc_str}{nhan_xet}")
        if to_hop_str:
            lines.append(f"    Tổ hợp: {', '.join(n['to_hop'][:4])}")

    lines.append("")
    lines.append(
        "Lưu ý: Điểm chuẩn 2026 chưa công bố — số liệu trên chỉ mang tính tham khảo."
    )
    return "\n".join(lines)


# ── Ngành theo khoa ───────────────────────────────────────────────────────────

def fmt_nganh_theo_khoa(result: dict) -> str:
    if not result["found"]:
        return result["thong_bao"]
    lines = [
        f"Trường/Khoa {result['ten_khoa']} có {result['so_nganh']} ngành:",
    ]
    for i, n in enumerate(result["nganh_list"], 1):
        lines.append(f"  {i:2}. {n['ten_nganh']} (mã {n['ma_nganh']})")
    return "\n".join(lines)


# ── Điểm chuẩn theo khoa ─────────────────────────────────────────────────────

def fmt_diem_chuan_theo_khoa(result: dict, diem_user: float | None = None) -> str:
    if not result["found"]:
        return result["thong_bao"]

    lines = [
        f"Điểm chuẩn năm {result['nam']} — Trường {result['ten_khoa']} "
        f"({result['so_nganh']} ngành):",
    ]

    for r in result["ket_qua"]:
        if diem_user is not None:
            chenh = round(diem_user - r["diem_chuan"], 2)
            if chenh > 0.5:
                nhan = f"✅ +{chenh} điểm"
            elif chenh >= 0:
                nhan = f"⚠️ sát nút (+{chenh})"
            else:
                nhan = f"❌ thiếu {abs(chenh)} điểm"
            lines.append(
                f"  - {r['ten_nganh']}: {r['diem_chuan']} điểm → {nhan}"
            )
        else:
            lines.append(
                f"  - {r['ten_nganh']} (mã {r['ma_nganh']}): "
                f"{r['diem_chuan']} điểm"
            )

    if diem_user is not None:
        lines.append(
            f"\nĐiểm tham khảo trên là năm {result['nam']}. "
            f"Điểm chuẩn 2026 chưa công bố — "
            f"theo dõi tại tuyensinh.haui.edu.vn."
        )
    return "\n".join(lines)