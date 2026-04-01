"""
run_chatbot_tests_v2.py
=======================
Bộ 95 câu hỏi kiểm tra chatbot HaUI — phiên bản có auto-scoring.

Cải tiến so với v1:
- AUTO-SCORING: mỗi TestCase có `check_keywords` (phải có) và `fail_keywords`
  (không được có). Runner tự chấm PASS/PARTIAL/FAIL mà không cần Claude đọc.
- DIAGNOSIS: mỗi FAIL/PARTIAL tự gán `diagnosis_code` để chỉ thẳng vào code
  cần sửa (vd: "RETRIEVAL_MISS", "CALC_ERROR", "HALLUCINATION", ...).
- REPORT tập trung vào "sửa chỗ nào trong code" thay vì "đọc từng câu".
  File output chứa bảng tổng hợp theo diagnosis_code và file/function liên quan.

Cách dùng:
    python run_chatbot_tests_v2.py                  # chạy tất cả
    python run_chatbot_tests_v2.py --groups 1 2     # chỉ nhóm 1,2
    python run_chatbot_tests_v2.py --ids 34 35      # chỉ câu 34,35
    python run_chatbot_tests_v2.py --dry-run        # in danh sách, không gọi bot
    python run_chatbot_tests_v2.py --delay 2.0      # nghỉ 2s giữa câu
    python run_chatbot_tests_v2.py --output my_rep  # tên file (không cần .md)
    python run_chatbot_tests_v2.py --score-only     # chỉ chạm điểm, không gọi bot
                                                    # (cần file JSON từ lần chạy trước)
    python run_chatbot_tests_v2.py --rescore my_rep.json  # re-score JSON cũ

Output:
    <name>.md   — báo cáo cho Claude đọc để biết sửa chỗ nào trong code
    <name>.json — raw data để phân tích thêm
"""

from __future__ import annotations

import sys
import os
import re
import json
import time
import argparse
import logging
import traceback
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_runner_v2")


# ══════════════════════════════════════════════════════════════════════════════
# 0. DIAGNOSIS CODES — mỗi code tương ứng với một loại bug/vấn đề trong code
# ══════════════════════════════════════════════════════════════════════════════
#
# Khi thêm câu test mới, chọn diagnosis_code phù hợp nhất từ bảng sau:
#
# RETRIEVAL_MISS      — Bot không tìm được chunk dữ liệu đúng (retriever/vector DB)
# RETRIEVAL_WRONG     — Bot tìm được nhưng chunk sai / không liên quan
# CALC_ERROR          — Bot tính sai (điểm ưu tiên, quy đổi, học phí...)
# HALLUCINATION       — Bot bịa ra thông tin không có trong dữ liệu
# OUT_OF_SCOPE        — Bot trả lời thay vì từ chối câu hỏi ngoài dữ liệu
# INCOMPLETE          — Bot trả lời đúng nhưng thiếu thông tin cần thiết
# WRONG_YEAR          — Bot nhầm năm (ví dụ 2024 vs 2025)
# WRONG_METHOD        — Bot nhầm phương thức tuyển sinh (PT2 vs PT3...)
# AMBIGUITY_FAIL      — Bot không xử lý tốt câu hỏi mơ hồ / đa nghĩa
# MULTI_STEP_FAIL     — Bot thất bại ở câu hỏi cần nhiều bước suy luận
# COMPARISON_FAIL     — Bot không so sánh/liệt kê được khi cần
# SLOW                — Bot trả lời đúng nhưng quá chậm (>threshold giây)

DIAGNOSIS_THRESHOLDS = {
    "SLOW": 10.0,  # giây — cấu hình ngưỡng tốc độ ở đây
}

# Map diagnosis_code → file thực tế trong project
# Cấu trúc thực: tất cả file nằm cùng cấp (không có src/pipeline/ hay src/retrieval/)
#   retriever.py      — hybrid retrieval (Vector+BM25+RRF+Reranker+HyDE+QueryRewriter)
#   router.py         — intent classification (rule → LLM → embedding fallback)
#   chatbot.py        — pipeline chính, ContextBuilder, JSON handlers
#   chatbot_patch.py  — v8 patches: entity extractor v3, calc fixes, sanitizer
#   nganh.py          — alias khoa/trường, get_nganh_theo_khoa, _resolve_khoa
#   diem_chuan.py     — get_diem_chuan, get_diem_chuan_theo_khoa
#   diem_xet_tuyen.py — quy_doi_HSA/TSA/KQHB, tinh_diem_uu_tien, kiem_tra_dau_truot
#   hoc_phi.py        — get_hoc_phi
#   formatter.py      — fmt_* functions
#   embedder.py       — ChromaDB + embedding backend
#   chunker.py        — MarkdownChunker, chunk strategy
#   build_index.py    — rebuild vector index
DIAGNOSIS_TO_COMPONENT: dict[str, str] = {
    "RETRIEVAL_MISS"  : "retriever.py (QueryRewriter/HyDE/BM25) + embedder.py/chunker.py (vector DB)",
    "RETRIEVAL_WRONG" : "retriever.py — RRF/reranker scoring, ChunkScorer.compute_bonus()",
    "CALC_ERROR"      : "chatbot_patch.py — _tinh_diem_uu_tien_v8(), quy_doi_HSA/TSA_fixed(), tinh_diem_PT2()",
    "HALLUCINATION"   : "chatbot.py — SYSTEM_PROMPT / chatbot_patch.py — SYSTEM_PROMPT override",
    "OUT_OF_SCOPE"    : "router.py — _fast_path() / LLMClassifier, chatbot.py — _OFF_TOPIC_REPLY",
    "INCOMPLETE"      : "retriever.py — FINAL_TOP_K, chatbot.py — ContextCompressor.compress()",
    "WRONG_YEAR"      : "diem_chuan.py — get_diem_chuan(nam=), chatbot_patch.py — _ctx_diem_chuan_v8()",
    "WRONG_METHOD"    : "diem_chuan.py — phuong_thuc filter, chatbot_patch.py — _ctx_diem_chuan_v8() PT map",
    "AMBIGUITY_FAIL"  : "router.py — LLMClassifier / _CLASSIFY_SYSTEM prompt, chatbot.py — EntityTracker",
    "MULTI_STEP_FAIL" : "chatbot_patch.py — _ctx_dau_truot_v8(), _ctx_quy_doi_diem_v8(), tinh_diem_PT2()",
    "COMPARISON_FAIL" : "retriever.py — FINAL_TOP_K/multi-doc, chatbot.py — ContextBuilder._ctx_nganh_theo_khoa()",
    "SLOW"            : "chatbot.py — ResponseCache, retriever.py — _retrieve_cache, LLM_TIMEOUT",
    "UNKNOWN"         : "(chưa xác định — xem answer + context_preview để phân loại)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEST CASE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    id               : int
    group            : int
    group_name       : str
    difficulty       : str          # "dễ" | "trung bình" | "khó" | "rất khó"
    question         : str
    hint             : str          # Dữ liệu đúng để so sánh
    tags             : list[str]
    expected_behavior: str
    # ── Auto-scoring ──────────────────────────────────────────────────────────
    check_keywords   : list[str]    # TẤT CẢ phải xuất hiện trong câu trả lời → PASS
    #                                 (so sánh lowercase, loại bỏ dấu nếu check_strip_accents=True)
    any_keywords     : list[str]    # ÍT NHẤT 1 phải xuất hiện → PASS (dùng khi check_keywords trống)
    fail_keywords    : list[str]    # NẾU có bất kỳ cái nào → FAIL (hallucination guard)
    check_strip_accents: bool = True  # True = so sánh không phân biệt dấu tiếng Việt
    # ── Diagnosis ─────────────────────────────────────────────────────────────
    diagnosis_on_fail: str = "RETRIEVAL_MISS"  # code lý do khi FAIL/PARTIAL
    diagnosis_note   : str = ""  # ghi chú: tại sao chọn code này, khi nào đổi code khác


# ── Utility: bỏ dấu tiếng Việt để so sánh ────────────────────────────────────
def _strip_accents(text: str) -> str:
    """Chuyển 'ngôn ngữ Anh' → 'ngon ngu anh' để so sánh dễ hơn."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _normalize(text: str, strip: bool = True) -> str:
    t = text.lower()
    return _strip_accents(t) if strip else t


def auto_score(tc: TestCase, answer: str) -> tuple[str, str]:
    """
    Trả về (score, diagnosis_code).
    score: "PASS" | "PARTIAL" | "FAIL"
    """
    if not answer or answer.strip() == "":
        return "FAIL", tc.diagnosis_on_fail

    strip = tc.check_strip_accents
    ans_n = _normalize(answer, strip)

    # 1. Kiểm tra fail_keywords (hallucination / sai thông tin)
    for kw in tc.fail_keywords:
        if _normalize(kw, strip) in ans_n:
            return "FAIL", "HALLUCINATION"

    # 2. Kiểm tra check_keywords (tất cả phải có)
    if tc.check_keywords:
        found_all  = all(_normalize(kw, strip) in ans_n for kw in tc.check_keywords)
        found_some = any(_normalize(kw, strip) in ans_n for kw in tc.check_keywords)
        if found_all:
            return "PASS", ""
        elif found_some:
            return "PARTIAL", tc.diagnosis_on_fail
        else:
            return "FAIL", tc.diagnosis_on_fail

    # 3. Fallback: any_keywords
    if tc.any_keywords:
        if any(_normalize(kw, strip) in ans_n for kw in tc.any_keywords):
            return "PASS", ""
        return "FAIL", tc.diagnosis_on_fail

    # 4. Không có keyword nào → không thể tự chấm
    return "UNKNOWN", "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEST CASES (95 câu)
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES: list[TestCase] = [

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 1 — Thông tin cơ bản (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=1, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="HaUI là viết tắt của trường nào? Trụ sở chính ở đâu?",
        hint="Hanoi University of Industry — Đại học Công nghiệp Hà Nội. Số 298, Đường Cầu Diễn, Phường Minh Khai, Quận Bắc Từ Liêm, Hà Nội.",
        tags=["gioi_thieu", "co_ban"],
        expected_behavior="Trả lời đúng tên đầy đủ tiếng Việt + tiếng Anh và địa chỉ chính xác.",
        check_keywords=["Công nghiệp Hà Nội", "Cầu Diễn"],
        any_keywords=[],
        fail_keywords=["Bách Khoa", "Kinh tế Quốc dân"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=2, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Năm 2026, HaUI tuyển bao nhiêu chỉ tiêu đại học chính quy?",
        hint="8.300 chỉ tiêu đại học chính quy năm 2026 (tổng tất cả hệ là 9.420).",
        tags=["chi_tieu", "nam_2026"],
        expected_behavior="Nêu đúng 8.300 chỉ tiêu ĐH chính quy, không nhầm với tổng 9.420.",
        check_keywords=[],
        any_keywords=["8.300", "8300"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=3, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Ngành Công nghệ thông tin mã 7480201 có bao nhiêu chỉ tiêu năm 2025?",
        hint="360 chỉ tiêu năm 2025.",
        tags=["chi_tieu", "CNTT", "nam_2025"],
        expected_behavior="Trả lời đúng 360 chỉ tiêu.",
        check_keywords=["360"],
        any_keywords=[],
        fail_keywords=["300", "400", "240"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=4, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Ngành Kế toán xét những tổ hợp môn nào?",
        hint="A01 (Toán, Vật lí, Tiếng Anh), D01 (Ngữ văn, Toán, Tiếng Anh), D0G (Toán, Tiếng Anh, GDKT pháp luật).",
        tags=["to_hop", "ke_toan"],
        expected_behavior="Liệt kê đủ 3 tổ hợp A01, D01, D0G với môn thi tương ứng.",
        check_keywords=["A01", "D01", "D0G"],
        any_keywords=[],
        fail_keywords=["A00", "B00"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=5, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Tổ hợp A00 gồm những môn gì?",
        hint="Toán, Vật lý, Hóa học.",
        tags=["to_hop"],
        expected_behavior="Liệt kê đúng 3 môn: Toán, Vật lý, Hóa học.",
        check_keywords=["Toán", "Vật lý", "Hóa học"],
        any_keywords=[],
        fail_keywords=["Tiếng Anh", "Ngữ văn"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=6, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Tổ hợp D07 gồm những môn gì?",
        hint="Toán, Hóa học, Tiếng Anh.",
        tags=["to_hop"],
        expected_behavior="Liệt kê đúng 3 môn: Toán, Hóa học, Tiếng Anh.",
        check_keywords=["Toán", "Hóa học", "Tiếng Anh"],
        any_keywords=[],
        fail_keywords=["Vật lý", "Ngữ văn", "Sinh học"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=7, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Ngành Thiết kế thời trang thuộc khoa/trường nào?",
        hint="Khoa Công nghệ May và Thiết kế Thời trang.",
        tags=["nganh", "khoa"],
        expected_behavior="Nêu đúng tên khoa: Khoa Công nghệ May và Thiết kế Thời trang.",
        check_keywords=["May", "Thiết kế Thời trang"],
        any_keywords=[],
        fail_keywords=["Điện", "Cơ khí", "Kinh tế"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=8, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Ngành Robot và trí tuệ nhân tạo có mã ngành là gì?",
        hint="Mã ngành 75102032.",
        tags=["ma_nganh"],
        expected_behavior="Trả lời đúng mã 75102032.",
        check_keywords=["75102032"],
        any_keywords=[],
        fail_keywords=["7510209", "75102033"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=9, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="HaUI có bao nhiêu cơ sở đào tạo? Ở đâu?",
        hint="3 cơ sở: CS1 tại Minh Khai (Bắc Từ Liêm, HN), CS2 tại Tây Tựu (Bắc Từ Liêm, HN), CS3 tại Phủ Lý (Hà Nam).",
        tags=["co_so", "gioi_thieu"],
        expected_behavior="Nêu đủ 3 cơ sở với địa chỉ cụ thể.",
        check_keywords=["3", "Minh Khai", "Phủ Lý"],
        any_keywords=[],
        fail_keywords=["2 cơ sở", "4 cơ sở"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=10, group=1, group_name="Thông tin cơ bản", difficulty="dễ",
        question="Lệ phí đăng ký xét tuyển năm 2025 là bao nhiêu?",
        hint="50.000 đồng/hồ sơ, nộp qua mã QR ngân hàng.",
        tags=["le_phi", "dang_ky"],
        expected_behavior="Nêu đúng 50.000 đồng và hình thức nộp qua QR.",
        check_keywords=["50.000", "50000"],
        any_keywords=[],
        fail_keywords=["100.000", "200.000", "30.000"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 2 — Điểm chuẩn (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=11, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Điểm chuẩn ngành Marketing năm 2025 là bao nhiêu?",
        hint="22.5 điểm (áp dụng chung PT2, PT3, PT4 năm 2025, thang 30).",
        tags=["diem_chuan", "marketing", "nam_2025"],
        expected_behavior="Trả lời đúng 22.5 điểm năm 2025.",
        check_keywords=["22.5", "22,5"],
        any_keywords=[],
        fail_keywords=["25.33", "25.24", "28.55"],
        diagnosis_on_fail="WRONG_YEAR",
    ),
    TestCase(
        id=12, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Ngành Logistics năm 2024 xét học bạ (PT4) điểm chuẩn là bao nhiêu?",
        hint="28.91 điểm (PT4 năm 2024).",
        tags=["diem_chuan", "logistics", "nam_2024", "hoc_ba"],
        expected_behavior="Trả lời đúng 28.91 điểm.",
        check_keywords=["28.91"],
        any_keywords=[],
        fail_keywords=["28.6", "25.89", "27.0"],
        diagnosis_on_fail="WRONG_METHOD",
    ),
    TestCase(
        id=13, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="So sánh điểm chuẩn ngành Công nghệ thông tin qua 3 năm 2023, 2024, 2025 theo phương thức thi THPT.",
        hint="2023: 25.19 (PT3), 2024: 25.22 (PT3), 2025: 23.09 (chung PT2/PT3/PT5).",
        tags=["diem_chuan", "CNTT", "so_sanh", "nhieu_nam"],
        expected_behavior="Liệt kê đủ 3 năm với điểm đúng.",
        check_keywords=["25.19", "25.22", "23.09"],
        any_keywords=[],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=14, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Năm 2025, ngành nào có điểm chuẩn cao nhất trong khối kỹ thuật?",
        hint="Công nghệ kỹ thuật điều khiển và tự động hóa: 26.27 điểm.",
        tags=["diem_chuan", "ky_thuat", "cao_nhat", "nam_2025"],
        expected_behavior="Xác định đúng ngành Điều khiển tự động hóa với điểm 26.27.",
        check_keywords=["26.27"],
        any_keywords=["tự động hóa", "điều khiển"],
        fail_keywords=["25.17", "24.3"],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=15, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Điểm chuẩn ngành Hóa dược năm 2023 theo phương thức thi THPT là bao nhiêu?",
        hint="19.45 điểm (PT3 năm 2023).",
        tags=["diem_chuan", "hoa_duoc", "nam_2023"],
        expected_behavior="Trả lời đúng 19.45 điểm.",
        check_keywords=["19.45"],
        any_keywords=[],
        fail_keywords=["21.55", "19.0"],
        diagnosis_on_fail="WRONG_YEAR",
    ),
    TestCase(
        id=16, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Năm 2024, ngành Công nghệ kỹ thuật điều khiển và tự động hóa điểm chuẩn PT3 là bao nhiêu?",
        hint="26.05 điểm (PT3 năm 2024).",
        tags=["diem_chuan", "tu_dong_hoa", "nam_2024"],
        expected_behavior="Trả lời đúng 26.05 điểm.",
        check_keywords=["26.05"],
        any_keywords=[],
        fail_keywords=["26.27", "25.47"],
        diagnosis_on_fail="WRONG_YEAR",
    ),
    TestCase(
        id=17, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Ngành Ngôn ngữ Trung Quốc chương trình liên kết 2+2 năm 2025 điểm chuẩn là bao nhiêu?",
        hint="22.5 điểm (năm 2025, mã 7220204LK).",
        tags=["diem_chuan", "ngon_ngu_trung", "lien_ket", "nam_2025"],
        expected_behavior="Nêu đúng 22.5 điểm và phân biệt với chương trình thường (23.0).",
        check_keywords=["22.5"],
        any_keywords=[],
        fail_keywords=["23.0", "24.91"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=18, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Năm 2025, nhóm ngành Du lịch – Khách sạn – Nhà hàng, ngành nào có điểm chuẩn thấp nhất?",
        hint="Du lịch tiếng Anh: 18.0; Quản trị dịch vụ du lịch tiếng Anh: 18.1; Quản trị khách sạn tiếng Anh: 18.25.",
        tags=["diem_chuan", "du_lich", "thap_nhat", "nam_2025"],
        expected_behavior="Xác định đúng Du lịch chương trình tiếng Anh với điểm 18.0 là thấp nhất.",
        check_keywords=["18.0"],
        any_keywords=["Du lịch", "tiếng Anh"],
        fail_keywords=["18.25", "18.1"],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=19, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Điểm chuẩn ngành Kỹ thuật phần mềm năm 2023 theo phương thức chứng chỉ HSG (PT2) là bao nhiêu?",
        hint="28.45 điểm (PT2 năm 2023).",
        tags=["diem_chuan", "phan_mem", "nam_2023", "PT2"],
        expected_behavior="Trả lời đúng 28.45 điểm.",
        check_keywords=["28.45"],
        any_keywords=[],
        fail_keywords=["24.54", "24.68", "21.75"],
        diagnosis_on_fail="WRONG_METHOD",
    ),
    TestCase(
        id=20, group=2, group_name="Điểm chuẩn", difficulty="trung bình",
        question="Ngành Công nghệ kỹ thuật môi trường năm 2024 điểm chuẩn PT3 là bao nhiêu?",
        hint="19.0 điểm (PT3 năm 2024).",
        tags=["diem_chuan", "moi_truong", "nam_2024"],
        expected_behavior="Trả lời đúng 19.0 điểm.",
        check_keywords=["19.0", "19,0"],
        any_keywords=[],
        fail_keywords=["18.75", "20.0"],
        diagnosis_on_fail="WRONG_YEAR",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 3 — Phương thức tuyển sinh (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=21, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="HaUI có bao nhiêu phương thức tuyển sinh năm 2025? Liệt kê tên từng phương thức.",
        hint="5 phương thức: PT1 (xét thẳng), PT2 (HSG/chứng chỉ quốc tế), PT3 (thi THPT), PT4 (ĐGNL ĐHQG HN), PT5 (ĐGTD ĐHBK HN).",
        tags=["phuong_thuc", "tong_quat"],
        expected_behavior="Liệt kê đủ 5 phương thức với mô tả ngắn.",
        check_keywords=["PT1", "PT2", "PT3", "PT4", "PT5"],
        any_keywords=[],
        fail_keywords=[],
        diagnosis_on_fail="INCOMPLETE",
    ),
    TestCase(
        id=22, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Phương thức 2 dành cho đối tượng nào? Điều kiện cụ thể là gì?",
        hint="Thí sinh đoạt giải HSG cấp tỉnh/TP HOẶC có chứng chỉ quốc tế; điểm TB môn tổ hợp lớp 10,11,12 ≥ 7.0.",
        tags=["phuong_thuc", "PT2", "dieu_kien"],
        expected_behavior="Nêu rõ 2 điều kiện: (1) giải HSG hoặc chứng chỉ QT và (2) TB môn ≥ 7.0.",
        check_keywords=["7.0", "HSG"],
        any_keywords=["chứng chỉ", "giải"],
        fail_keywords=[],
        diagnosis_on_fail="INCOMPLETE",
    ),
    TestCase(
        id=23, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Chứng chỉ IELTS bao nhiêu điểm thì đủ điều kiện xét tuyển theo PT2?",
        hint="IELTS Academic ≥ 5.5.",
        tags=["PT2", "IELTS", "chung_chi"],
        expected_behavior="Nêu đúng ngưỡng tối thiểu IELTS 5.5.",
        check_keywords=["5.5"],
        any_keywords=[],
        fail_keywords=["6.0", "6.5", "7.0"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=24, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Tôi có chứng chỉ TOPIK cấp 3, có đủ điều kiện xét PT2 không?",
        hint="TOPIK cấp 3 đủ điều kiện xét PT2 thường (không đủ cho LK 2+2 cần cấp 4).",
        tags=["PT2", "TOPIK", "dieu_kien"],
        expected_behavior="Xác nhận đủ điều kiện với PT2 thường, lưu ý LK 2+2 cần cấp 4.",
        check_keywords=["đủ"],
        any_keywords=["TOPIK 3", "cấp 3"],
        fail_keywords=["không đủ", "không được"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=25, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Công thức tính điểm xét tuyển theo phương thức 2 là gì?",
        hint="ĐXT = ĐKQHT × 2 + ĐQĐCC + Điểm ưu tiên.",
        tags=["PT2", "cong_thuc", "diem_xet_tuyen"],
        expected_behavior="Trình bày đúng công thức với nhân 2 cho ĐKQHT.",
        check_keywords=["× 2", "x 2", "*2", "nhân 2"],
        any_keywords=["ĐKQHT", "kết quả học tập"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=26, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Thí sinh tốt nghiệp THPT năm 2024 (thí sinh tự do) có được đăng ký PT4, PT5 không?",
        hint="Không. Thí sinh tốt nghiệp trước 2025 không được đăng ký PT2, PT4, PT5.",
        tags=["PT4", "PT5", "thi_sinh_tu_do"],
        expected_behavior="Trả lời rõ là KHÔNG.",
        check_keywords=["không"],
        any_keywords=[],
        fail_keywords=["được", "có thể"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=27, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Ngành Ngôn ngữ Anh có xét theo PT4 (ĐGNL) không?",
        hint="Ngành Ngôn ngữ Anh chỉ xét PT1, PT2, PT3 — không có PT4.",
        tags=["PT4", "ngon_ngu_anh", "phuong_thuc"],
        expected_behavior="Xác nhận đúng là Ngôn ngữ Anh KHÔNG xét PT4.",
        check_keywords=["không"],
        any_keywords=[],
        fail_keywords=["có xét", "được xét"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=28, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Phương thức 5 dùng kết quả thi của trường nào?",
        hint="Đại học Bách Khoa Hà Nội (ĐHBK HN) — kỳ thi Đánh giá tư duy (ĐGTD/TSA).",
        tags=["PT5", "DGTD", "Bach_Khoa"],
        expected_behavior="Nêu đúng ĐHBK Hà Nội.",
        check_keywords=["Bách Khoa"],
        any_keywords=["ĐHBK", "tư duy"],
        fail_keywords=["ĐHQG", "Quốc gia"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=29, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Thời gian đăng ký dự tuyển PT2, PT4, PT5 năm 2025 là từ ngày nào đến ngày nào?",
        hint="15/5/2025 đến 05/7/2025 tại xettuyen.haui.edu.vn.",
        tags=["lich_dang_ky", "PT2"],
        expected_behavior="Nêu đúng 15/5 – 05/7/2025.",
        check_keywords=["15/5", "05/7"],
        any_keywords=[],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=30, group=3, group_name="Phương thức tuyển sinh", difficulty="trung bình",
        question="Nếu muốn xét tuyển tổ hợp có môn Tiếng Anh mà có chứng chỉ IELTS, có được quy đổi thay điểm thi THPT không?",
        hint="KHÔNG. HaUI không quy đổi chứng chỉ ngoại ngữ thay điểm thi THPT trong PT3.",
        tags=["IELTS", "quy_doi", "PT3"],
        expected_behavior="Trả lời rõ KHÔNG.",
        check_keywords=["không"],
        any_keywords=[],
        fail_keywords=["được quy đổi", "có thể thay"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 4 — Điểm ưu tiên (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=31, group=4, group_name="Điểm ưu tiên", difficulty="trung bình",
        question="Thí sinh ở khu vực nông thôn (KV2-NT) được cộng bao nhiêu điểm ưu tiên?",
        hint="KV2-NT: +0.50 điểm.",
        tags=["uu_tien", "khu_vuc", "KV2-NT"],
        expected_behavior="Trả lời đúng +0.50 điểm.",
        check_keywords=["0.5", "0,5"],
        any_keywords=[],
        fail_keywords=["0.75", "0.25", "1.0"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=32, group=4, group_name="Điểm ưu tiên", difficulty="trung bình",
        question="Thí sinh là con thương binh suy giảm lao động 85% được cộng bao nhiêu điểm ưu tiên đối tượng?",
        hint="Con thương binh suy giảm ≥81% thuộc nhóm UT1, đối tượng 03: +2.0 điểm.",
        tags=["uu_tien", "doi_tuong", "UT1"],
        expected_behavior="Xác định đúng UT1 (+2.0 điểm).",
        check_keywords=["2.0", "2,0"],
        any_keywords=["UT1"],
        fail_keywords=["1.0", "0.75"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=33, group=4, group_name="Điểm ưu tiên", difficulty="trung bình",
        question="Khu vực 3 (KV3) được cộng bao nhiêu điểm ưu tiên?",
        hint="KV3: 0 điểm ưu tiên.",
        tags=["uu_tien", "KV3"],
        expected_behavior="Trả lời đúng 0 điểm.",
        check_keywords=["0"],
        any_keywords=[],
        fail_keywords=["0.25", "0.5", "0.75"],
        diagnosis_on_fail="RETRIEVAL_MISS",
        diagnosis_note="Nếu bot trả đúng KV3 nhưng nói có điểm ưu tiên → đổi thành HALLUCINATION",
    ),
    TestCase(
        id=34, group=4, group_name="Điểm ưu tiên", difficulty="khó",
        question="Thí sinh đạt 25 điểm, thuộc KV1, không có đối tượng ưu tiên — điểm xét tuyển là bao nhiêu?",
        hint="25 ≥ 22.5 → giảm dần: [(30-25)/7.5]×0.75=0.50 → ĐXT=25.50.",
        tags=["uu_tien", "tinh_toan", "giam_dan", "KV1"],
        expected_behavior="Tính đúng 25.50 điểm.",
        check_keywords=["25.5", "25,5"],
        any_keywords=[],
        fail_keywords=["25.75", "26.0"],
        diagnosis_on_fail="CALC_ERROR",
    ),
    TestCase(
        id=35, group=4, group_name="Điểm ưu tiên", difficulty="khó",
        question="Thí sinh đạt 20 điểm, thuộc KV2-NT, đối tượng UT2 — điểm xét tuyển là bao nhiêu?",
        hint="20 < 22.5 → cộng thẳng: 20 + 0.50 + 1.0 = 21.50.",
        tags=["uu_tien", "tinh_toan", "cong_thang"],
        expected_behavior="Tính đúng 21.50 điểm.",
        check_keywords=["21.5", "21,5"],
        any_keywords=[],
        fail_keywords=["21.0", "22.0"],
        diagnosis_on_fail="CALC_ERROR",
    ),
    TestCase(
        id=36, group=4, group_name="Điểm ưu tiên", difficulty="khó",
        question="Thí sinh đạt 23 điểm, thuộc KV2, đối tượng UT1 — điểm xét tuyển là bao nhiêu?",
        hint="23 ≥ 22.5 → giảm dần cho tổng mức=2.25: [(30-23)/7.5]×2.25=2.10 → ĐXT=25.10.",
        tags=["uu_tien", "tinh_toan", "giam_dan", "KV2", "UT1"],
        expected_behavior="Tính đúng 25.10.",
        check_keywords=["25.1", "25,1"],
        any_keywords=[],
        fail_keywords=["25.25", "25.0"],
        diagnosis_on_fail="CALC_ERROR",
    ),
    TestCase(
        id=37, group=4, group_name="Điểm ưu tiên", difficulty="trung bình",
        question="Thí sinh học THPT ở Hà Nội nội thành thì thuộc khu vực nào?",
        hint="Các phường của thành phố trực thuộc TW → KV3.",
        tags=["uu_tien", "khu_vuc", "Ha_Noi"],
        expected_behavior="Xác định đúng KV3.",
        check_keywords=["KV3"],
        any_keywords=["khu vực 3"],
        fail_keywords=["KV1", "KV2"],
        diagnosis_on_fail="RETRIEVAL_MISS",
        diagnosis_note="Nếu bot trả sai khu vực dù context đúng → đổi thành AMBIGUITY_FAIL (router classify nhầm intent)",
    ),
    TestCase(
        id=38, group=4, group_name="Điểm ưu tiên", difficulty="khó",
        question="Nếu thí sinh học lớp 10 ở tỉnh (KV2-NT), lớp 11-12 ở Hà Nội nội thành (KV3), thì được xác định là khu vực nào?",
        hint="Theo quy tắc trường học lâu nhất; 11-12 dài hơn (2 năm) ở HN nội thành → KV3.",
        tags=["uu_tien", "khu_vuc", "quy_tac"],
        expected_behavior="Xác định KV3.",
        check_keywords=["KV3"],
        any_keywords=["khu vực 3"],
        fail_keywords=["KV1", "KV2-NT"],
        diagnosis_on_fail="RETRIEVAL_MISS",
        diagnosis_note="Câu này cần chunk về quy tắc chuyển vùng — nếu context_preview trống → RETRIEVAL_MISS đúng. Nếu có context nhưng bot suy luận sai → MULTI_STEP_FAIL",
    ),
    TestCase(
        id=39, group=4, group_name="Điểm ưu tiên", difficulty="trung bình",
        question="Điểm ưu tiên khu vực được hưởng trong bao nhiêu năm sau tốt nghiệp THPT?",
        hint="Trong năm tốt nghiệp THPT và một năm kế tiếp (2 năm).",
        tags=["uu_tien", "thoi_gian"],
        expected_behavior="Nêu đúng: năm tốt nghiệp + 1 năm kế tiếp.",
        check_keywords=["1 năm", "một năm"],
        any_keywords=[],
        fail_keywords=["3 năm", "5 năm"],
        diagnosis_on_fail="RETRIEVAL_MISS",
        diagnosis_note="Nếu bot trả '2 năm' thay vì 'năm TN + 1 năm kế tiếp' → check_keywords chưa bắt được, xem xét thêm '2 năm' vào any_keywords",
    ),
    TestCase(
        id=40, group=4, group_name="Điểm ưu tiên", difficulty="khó",
        question="Một thí sinh vừa thuộc diện con thương binh, vừa là người dân tộc thiểu số KV1 — được hưởng mức ưu tiên nào?",
        hint="Chỉ được hưởng MỘT mức đối tượng cao nhất + khu vực KV1.",
        tags=["uu_tien", "nhieu_dien", "quy_tac"],
        expected_behavior="Giải thích đúng: chỉ 1 mức đối tượng cao nhất + khu vực.",
        check_keywords=["một", "cao nhất"],
        any_keywords=["không cộng 2", "chỉ 1"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
        diagnosis_note="Nếu bot cộng 2 lần đối tượng → đổi thành CALC_ERROR. Nếu bot không tìm được quy tắc → RETRIEVAL_MISS đúng",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 5 — Học phí & học bổng (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=41, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Sinh viên K20 chương trình đại trà học phí bao nhiêu tiền/tín chỉ?",
        hint="700.000 đồng/tín chỉ.",
        tags=["hoc_phi", "K20"],
        expected_behavior="Trả lời đúng 700.000 đồng/tín chỉ.",
        check_keywords=["700.000", "700000"],
        any_keywords=[],
        fail_keywords=["1.000.000", "550.000", "495.000"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=42, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Chương trình đào tạo bằng tiếng Anh học phí là bao nhiêu?",
        hint="1.000.000 đồng/tín chỉ.",
        tags=["hoc_phi", "tieng_anh"],
        expected_behavior="Trả lời đúng 1.000.000 đồng/tín chỉ.",
        check_keywords=["1.000.000", "1000000"],
        any_keywords=[],
        fail_keywords=["700.000", "550.000"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=43, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Học bổng HaUI toàn khóa dành cho đối tượng nào?",
        hint="Thủ khoa từng tổ hợp/phương thức, hoặc đoạt giải HSG/tay nghề quốc gia/quốc tế.",
        tags=["hoc_bong", "HaUI", "toan_khoa"],
        expected_behavior="Liệt kê các diện.",
        check_keywords=["thủ khoa"],
        any_keywords=["giải", "HSG"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=44, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Điều kiện để được duy trì học bổng HaUI sau mỗi học kỳ là gì?",
        hint="TBC học kỳ ≥ 2.5 VÀ rèn luyện loại Tốt VÀ tổng tín chỉ xét ≥ 15.",
        tags=["hoc_bong", "dieu_kien_duy_tri"],
        expected_behavior="Nêu đủ cả 3 điều kiện.",
        check_keywords=["2.5", "15"],
        any_keywords=["Tốt"],
        fail_keywords=[],
        diagnosis_on_fail="INCOMPLETE",
    ),
    TestCase(
        id=45, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Học bổng KKHT loại Xuất sắc cần điều kiện gì?",
        hint="TBC học kỳ ≥ 3.6/4.0 + rèn luyện Xuất sắc hoặc Tốt.",
        tags=["hoc_bong", "KKHT", "xuat_sac"],
        expected_behavior="Nêu đúng ngưỡng điểm 3.6.",
        check_keywords=["3.6"],
        any_keywords=["Xuất sắc"],
        fail_keywords=["3.2", "2.5"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=46, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Học bổng Nguyễn Thanh Bình dành cho ai?",
        hint="Sinh viên có hoàn cảnh khó khăn: bệnh hiểm nghèo, khuyết tật, mồ côi, hộ nghèo/cận nghèo.",
        tags=["hoc_bong", "NTB"],
        expected_behavior="Liệt kê đúng các diện.",
        check_keywords=["khó khăn"],
        any_keywords=["mồ côi", "khuyết tật", "hộ nghèo"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=47, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Sinh viên đã nhận học bổng HaUI toàn khóa có được xét học bổng Nguyễn Thanh Bình không?",
        hint="KHÔNG. Sinh viên nhận HB HaUI toàn khóa bị loại khỏi diện xét NTB trong toàn khóa.",
        tags=["hoc_bong", "loai_tru"],
        expected_behavior="Trả lời rõ KHÔNG.",
        check_keywords=["không"],
        any_keywords=[],
        fail_keywords=["được xét", "có thể"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=48, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Điều kiện tín chỉ tối thiểu trong học kỳ để xét học bổng KKHT là bao nhiêu?",
        hint="≥ 15 tín chỉ (học kỳ cuối ≥ 7 tín chỉ).",
        tags=["hoc_bong", "KKHT", "tin_chi"],
        expected_behavior="Nêu đúng ≥15 tín chỉ.",
        check_keywords=["15"],
        any_keywords=[],
        fail_keywords=["10", "12", "20"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=49, group=5, group_name="Học phí & học bổng", difficulty="trung bình",
        question="Thủ khoa xét theo PT4 (ĐGNL) có được học bổng gì?",
        hint="Học bổng HaUI toàn khóa (100% học phí toàn khóa học).",
        tags=["hoc_bong", "thu_khoa", "PT4"],
        expected_behavior="Nêu đúng học bổng HaUI toàn khóa = 100% học phí.",
        check_keywords=["100%", "toàn khóa"],
        any_keywords=[],
        fail_keywords=["năm thứ nhất", "5 triệu"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=50, group=5, group_name="Học phí & học bổng", difficulty="khó",
        question="Sinh viên năm 4 muốn học trước học phần thạc sĩ được hỗ trợ học bổng như thế nào?",
        hint="30% học phí học phần thạc sĩ, TBC tích lũy ≥ 2.5, tối đa 15 tín chỉ.",
        tags=["hoc_bong", "thac_si", "nam_4"],
        expected_behavior="Nêu đúng 30% và điều kiện TBC ≥ 2.5.",
        check_keywords=["30%"],
        any_keywords=["2.5", "15 tín chỉ"],
        fail_keywords=["50%", "100%"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 6 — Ký túc xá (5 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=51, group=6, group_name="Ký túc xá", difficulty="dễ",
        question="Phòng KTX chất lượng cao 4 người giá bao nhiêu/tháng?",
        hint="600.000 đồng/sinh viên/tháng.",
        tags=["KTX", "gia_phong"],
        expected_behavior="Trả lời đúng 600.000 đồng.",
        check_keywords=["600.000", "600000"],
        any_keywords=[],
        fail_keywords=["800.000", "400.000", "465.000"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=52, group=6, group_name="Ký túc xá", difficulty="dễ",
        question="Phòng tiêu chuẩn 6 người tại cơ sở 2 giá bao nhiêu/tháng?",
        hint="280.000 đồng/sinh viên/tháng (CS2, phòng 6 người tiêu chuẩn).",
        tags=["KTX", "co_so_2", "tieu_chuan"],
        expected_behavior="Trả lời đúng 280.000 đồng.",
        check_keywords=["280.000", "280000"],
        any_keywords=[],
        fail_keywords=["310.000", "400.000"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=53, group=6, group_name="Ký túc xá", difficulty="dễ",
        question="Phòng KTX có điều hòa không? Loại phòng nào có?",
        hint="Chỉ phòng CLC (chất lượng cao) có điều hòa. Phòng tiêu chuẩn không có.",
        tags=["KTX", "dieu_hoa"],
        expected_behavior="Nêu rõ chỉ phòng CLC có điều hòa.",
        check_keywords=["chất lượng cao"],
        any_keywords=["CLC"],
        fail_keywords=["tiêu chuẩn có"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=54, group=6, group_name="Ký túc xá", difficulty="dễ",
        question="Điện nước KTX tính theo giá nào?",
        hint="Theo chỉ số đồng hồ thực tế, áp dụng giá nhà nước.",
        tags=["KTX", "dien_nuoc"],
        expected_behavior="Nêu đúng: tính theo đồng hồ, giá nhà nước.",
        check_keywords=["nhà nước"],
        any_keywords=["đồng hồ"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=55, group=6, group_name="Ký túc xá", difficulty="dễ",
        question="KTX HaUI có những tiện ích gì?",
        hint="Nhà ăn, nhà xe, siêu thị mini, quán cafe, sân thể thao.",
        tags=["KTX", "tien_ich"],
        expected_behavior="Liệt kê đủ 5 tiện ích.",
        check_keywords=["nhà ăn", "sân thể thao"],
        any_keywords=["siêu thị", "cafe"],
        fail_keywords=[],
        diagnosis_on_fail="INCOMPLETE",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 7 — Quy trình đăng ký & nhập học (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=56, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="trung bình",
        question="Đăng ký dự tuyển online cần chuẩn bị những tài liệu gì?",
        hint="9 loại: chân dung, 2 mặt CCCD, học bạ, chứng chỉ QT, giải HSG, ĐGNL, ĐGTD, ưu tiên đối tượng.",
        tags=["dang_ky", "ho_so"],
        expected_behavior="Liệt kê đầy đủ các loại tài liệu.",
        check_keywords=["CCCD", "học bạ"],
        any_keywords=["chứng chỉ", "ảnh"],
        fail_keywords=[],
        diagnosis_on_fail="INCOMPLETE",
    ),
    TestCase(
        id=57, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="trung bình",
        question="Sau khi xác nhận đăng ký, có thể tự sửa nguyện vọng không?",
        hint="Không tự sửa được — hệ thống tự khóa. Phải liên hệ Nhà trường.",
        tags=["dang_ky", "sua_nguyen_vong"],
        expected_behavior="Xác nhận KHÔNG tự sửa được.",
        check_keywords=["không"],
        any_keywords=["khóa", "liên hệ"],
        fail_keywords=["có thể sửa", "được sửa"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=58, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="dễ",
        question="Đăng ký xong rồi, có cần gửi hồ sơ bản cứng về trường không?",
        hint="KHÔNG. Chỉ nộp bản cứng khi trúng tuyển.",
        tags=["dang_ky", "ho_so_cung"],
        expected_behavior="Trả lời KHÔNG.",
        check_keywords=["không"],
        any_keywords=[],
        fail_keywords=["có", "phải gửi"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=59, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="dễ",
        question="Sinh viên trúng tuyển nhập học trực tuyến qua đâu?",
        hint="Ứng dụng MyHaUI hoặc nhaphoc.haui.edu.vn.",
        tags=["nhap_hoc", "truc_tuyen"],
        expected_behavior="Nêu đúng 2 kênh.",
        check_keywords=["MyHaUI", "nhaphoc.haui.edu.vn"],
        any_keywords=[],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=60, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="trung bình",
        question="Nhập học năm học 2025-2026, học kỳ I bắt đầu khi nào?",
        hint="Học kỳ I: từ 07/9/2026 (theo lịch 2026).",
        tags=["nhap_hoc", "lich"],
        expected_behavior="Nêu đúng 07/9/2026.",
        check_keywords=["07/9", "7/9"],
        any_keywords=["tháng 9"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=61, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="trung bình",
        question="Thẻ sinh viên HaUI tích hợp với ngân hàng nào?",
        hint="Vietinbank.",
        tags=["the_SV", "ngan_hang"],
        expected_behavior="Nêu đúng Vietinbank.",
        check_keywords=["Vietinbank"],
        any_keywords=[],
        fail_keywords=["BIDV", "Vietcombank", "MB Bank"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=62, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="trung bình",
        question="Hồ sơ miễn giảm học phí cho sinh viên dân tộc thiểu số cần giấy tờ gì?",
        hint="Giấy xác nhận vùng ĐBKK hoặc hộ nghèo/cận nghèo + CT07 xác nhận cư trú + bản sao giấy khai sinh.",
        tags=["mien_giam_hoc_phi", "dan_toc"],
        expected_behavior="Liệt kê đúng 3 loại giấy tờ.",
        check_keywords=["khai sinh"],
        any_keywords=["CT07", "hộ nghèo"],
        fail_keywords=[],
        diagnosis_on_fail="INCOMPLETE",
    ),
    TestCase(
        id=63, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="dễ",
        question="Muốn ở KTX thì đăng ký ở đâu?",
        hint="ssc.haui.edu.vn.",
        tags=["KTX", "dang_ky"],
        expected_behavior="Nêu đúng địa chỉ ssc.haui.edu.vn.",
        check_keywords=["ssc.haui.edu.vn"],
        any_keywords=[],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=64, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="dễ",
        question="Khi đăng nhập hệ thống báo 'Đăng nhập không thành công' thì phải làm gì?",
        hint="Kiểm tra CCCD và mật khẩu; thử copy-paste từ email; bấm 'Quên mật khẩu'; hoặc gọi 0834560255.",
        tags=["dang_ky", "su_co", "FAQ"],
        expected_behavior="Hướng dẫn đúng.",
        check_keywords=["mật khẩu"],
        any_keywords=["CCCD", "email", "quên"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=65, group=7, group_name="Quy trình đăng ký & nhập học", difficulty="trung bình",
        question="Kết quả thi ĐGTD muốn cập nhật lại nếu thi lại điểm cao hơn thì phải làm gì và trước ngày nào?",
        hint="Liên hệ Nhà trường trước ngày 05/7/2025 và upload lại ảnh kết quả.",
        tags=["DGTD", "cap_nhat_ket_qua"],
        expected_behavior="Nêu đúng deadline 05/7/2025.",
        check_keywords=["05/7", "5/7"],
        any_keywords=[],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 8 — So sánh & tư vấn chọn ngành (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=66, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Tôi thích lập trình, nên chọn ngành Công nghệ thông tin, Kỹ thuật phần mềm hay Khoa học máy tính? Sự khác biệt là gì?",
        hint="CNTT rộng; KTPM thiên quy trình phần mềm; KHMT thiên lý thuyết AI.",
        tags=["tu_van", "CNTT", "KTPM", "KHMT"],
        expected_behavior="Phân biệt rõ 3 ngành.",
        check_keywords=["Kỹ thuật phần mềm", "Khoa học máy tính"],
        any_keywords=["Công nghệ thông tin"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=67, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Ngành Cơ điện tử và ngành Robot & Trí tuệ nhân tạo khác nhau như thế nào?",
        hint="Cơ điện tử: cơ-điện-điều khiển công nghiệp. Robot & TTNT: AI, machine learning, robot thông minh.",
        tags=["tu_van", "co_dien_tu", "robot_AI"],
        expected_behavior="Phân biệt dựa trên chuẩn đầu ra.",
        check_keywords=["Cơ điện tử", "Robot"],
        any_keywords=["trí tuệ nhân tạo", "AI"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=68, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Tôi muốn làm trong ngành ô tô, nên chọn ngành CN kỹ thuật ô tô hay Cơ điện tử ô tô?",
        hint="CN kỹ thuật ô tô: bảo trì, sửa chữa. Cơ điện tử ô tô: hệ thống điện tử, ECU, xe điện.",
        tags=["tu_van", "o_to"],
        expected_behavior="Phân biệt rõ focus của từng ngành.",
        check_keywords=["ô tô"],
        any_keywords=["điện tử", "sửa chữa"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=69, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Ngành Kế toán và ngành Kiểm toán có gì khác nhau về cơ hội việc làm?",
        hint="Kế toán: ghi sổ, báo cáo nội bộ. Kiểm toán: kiểm toán độc lập, Big4.",
        tags=["tu_van", "ke_toan", "kiem_toan"],
        expected_behavior="Phân biệt rõ.",
        check_keywords=["Kế toán", "Kiểm toán"],
        any_keywords=["Big4", "nội bộ"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=70, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Tôi học tổ hợp D01, muốn học ngành kỹ thuật tại HaUI thì có những ngành nào phù hợp?",
        hint="D01 xét được nhiều ngành kỹ thuật nhận A01 (có Toán+Anh).",
        tags=["tu_van", "D01", "ky_thuat"],
        expected_behavior="Liệt kê ngành kỹ thuật nhận D01.",
        check_keywords=["D01"],
        any_keywords=["Công nghệ thông tin", "Cơ khí"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=71, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="trung bình",
        question="Ngành nào ở HaUI có thể học chương trình tiếng Anh?",
        hint="Các ngành có mã TA: KHMT, Kế toán, Cơ khí, Ô tô, Điện, Điện tử VT, Du lịch, Khách sạn, v.v.",
        tags=["chuong_trinh_TA"],
        expected_behavior="Liệt kê các ngành có chương trình tiếng Anh.",
        check_keywords=["tiếng Anh"],
        any_keywords=["Khoa học máy tính", "Kế toán"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=72, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="So sánh điểm chuẩn các ngành nhóm CNTT năm 2025 — ngành nào cạnh tranh nhất?",
        hint="KHMT: 23.72 là cao nhất. CNTT: 23.09; KTPM: 21.75...",
        tags=["diem_chuan", "CNTT", "so_sanh", "nam_2025"],
        expected_behavior="Xác định KHMT là cao nhất.",
        check_keywords=["Khoa học máy tính", "23.72"],
        any_keywords=["cao nhất"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=73, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Tôi điểm thi THPT khoảng 22 điểm tổ hợp A01, đăng ký được những ngành nào ở HaUI?",
        hint="Nhiều ngành có điểm chuẩn 2025 ≤22: KTPM(21.75), Mạng MT(21.7), HTTT(21.1)...",
        tags=["tu_van", "22_diem"],
        expected_behavior="Đề xuất ngành phù hợp dưới 22 điểm.",
        check_keywords=["21"],
        any_keywords=["Kỹ thuật phần mềm", "Mạng máy tính"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),
    TestCase(
        id=74, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Chương trình liên kết 2+2 Ngôn ngữ Trung Quốc có gì đặc biệt so với chương trình thường?",
        hint="2 năm HaUI + 2 năm ĐH KH Kỹ thuật Quảng Tây, nhận 2 bằng, chỉ 30 chỉ tiêu.",
        tags=["lien_ket", "TQ"],
        expected_behavior="Nêu 2 bằng và học tại Trung Quốc.",
        check_keywords=["2 bằng", "hai bằng"],
        any_keywords=["Quảng Tây"],
        fail_keywords=[],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=75, group=8, group_name="So sánh & tư vấn chọn ngành", difficulty="khó",
        question="Học ngành Hóa dược ra trường làm được những vị trí gì? Có khác gì so với ngành Công nghệ thực phẩm không?",
        hint="Hóa dược: R&D thuốc, QA/QC dược. Thực phẩm: kỹ sư chế biến, QA/QC thực phẩm.",
        tags=["tu_van", "hoa_duoc", "thuc_pham"],
        expected_behavior="Nêu việc làm từng ngành và chỉ ra điểm khác.",
        check_keywords=["Hóa dược", "thực phẩm"],
        any_keywords=["dược phẩm", "chế biến"],
        fail_keywords=[],
        diagnosis_on_fail="COMPARISON_FAIL",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 9 — Quy đổi điểm (5 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=76, group=9, group_name="Quy đổi điểm", difficulty="khó",
        question="Điểm ĐGNL (HSA) đạt 95 điểm thì quy đổi được bao nhiêu điểm tương đương thang 30?",
        hint="95 → 23.50 điểm.",
        tags=["quy_doi", "HSA"],
        expected_behavior="Trả lời 23.50 điểm.",
        check_keywords=["23.5", "23,5"],
        any_keywords=[],
        fail_keywords=["23.0", "24.0"],
        diagnosis_on_fail="CALC_ERROR",
    ),
    TestCase(
        id=77, group=9, group_name="Quy đổi điểm", difficulty="khó",
        question="Điểm ĐGTD (TSA) đạt 62 điểm thì quy đổi được bao nhiêu điểm?",
        hint="62.00–62.99 → 26.00 điểm.",
        tags=["quy_doi", "TSA"],
        expected_behavior="Trả lời 26.00 điểm.",
        check_keywords=["26.0", "26,0"],
        any_keywords=[],
        fail_keywords=["25.75", "26.25"],
        diagnosis_on_fail="CALC_ERROR",
    ),
    TestCase(
        id=78, group=9, group_name="Quy đổi điểm", difficulty="khó",
        question="Điểm ĐGNL đạt 130 điểm thì quy đổi được bao nhiêu?",
        hint="Từ 130 đến 150 → 30.00 điểm.",
        tags=["quy_doi", "HSA", "toi_da"],
        expected_behavior="Trả lời 30.00 điểm.",
        check_keywords=["30.0", "30,0"],
        any_keywords=[],
        fail_keywords=["29", "27"],
        diagnosis_on_fail="CALC_ERROR",
    ),
    TestCase(
        id=79, group=9, group_name="Quy đổi điểm", difficulty="khó",
        question="Điểm ĐGTD đạt 85 điểm thì quy đổi được bao nhiêu?",
        hint="85.00–100.00 → 30.00 điểm.",
        tags=["quy_doi", "TSA", "toi_da"],
        expected_behavior="Trả lời 30.00 điểm.",
        check_keywords=["30.0", "30,0"],
        any_keywords=[],
        fail_keywords=["29", "28"],
        diagnosis_on_fail="CALC_ERROR",
    ),
    TestCase(
        id=80, group=9, group_name="Quy đổi điểm", difficulty="khó",
        question="Điểm trung bình học bạ môn Toán là 9.25, quy đổi sang thang 10 là bao nhiêu?",
        hint="9.20–9.29 → 8.25 điểm.",
        tags=["quy_doi", "hoc_ba"],
        expected_behavior="Trả lời 8.25 điểm.",
        check_keywords=["8.25"],
        any_keywords=[],
        fail_keywords=["8.75", "7.74"],
        diagnosis_on_fail="CALC_ERROR",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 10 — Câu hỏi bẫy / ngoài dữ liệu (10 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=81, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Điểm chuẩn năm 2026 ngành Công nghệ thông tin là bao nhiêu?",
        hint="Chưa có dữ liệu — chatbot phải từ chối.",
        tags=["bep", "ngoai_du_lieu", "nam_2026"],
        expected_behavior="Từ chối, giải thích chưa có dữ liệu.",
        check_keywords=[],
        any_keywords=["chưa", "chưa có", "không có thông tin"],
        fail_keywords=["điểm chuẩn 2026 là", "năm 2026 là"],
        diagnosis_on_fail="OUT_OF_SCOPE",
    ),
    TestCase(
        id=82, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="HaUI có ngành Y khoa không?",
        hint="Không có ngành Y khoa trong dữ liệu.",
        tags=["bep", "ngoai_du_lieu", "y_khoa"],
        expected_behavior="Trả lời KHÔNG, không bịa.",
        check_keywords=[],
        any_keywords=["không có", "chưa có", "hiện tại không"],
        fail_keywords=["có ngành Y khoa", "HaUI có ngành Y"],
        diagnosis_on_fail="HALLUCINATION",
    ),
    TestCase(
        id=83, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Học phí năm 2026-2027 là bao nhiêu?",
        hint="Chưa có dữ liệu 2026-2027.",
        tags=["bep", "ngoai_du_lieu", "hoc_phi_tuong_lai"],
        expected_behavior="Từ chối, cung cấp học phí 2025-2026 hiện có.",
        check_keywords=[],
        any_keywords=["chưa có", "2025-2026", "chưa thông báo"],
        fail_keywords=["2026-2027 là", "năm học 2026 học phí"],
        diagnosis_on_fail="OUT_OF_SCOPE",
    ),
    TestCase(
        id=84, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Mã ngành 7480202 là ngành gì?",
        hint="Trong dữ liệu mã An toàn thông tin là 74802021 (7 chữ số). Cần xử lý đúng.",
        tags=["bep", "ma_nganh", "an_toan_TT"],
        expected_behavior="Nhận dạng An toàn thông tin.",
        check_keywords=[],
        any_keywords=["An toàn thông tin"],
        fail_keywords=["Kỹ thuật phần mềm", "Công nghệ thông tin"],
        diagnosis_on_fail="RETRIEVAL_WRONG",
    ),
    TestCase(
        id=85, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Ngành Công nghệ kỹ thuật nhiệt (7510206) thuộc Khoa nào?",
        hint="Theo dữ liệu: Trường Điện - Điện tử.",
        tags=["bep", "du_lieu_loi", "nhiet"],
        expected_behavior="Nêu theo dữ liệu: Trường Điện - Điện tử.",
        check_keywords=["Điện"],
        any_keywords=[],
        fail_keywords=["Cơ khí", "Kinh tế"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=86, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Tổ hợp X06 gồm những môn gì?",
        hint="X06: Toán, Tin học, Công nghệ.",
        tags=["to_hop", "it_pho_bien"],
        expected_behavior="Trả lời: Toán, Tin học, Công nghệ.",
        check_keywords=["Tin học", "Công nghệ"],
        any_keywords=[],
        fail_keywords=["Vật lý", "Hóa học"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=87, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Tôi có chứng chỉ Tiếng Pháp, học bạ Tiếng Pháp, có xét tuyển ngành Ngôn ngữ Anh được không?",
        hint="Ngôn ngữ Anh chỉ xét D01 (Tiếng Anh). Tiếng Pháp không nằm trong tổ hợp.",
        tags=["bep", "tieng_phap"],
        expected_behavior="Từ chối rõ ràng.",
        check_keywords=["không"],
        any_keywords=["D01", "Tiếng Anh"],
        fail_keywords=["được xét", "có thể"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=88, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Năm 2025 HaUI có tuyển hệ Cao đẳng không?",
        hint="Không thấy chỉ tiêu CĐ trong đề án 2025, nhưng trường có đào tạo CĐ.",
        tags=["bep", "cao_dang"],
        expected_behavior="Trả lời thận trọng dựa trên dữ liệu.",
        check_keywords=[],
        any_keywords=["không có thông tin", "không thấy", "chưa rõ"],
        fail_keywords=[],
        diagnosis_on_fail="AMBIGUITY_FAIL",
    ),
    TestCase(
        id=89, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Học bổng KKHT và học bổng HaUI có thể nhận đồng thời không?",
        hint="KHÔNG. Sinh viên đã nhận HB HaUI không được xét KKHT cùng học kỳ.",
        tags=["hoc_bong", "loai_tru"],
        expected_behavior="Trả lời đúng KHÔNG.",
        check_keywords=["không"],
        any_keywords=[],
        fail_keywords=["được nhận đồng thời", "có thể kết hợp"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),
    TestCase(
        id=90, group=10, group_name="Câu bẫy / ngoài dữ liệu", difficulty="rất khó",
        question="Ngành Phân tích dữ liệu kinh doanh năm 2023 điểm chuẩn PT3 là bao nhiêu?",
        hint="Năm 2023 mã là 7340125, điểm PT3 = 23.67.",
        tags=["diem_chuan", "PTDLKD", "nam_2023"],
        expected_behavior="Tra đúng 23.67 điểm (mã 7340125 năm 2023).",
        check_keywords=["23.67"],
        any_keywords=[],
        fail_keywords=["24.25", "20.0"],
        diagnosis_on_fail="RETRIEVAL_MISS",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # NHÓM 11 — Câu hỏi tổng hợp đa bước (5 câu)
    # ──────────────────────────────────────────────────────────────────────────
    TestCase(
        id=91, group=11, group_name="Câu hỏi tổng hợp đa bước", difficulty="rất khó",
        question="Tôi đạt 24 điểm tổ hợp A01, KV2-NT, muốn học ngành Kỹ thuật phần mềm — tôi có đỗ không? Tính cả điểm ưu tiên.",
        hint="24 > 22.5 → giảm dần: [(30-24)/7.5]×0.50=0.40 → ĐXT=24.40. ĐC 2025: 21.75 → Đủ.",
        tags=["tinh_toan", "da_buoc", "KTPM"],
        expected_behavior="Tính đúng ĐXT=24.40, kết luận đủ tham chiếu.",
        check_keywords=["24.4", "24,4"],
        any_keywords=["21.75"],
        fail_keywords=[],
        diagnosis_on_fail="MULTI_STEP_FAIL",
    ),
    TestCase(
        id=92, group=11, group_name="Câu hỏi tổng hợp đa bước", difficulty="rất khó",
        question="Tôi có IELTS 6.5 và điểm TB học bạ 3 môn A01 là 8.5 — tính điểm xét tuyển PT2 là bao nhiêu?",
        hint="IELTS 6.5→10.0; HB 8.5→8.50–8.59→6.55; ĐXT=6.55×2+10.0=23.10.",
        tags=["PT2", "IELTS", "hoc_ba", "tinh_toan"],
        expected_behavior="Tính đúng ĐXT=23.10.",
        check_keywords=["23.1", "23,1"],
        any_keywords=[],
        fail_keywords=[],
        diagnosis_on_fail="MULTI_STEP_FAIL",
    ),
    TestCase(
        id=93, group=11, group_name="Câu hỏi tổng hợp đa bước", difficulty="rất khó",
        question="Thí sinh đạt HSA 100 điểm, KV1, muốn xét vào ngành Điều khiển và Tự động hóa — có đủ điểm chuẩn 2025 không?",
        hint="HSA 100→24.25; KV1 giảm dần: [(30-24.25)/7.5]×0.75≈0.575→ĐXT≈24.83. ĐC 2025: 26.27 → Không đủ.",
        tags=["quy_doi", "HSA", "uu_tien", "da_buoc"],
        expected_behavior="Tính đúng ~24.83, kết luận không đủ so với 26.27.",
        check_keywords=["không đủ", "chưa đủ"],
        any_keywords=["24.8", "26.27"],
        fail_keywords=["đủ điều kiện"],
        diagnosis_on_fail="MULTI_STEP_FAIL",
    ),
    TestCase(
        id=94, group=11, group_name="Câu hỏi tổng hợp đa bước", difficulty="rất khó",
        question="Một thí sinh có TOPIK 4 và học bạ các môn D01 trung bình 8.0 — điểm xét tuyển PT2 là bao nhiêu?",
        hint="TOPIK 4→9.5; HB 8.0→8.00–8.09→6.27; ĐXT=6.27×2+9.5=22.04.",
        tags=["PT2", "TOPIK", "han_quoc", "da_buoc"],
        expected_behavior="Tính đúng ĐXT~22.04.",
        check_keywords=["22.0", "22,0"],
        any_keywords=["22.04"],
        fail_keywords=[],
        diagnosis_on_fail="MULTI_STEP_FAIL",
    ),
    TestCase(
        id=95, group=11, group_name="Câu hỏi tổng hợp đa bước", difficulty="rất khó",
        question="Nếu tôi muốn học ngành Logistics và ở KTX, chi phí 1 tháng khoảng bao nhiêu tiền?",
        hint="Logistics K20: 700.000đ/TC; ~15TC/HK → 10.5tr/HK ≈ 2.1tr/tháng. KTX CLC 4 người: 600.000đ. Tổng ≈ 2.7tr/tháng.",
        tags=["hoc_phi", "KTX", "chi_phi", "da_buoc"],
        expected_behavior="Ước tính hợp lý, nêu giả định.",
        check_keywords=["700.000", "600.000"],
        any_keywords=["tháng", "tín chỉ"],
        fail_keywords=[],
        diagnosis_on_fail="MULTI_STEP_FAIL",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    id               : int
    group            : int
    group_name       : str
    difficulty       : str
    question         : str
    hint             : str
    tags             : list[str]
    expected_behavior: str
    check_keywords   : list[str]
    fail_keywords    : list[str]
    diagnosis_on_fail: str
    any_keywords     : list[str] = field(default_factory=list)
    answer           : str   = ""
    error            : str   = ""
    duration_sec     : float = 0.0
    intent_type      : str   = ""
    router_method    : str   = ""
    retrieval_tier   : int   = 0
    confidence       : float = 0.0
    cache_hit        : bool  = False
    hyde_used        : bool  = False
    context_preview  : str   = ""   # 600 ký tự đầu của context để debug retrieval
    diagnosis_note   : str   = ""   # ghi chú khi diagnosis_code có thể thay đổi
    # Auto-scored
    score            : str   = ""   # PASS | PARTIAL | FAIL | UNKNOWN
    diagnosis_code   : str   = ""   # xem DIAGNOSIS_TO_COMPONENT


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(
    test_cases : list[TestCase],
    delay_sec  : float = 1.0,
    dry_run    : bool  = False,
) -> list[TestResult]:

    results: list[TestResult] = []

    if dry_run:
        logger.info("=== DRY RUN — liệt kê câu hỏi ===")
        for tc in test_cases:
            logger.info(f"[{tc.id:02d}] G{tc.group} [{tc.difficulty}] {tc.question[:80]}")
        return []

    logger.info("Đang khởi động chatbot...")
    try:
        # Project structure: chatbot.py và chatbot_patch.py nằm cùng cấp với file test
        # (không có src/pipeline/ — import trực tiếp)
        import importlib, sys as _sys

        # Đảm bảo ROOT_DIR (thư mục chứa file test) trong sys.path
        if ROOT_DIR not in _sys.path:
            _sys.path.insert(0, ROOT_DIR)

        chatbot_mod = importlib.import_module("src.pipeline.chatbot")
        Chatbot = chatbot_mod.Chatbot

        # Áp dụng patches từ chatbot_patch.py
        try:
            patch_mod = importlib.import_module("src.pipeline.chatbot_patch")
            if hasattr(patch_mod, "apply_patches_direct_v9"):
                patch_mod.apply_patches_direct_v9(chatbot_mod)
                logger.info("Đã áp dụng chatbot_patch (apply_patches_direct).")
            elif hasattr(patch_mod, "apply_patches"):
                patch_mod.apply_patches_v9()
                logger.info("Đã áp dụng chatbot_patch (apply_patches).")
        except ImportError:
            logger.warning("Không tìm thấy chatbot_patch.py, bỏ qua.")
        except Exception as pe:
            logger.warning(f"chatbot_patch lỗi: {pe} — tiếp tục không patch.")

        bot = Chatbot()
        logger.info("Chatbot sẵn sàng.")
    except Exception as e:
        logger.error(f"Không thể khởi động chatbot: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    total = len(test_cases)
    for idx, tc in enumerate(test_cases, 1):
        logger.info(f"[{idx:02d}/{total}] Câu #{tc.id} — {tc.question[:60]}...")

        result = TestResult(
            id=tc.id, group=tc.group, group_name=tc.group_name,
            difficulty=tc.difficulty, question=tc.question, hint=tc.hint,
            tags=tc.tags, expected_behavior=tc.expected_behavior,
            check_keywords=tc.check_keywords, fail_keywords=tc.fail_keywords,
            any_keywords=tc.any_keywords,
            diagnosis_on_fail=tc.diagnosis_on_fail,
            diagnosis_note=tc.diagnosis_note,
        )

        try:
            bot.reset()
            t0 = time.perf_counter()
            response = bot.chat(tc.question)
            t1 = time.perf_counter()

            result.answer        = response.answer
            result.duration_sec  = round(t1 - t0, 2)
            # response.intent là IntentType (str enum) — lấy .value hoặc str()
            _intent = getattr(response, "intent", None)
            result.intent_type   = _intent.value if hasattr(_intent, "value") else str(_intent or "")
            result.router_method = getattr(response, "router_method", "") or getattr(response, "method", "")
            result.retrieval_tier= getattr(response, "retrieval_tier", 0)
            result.confidence    = round(float(getattr(response, "confidence", 0.0)), 3)
            result.cache_hit     = bool(getattr(response, "cache_hit", False))
            result.hyde_used     = bool(getattr(response, "hyde_used", False))
            ctx = getattr(response, "context", "") or ""
            result.context_preview = ctx[:600].replace("\n", " ") if ctx not in (
                "__GREETING__", "__OFF_TOPIC__", "[cached]", ""
            ) else ctx

            # Auto-score
            score, diag = auto_score(tc, result.answer)
            # Override nếu quá chậm
            if result.duration_sec > DIAGNOSIS_THRESHOLDS["SLOW"]:
                if score == "PASS":
                    score = "PARTIAL"
                diag = "SLOW"
            result.score = score
            result.diagnosis_code = diag

            logger.info(f"   → {score} | diag={diag} | time={result.duration_sec}s")

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.answer = ""
            result.score = "FAIL"
            result.diagnosis_code = "RETRIEVAL_MISS"
            logger.error(f"   ✗ Lỗi: {result.error}")

        results.append(result)

        if idx < total and delay_sec > 0:
            time.sleep(delay_sec)

    return results


def rescore_from_json(json_path: str) -> list[TestResult]:
    """Load kết quả cũ từ JSON và re-score lại (không cần gọi chatbot lại)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Map id → TestCase để lấy keywords mới nhất
    tc_map = {tc.id: tc for tc in TEST_CASES}
    results = []
    for r in data["results"]:
        # Lọc bỏ key không có trong dataclass (tương thích ngược)
        valid_keys = set(TestResult.__dataclass_fields__.keys())
        filtered   = {k: v for k, v in r.items() if k in valid_keys}
        result = TestResult(**filtered)
        tc = tc_map.get(result.id)
        if tc:
            score, diag = auto_score(tc, result.answer)
            result.score = score
            result.diagnosis_code = diag
            # Cập nhật keywords theo version mới nhất
            result.check_keywords = tc.check_keywords
            result.fail_keywords  = tc.fail_keywords
            result.diagnosis_on_fail = tc.diagnosis_on_fail
        results.append(result)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORT BUILDER — tập trung vào "sửa chỗ nào trong code"
# ══════════════════════════════════════════════════════════════════════════════

def _diff_icon(d: str) -> str:
    return {"dễ": "🟢", "trung bình": "🟡", "khó": "🔴", "rất khó": "⚫"}.get(d, "⚪")

def _score_icon(s: str) -> str:
    return {"PASS": "✅", "PARTIAL": "🔶", "FAIL": "❌", "UNKNOWN": "❓"}.get(s, "❓")


def build_markdown_report(results: list[TestResult], meta: dict) -> str:
    now_str = meta.get("run_time", "")
    total   = len(results)
    errors  = sum(1 for r in results if r.error)
    passed  = sum(1 for r in results if r.score == "PASS")
    partial = sum(1 for r in results if r.score == "PARTIAL")
    failed  = sum(1 for r in results if r.score == "FAIL")
    unknown = sum(1 for r in results if r.score == "UNKNOWN")
    pass_rate = round(passed / total * 100, 1) if total else 0

    lines: list[str] = []

    # ── HEADER ──────────────────────────────────────────────────────────────
    lines += [
        "# BÁO CÁO KIỂM TRA CHATBOT HAUI v2",
        "",
        f"> **Thời gian:** {now_str} | **Model:** {meta.get('model', 'N/A')}",
        f"> **Tổng:** {total} | ✅ PASS: {passed} ({pass_rate}%) | "
        f"🔶 PARTIAL: {partial} | ❌ FAIL: {failed} | ❓ UNKNOWN: {unknown} | ⚠️ Lỗi kỹ thuật: {errors}",
        "",
        "---",
        "",
    ]

    # ── SECTION 1: BẢNG ĐIỂM THEO NHÓM ─────────────────────────────────────
    lines += [
        "## 📊 KẾT QUẢ THEO NHÓM",
        "",
        "| Nhóm | Tên | Tổng | ✅ PASS | 🔶 PARTIAL | ❌ FAIL | Pass% |",
        "|------|-----|------|---------|-----------|---------|-------|",
    ]
    from itertools import groupby
    sorted_r = sorted(results, key=lambda r: r.group)
    for gid, gr in groupby(sorted_r, key=lambda r: r.group):
        g = list(gr)
        gname = g[0].group_name
        gp = sum(1 for r in g if r.score == "PASS")
        gpa = sum(1 for r in g if r.score == "PARTIAL")
        gf = sum(1 for r in g if r.score == "FAIL")
        gt = len(g)
        grate = round(gp / gt * 100) if gt else 0
        lines.append(f"| {gid} | {gname} | {gt} | {gp} | {gpa} | {gf} | {grate}% |")
    lines += ["", "---", ""]

    # ── SECTION 2: BẢN ĐỒ BUG — mỗi diagnosis_code bao nhiêu câu fail ────────
    lines += [
        "## 🗺️ BẢN ĐỒ BUG — SỬA Ở ĐÂU TRONG CODE",
        "",
        "Mỗi `diagnosis_code` tương ứng với một component cụ thể trong codebase.",
        "Nhìn vào bảng này để biết nên ưu tiên sửa chỗ nào trước.",
        "",
        "| Diagnosis Code | Số câu FAIL/PARTIAL | Component nghi ngờ | Câu ID |",
        "|----------------|---------------------|---------------------|--------|",
    ]

    from collections import defaultdict
    diag_map: dict[str, list[TestResult]] = defaultdict(list)
    for r in results:
        if r.score in ("FAIL", "PARTIAL") and r.diagnosis_code:
            diag_map[r.diagnosis_code].append(r)
        elif r.score == "FAIL" and not r.diagnosis_code:
            diag_map["UNKNOWN"].append(r)

    for code in sorted(diag_map.keys(), key=lambda c: -len(diag_map[c])):
        items = diag_map[code]
        ids = ", ".join(f"#{r.id}" for r in items[:10])
        if len(items) > 10:
            ids += f" +{len(items)-10} câu"
        comp = DIAGNOSIS_TO_COMPONENT.get(code, "?")
        lines.append(f"| `{code}` | **{len(items)}** | `{comp}` | {ids} |")

    lines += ["", "---", ""]

    # ── SECTION 3: DANH SÁCH CÁC CÂU FAIL/PARTIAL (để fix) ─────────────────
    fail_results = [r for r in results if r.score in ("FAIL", "PARTIAL")]
    if fail_results:
        lines += [
            "## ❌ CHI TIẾT CÁC CÂU FAIL / PARTIAL",
            "",
            "> **Hướng dẫn đọc:** Nhìn vào `diagnosis_code` → tìm component → sửa code.",
            "> Dấu `[MISSING]` = keyword cần có nhưng bot không trả lời.",
            "> Dấu `[FOUND!]`  = keyword cấm xuất hiện trong câu trả lời.",
            "",
        ]
        for r in fail_results:
            icon = _score_icon(r.score)
            diff = _diff_icon(r.difficulty)
            comp = DIAGNOSIS_TO_COMPONENT.get(r.diagnosis_code, "?")

            # Tính keyword nào bị thiếu / nào bị trigger
            strip = True  # default
            ans_n = _normalize(r.answer, strip)

            missing_kw = [kw for kw in r.check_keywords
                          if _normalize(kw, strip) not in ans_n]
            triggered_fail = [kw for kw in r.fail_keywords
                              if _normalize(kw, strip) in ans_n]
            missing_any = r.any_keywords if r.any_keywords and not any(
                _normalize(kw, strip) in ans_n for kw in r.any_keywords
            ) else []

            lines += [
                f"### [{r.id:02d}] {icon} {diff} {r.difficulty.upper()} | "
                f"`{r.diagnosis_code}` → `{comp}`",
                "",
                f"**Câu hỏi:** {r.question}",
                "",
                f"**Đáp án đúng (hint):** `{r.hint}`",
                "",
            ]
            if r.diagnosis_note:
                lines += [f"**Lưu ý chẩn đoán:** _{r.diagnosis_note}_", ""]

            if missing_kw:
                lines.append(f"**Keywords thiếu** (phải có): "
                             f"`{'` | `'.join(missing_kw)}`  → `[MISSING]`")
            if missing_any:
                lines.append(f"**any_keywords thiếu** (cần ít nhất 1): "
                             f"`{'` | `'.join(missing_any)}`  → `[MISSING]`")
            if triggered_fail:
                lines.append(f"**Keywords cấm** (không được có): "
                             f"`{'` | `'.join(triggered_fail)}`  → `[FOUND!]`")

            # Câu trả lời (rút gọn tới 800 ký tự)
            answer_preview = (r.answer or r.error or "(trống)")[:800]
            if len(r.answer or "") > 800:
                answer_preview += "\n... [bị cắt bớt]"
            lines += [
                "",
                "**Câu trả lời bot:**",
                "```",
                answer_preview,
                "```",
                f"**Meta:** intent=`{r.intent_type}` | tier=`{r.retrieval_tier}` | "
                f"conf=`{r.confidence}` | router=`{r.router_method}` | time=`{r.duration_sec}s`",
            ]
            if r.context_preview:
                lines += [
                    "",
                    f"**Context (600 ký tự đầu):** `{r.context_preview[:600]}`",
                ]
            lines += ["", "---", ""]
    else:
        lines += ["## ✅ Không có câu FAIL/PARTIAL!", "", "---", ""]

    # ── SECTION 4: CÁC CÂU PASS (tóm tắt, không in chi tiết) ───────────────
    pass_results = [r for r in results if r.score == "PASS"]
    if pass_results:
        lines += [
            "## ✅ CÁC CÂU PASS (tóm tắt)",
            "",
            "| ID | Nhóm | Difficulty | Time(s) |",
            "|----|------|------------|---------|",
        ]
        for r in pass_results:
            lines.append(f"| {r.id} | {r.group_name} | {r.difficulty} | {r.duration_sec} |")
        lines += ["", "---", ""]

    # ── SECTION 5: GỢI Ý SỬA CODE — tự động theo diagnosis_code ───────────────
    lines += [
        "## 🔧 GỢI Ý SỬA CODE — THEO THỨ TỰ ƯU TIÊN",
        "",
        "Mỗi mục liệt kê: file cần sửa + function cụ thể + gợi ý hành động.",
        "Claude chỉ cần đọc mục này + chi tiết câu FAIL bên trên là đủ để patch.",
        "",
    ]

    # Gợi ý chi tiết theo từng diagnosis_code
    _FIX_HINTS: dict[str, list[str]] = {
        "RETRIEVAL_MISS": [
            "retriever.py: kiểm tra FACTUAL_KEYWORDS, VECTOR_TOP_K, BM25_TOP_K — tăng nếu cần",
            "retriever.py: QueryRewriter._REWRITE_EXAMPLES — thêm ví dụ cho các query bị miss",
            "retriever.py: HyDERetriever — bật HYDE_ENABLED=1, kiểm tra hypothesis có liên quan không",
            "chunker.py + build_index.py: chạy --reset nếu chunk bị thiếu (xem verify_critical_chunks)",
            "retriever.py: SelfReflector — kiểm tra retry_query có đúng không khi context thiếu",
        ],
        "RETRIEVAL_WRONG": [
            "retriever.py: ChunkScorer.compute_bonus() — điều chỉnh SCORE_BONUS_* cho đúng loại content",
            "retriever.py: reciprocal_rank_fusion() — kiểm tra RRF_K, điều chỉnh trọng số vector vs BM25",
            "retriever.py: CrossEncoderReranker — kiểm tra RERANKER_MODEL có phù hợp tiếng Việt không",
        ],
        "CALC_ERROR": [
            "chatbot_patch.py: _tinh_diem_uu_tien_v8() — kiểm tra công thức giảm dần, ngưỡng 22.5",
            "chatbot_patch.py: quy_doi_HSA_fixed() / quy_doi_TSA_fixed() — dùng _tra_bang_safe()",
            "chatbot_patch.py: tinh_diem_PT2() — kiểm tra công thức ĐXT = ĐKQHT×2 + ĐQĐCC + ưu tiên",
            "chatbot_patch.py: _fmt_tinh_diem_uu_tien_v8() — đảm bảo LLM không đọc nhầm số giữa chừng",
            "diem_xet_tuyen.py: tinh_diem_uu_tien() — kiểm tra logic doi_tuong khi extract từ câu hỏi",
        ],
        "HALLUCINATION": [
            "chatbot_patch.py: SYSTEM_PROMPT — tăng cường phần 'KHÔNG suy luận, KHÔNG suy đoán'",
            "chatbot.py: _build_user_prompt_patched() — kiểm tra [LƯU Ý QUAN TRỌNG] có được inject không",
            "chatbot_patch.py: _sanitize_output() — thêm pattern lọc thêm nếu bot trả tiếng Trung/hallucinate",
        ],
        "OUT_OF_SCOPE": [
            "router.py: _FAST_OFF_TOPIC regex — thêm pattern cho câu hỏi ngoài phạm vi",
            "chatbot.py: SYSTEM_PROMPT — rule 2 'Nếu không có thông tin → dùng mẫu câu từ chối'",
            "chatbot.py: _OFF_TOPIC_REPLY — kiểm tra bot có dùng đúng reply này không",
        ],
        "INCOMPLETE": [
            "retriever.py: FINAL_TOP_K — tăng từ 6 lên 8-10 cho các câu hỏi liệt kê",
            "chatbot.py: ContextCompressor.compress() — kiểm tra MAX_CONTEXT_CHARS có đủ không",
            "chatbot.py: SYSTEM_PROMPT — thêm instruction 'liệt kê đầy đủ khi được hỏi danh sách'",
        ],
        "WRONG_YEAR": [
            "chatbot_patch.py: _ctx_diem_chuan_v8() — kiểm tra fallback năm, ưu tiên năm mới nhất",
            "diem_chuan.py: get_diem_chuan(nam=) — kiểm tra filter năm có đúng không",
            "retriever.py: metadata filter — thêm năm vào where clause nếu query có 'năm X'",
        ],
        "WRONG_METHOD": [
            "chatbot_patch.py: _ctx_diem_chuan_v8() — kiểm tra map PT ngôn ngữ tự nhiên → PT code",
            "diem_chuan.py: get_diem_chuan(phuong_thuc=) — kiểm tra PT filter có match không",
        ],
        "AMBIGUITY_FAIL": [
            "router.py: _CLASSIFY_SYSTEM prompt — thêm ví dụ cho loại câu hỏi bị nhầm intent",
            "router.py: LLMClassifier — kiểm tra ROUTER_MODEL, ROUTER_TIMEOUT đủ chưa",
            "router.py: INTENT_EXAMPLES — thêm câu mẫu cho intent bị miss",
        ],
        "MULTI_STEP_FAIL": [
            "chatbot_patch.py: _ctx_dau_truot_v8() — kiểm tra quy trình: quy đổi → ưu tiên → so sánh DC",
            "chatbot_patch.py: _ctx_quy_doi_diem_v8() — kiểm tra detect loại điểm (HSA/TSA/học bạ/CC)",
            "chatbot_patch.py: tinh_diem_PT2() — kiểm tra bảng _PT2_CC_BANG đủ loại chứng chỉ chưa",
        ],
        "COMPARISON_FAIL": [
            "retriever.py: FINAL_TOP_K — tăng để lấy đủ ngành khi so sánh",
            "chatbot.py: ContextBuilder._ctx_nganh_theo_khoa() — kiểm tra có trả đủ ngành không",
            "nganh.py: get_nganh_theo_khoa() — kiểm tra _resolve_khoa() match đúng alias không",
            "diem_chuan.py: get_diem_chuan_theo_khoa() — kiểm tra filter nhom_nganh",
        ],
        "SLOW": [
            "chatbot.py: ResponseCache — kiểm tra TTL, cache key có hoạt động không",
            "retriever.py: _retrieve_cache — kiểm tra cache hit rate",
            "retriever.py: LLM_TIMEOUT — điều chỉnh timeout QueryRewriter/HyDE/SelfReflector",
        ],
    }

    for rank, (code, items) in enumerate(
        sorted(diag_map.items(), key=lambda x: -len(x[1])), 1
    ):
        comp     = DIAGNOSIS_TO_COMPONENT.get(code, "?")
        ids_str  = ", ".join(f"#{r.id}" for r in items)
        hints    = _FIX_HINTS.get(code, ["(xem chi tiết câu trả lời bên trên)"])
        # Lấy hints từ các câu fail để gợi ý thêm
        sample_hints = set()
        for r in items[:3]:
            if r.hint:
                sample_hints.add(f"Đáp đúng ví dụ: `{r.hint[:80]}`")

        lines += [
            f"### {rank}. `{code}` — {len(items)} câu ({ids_str})",
            f"**File cần xem:** `{comp}`",
            "**Các bước sửa:**",
        ]
        for h in hints:
            lines.append(f"- {h}")
        if sample_hints:
            lines.append("**Tham chiếu đáp án đúng (từ hint):**")
            for sh in sample_hints:
                lines.append(f"  - {sh}")
        lines.append("")

    lines += [
        "---",
        "",
        f"*Báo cáo tự động bởi run_chatbot_tests_v2.py — {now_str}*",
        f"*Auto-score dựa trên keyword matching. Câu UNKNOWN cần review thủ công.*",
    ]

    return "\n".join(lines)


def build_json_export(results: list[TestResult], meta: dict) -> str:
    export = {
        "meta"   : meta,
        "results": [asdict(r) for r in results],
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Chạy bộ câu hỏi kiểm tra chatbot HaUI v2 (có auto-scoring).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--groups", nargs="+", type=int, default=None)
    p.add_argument("--ids",    nargs="+", type=int, default=None)
    p.add_argument("--difficulty", choices=["dễ", "trung bình", "khó", "rất khó"])
    p.add_argument("--delay",  type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--no-json", action="store_true")
    p.add_argument(
        "--rescore", type=str, default=None, metavar="JSON_FILE",
        help="Load kết quả từ JSON cũ và re-score lại (không gọi chatbot).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    now     = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    ts_str  = now.strftime("%Y%m%d_%H%M%S")
    base_name = args.output or f"test_results_{ts_str}"

    meta = {
        "run_time"   : now_str,
        "model"      : os.getenv("OLLAMA_MODEL", os.getenv("GROQ_MODEL", "unknown")),
        "version"    : "run_chatbot_tests_v2.py",
        "slow_threshold": DIAGNOSIS_THRESHOLDS["SLOW"],
    }

    # ── Re-score mode ──────────────────────────────────────────────────────
    if args.rescore:
        logger.info(f"Re-scoring từ file: {args.rescore}")
        results = rescore_from_json(args.rescore)
        meta["rescore_source"] = args.rescore
    else:
        # ── Normal run ─────────────────────────────────────────────────────
        selected = list(TEST_CASES)
        if args.groups:
            selected = [tc for tc in selected if tc.group in args.groups]
        if args.ids:
            selected = [tc for tc in selected if tc.id in args.ids]
        if args.difficulty:
            selected = [tc for tc in selected if tc.difficulty == args.difficulty]

        if not selected:
            logger.error("Không có câu hỏi nào được chọn.")
            sys.exit(1)

        meta["total_cases"] = len(selected)
        meta["groups"]      = args.groups or "all"

        results = run_tests(selected, delay_sec=args.delay, dry_run=args.dry_run)

        if args.dry_run:
            logger.info("Dry run hoàn tất.")
            return

    # ── Xuất kết quả ──────────────────────────────────────────────────────
    md_path = f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(results, meta))
    logger.info(f"Báo cáo Markdown: {md_path}")

    if not args.no_json:
        json_path = f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(build_json_export(results, meta))
        logger.info(f"Raw data JSON: {json_path}")

    # ── Tóm tắt terminal ──────────────────────────────────────────────────
    total   = len(results)
    passed  = sum(1 for r in results if r.score == "PASS")
    partial = sum(1 for r in results if r.score == "PARTIAL")
    failed  = sum(1 for r in results if r.score == "FAIL")
    print("\n" + "="*65)
    print(f"HOÀN TẤT — {total} câu hỏi")
    print(f"  ✅ PASS    : {passed} ({round(passed/total*100)}%)")
    print(f"  🔶 PARTIAL : {partial}")
    print(f"  ❌ FAIL    : {failed}")
    print(f"  Báo cáo   : {md_path}")
    print("="*65)
    print("\n📌 Gửi file .md này cho Claude — Claude nhìn vào là biết sửa chỗ nào.")
    print("   Claude KHÔNG cần xem data gốc, chỉ cần đọc báo cáo này.")


if __name__ == "__main__":
    main()