"""
chatbot.py  (v3 — Ollama backend)
RAG Pipeline: Router + query_json + Retriever + Qwen2.5 qua Ollama

THAY ĐỔI so với v2:
  - LLM backend: llama-cpp-python → Ollama HTTP API
  - Không cần cài llama-cpp-python, không cần path file GGUF
  - Ollama chạy background, gọi qua http://localhost:11434
  - Giữ nguyên toàn bộ: Router, Retriever, ContextBuilder, multi-turn

YÊU CẦU:
  1. Cài Ollama: https://ollama.com/download/windows
  2. Tải model: ollama pull qwen2.5:3b
  3. Ollama tự chạy background sau khi cài — không cần làm gì thêm
  4. pip install requests  (thường đã có sẵn)
"""

from __future__ import annotations

import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
import random
import logging
import requests
from dataclasses import dataclass
from typing import Iterator, Optional

from src.pipeline.router     import Router, Intent, IntentType
from src.retrieval.retriever import Retriever
from src.pipeline.profiler   import LatencyProfiler
from src.query_json import (
    get_chi_tieu_nganh, get_nganh_theo_to_hop,
    get_chi_tieu_tong_2026, get_mon_thi_to_hop,
    get_diem_chuan, get_diem_chuan_moi_nhat, get_lich_su_diem_chuan,
    get_hoc_phi,
    quy_doi_HSA, quy_doi_TSA, quy_doi_KQHB,
    get_diem_uu_tien_khu_vuc, get_diem_uu_tien_doi_tuong,
    tinh_diem_uu_tien, kiem_tra_dau_truot,
    fmt_diem_chuan, fmt_hoc_phi, fmt_chi_tieu_nganh,
    fmt_nganh_theo_to_hop, fmt_chi_tieu_2026, fmt_mon_thi_to_hop,
    fmt_tinh_diem_uu_tien, fmt_quy_doi, fmt_kiem_tra_dau_truot,
)

logger = logging.getLogger("haui.chatbot")

# ── Cấu hình ─────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MAX_HISTORY     = 6
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.3

# ── Groq config (thay thế Ollama khi có internet) ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
USE_GROQ     = bool(GROQ_API_KEY)   # tự động dùng Groq nếu có API key

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn tuyển sinh của Đại học Công nghiệp Hà Nội (HaUI).

NGUYÊN TẮC BẮT BUỘC:
1. CHỈ trả lời đúng điều người dùng hỏi — KHÔNG lan man sang chủ đề khác.
2. LUÔN dùng số liệu cụ thể từ THÔNG TIN THAM KHẢO — điểm, tiền, năm, tên.
3. Nếu context có số liệu → PHẢI trích dẫn, KHÔNG được nói "chưa có thông tin".
4. Nếu context THỰC SỰ không có → nói "Tôi chưa có dữ liệu về vấn đề này."
5. KHÔNG bịa số liệu ngoài context.
6. Trả lời ngắn gọn, đúng trọng tâm, tối đa 5 câu, không dùng emoji.
7. KHÔNG thêm câu hỏi ngược, không thêm thông tin ngoài phạm vi câu hỏi.
8. Trả lời bằng tiếng Việt.

VÍ DỤ:
- Hỏi học phí → CHỈ trả lời học phí, KHÔNG đề cập ký túc xá hay học bổng.
- Hỏi điểm chuẩn → CHỈ trả lời điểm chuẩn, KHÔNG đề cập học phí hay ngành học.
- Hỏi ngành học → CHỈ trả lời về ngành đó, KHÔNG liệt kê các ngành khác."""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Message:
    role   : str
    content: str


@dataclass
class ChatResponse:
    answer    : str
    intent    : IntentType
    method    : str
    context   : str
    confidence: float


# ── Ollama LLM wrapper ────────────────────────────────────────────────────────

class OllamaLLM:
    """
    Wrapper gọi Qwen2.5 qua Ollama HTTP API.
    Ollama chạy background ở http://localhost:11434.
    """

    def __init__(
        self,
        model   : str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self._model    = model
        self._base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self) -> None:
        """Kiểm tra Ollama đang chạy và model đã tải."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Không kết nối được Ollama.\n"
                "Kiểm tra Ollama đang chạy — mở Ollama app hoặc chạy 'ollama serve'"
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi kết nối Ollama: {e}")

        # Kiểm tra model đã pull chưa
        models    = [m["name"] for m in resp.json().get("models", [])]
        model_base = self._model.split(":")[0]
        found      = any(model_base in m for m in models)
        if not found:
            raise RuntimeError(
                f"Model '{self._model}' chưa được tải.\n"
                f"Chạy lệnh: ollama pull {self._model}\n"
                f"Models hiện có: {models}"
            )
        logger.info(f"Ollama OK — model: {self._model}")

    def _build_messages(
        self,
        system  : str,
        history : list[Message],
        user_msg: str,
    ) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_msg})
        return messages

    def generate(
        self,
        system  : str,
        history : list[Message],
        user_msg: str,
    ) -> str:
        messages = self._build_messages(system, history, user_msg)
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model"   : self._model,
                    "messages": messages,
                    "stream"  : False,
                    "options" : {
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_NEW_TOKENS,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.error("Ollama timeout")
            return "Xin lỗi, model phản hồi quá chậm. Vui lòng thử lại."
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi."

    def generate_stream(
        self,
        system  : str,
        history : list[Message],
        user_msg: str,
    ) -> Iterator[str]:
        messages = self._build_messages(system, history, user_msg)
        try:
            with requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model"   : self._model,
                    "messages": messages,
                    "stream"  : True,
                    "options" : {
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_NEW_TOKENS,
                    },
                },
                stream = True,
                timeout= 120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data  = json.loads(line.decode("utf-8"))
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.Timeout:
            yield "\n[Lỗi: model phản hồi quá chậm]"
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield "\n[Lỗi kết nối Ollama]"


# ── Groq LLM wrapper ─────────────────────────────────────────────────────────

class GroqLLM:
    """
    Wrapper gọi LLM qua Groq API — nhanh ~1-2s, miễn phí.
    Tự động dùng khi có GROQ_API_KEY trong .env.

    Cài: pip install groq
    Key: https://console.groq.com
    """

    def __init__(self, model: str = GROQ_MODEL, api_key: str = GROQ_API_KEY):
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("Chạy: pip install groq")
        if not api_key:
            raise RuntimeError("Thiếu GROQ_API_KEY — thêm vào file .env")
        self._client = Groq(api_key=api_key)
        self._model  = model
        logger.info(f"GroqLLM OK — model: {self._model}")

    def _build_messages(self, system, history, user_msg):
        messages = [{"role": "system", "content": system}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_msg})
        return messages

    def generate(self, system, history, user_msg) -> str:
        messages = self._build_messages(system, history, user_msg)
        try:
            resp = self._client.chat.completions.create(
                model       = self._model,
                messages    = messages,
                max_tokens  = MAX_NEW_TOKENS,
                temperature = TEMPERATURE,
                stream      = False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi."

    def generate_stream(self, system, history, user_msg) -> Iterator[str]:
        messages = self._build_messages(system, history, user_msg)
        try:
            stream = self._client.chat.completions.create(
                model       = self._model,
                messages    = messages,
                max_tokens  = MAX_NEW_TOKENS,
                temperature = TEMPERATURE,
                stream      = True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            yield "\n[Lỗi kết nối Groq]"


# ── Normalize nhẹ (không dùng LLM) ───────────────────────────────────────────

def _light_normalize(query: str) -> str:
    """Chuẩn hóa nhẹ bằng regex — không gọi LLM."""
    q = query.strip()
    abbrevs = {
        r"\bcntt\b": "công nghệ thông tin",
        r"\bktpm\b": "kỹ thuật phần mềm",
        r"\bqtkd\b": "quản trị kinh doanh",
        r"\bhttt\b": "hệ thống thông tin",
        r"\btmdt\b": "thương mại điện tử",
        r"\btmđt\b": "thương mại điện tử",
    }
    q_lower = q.lower()
    for pattern, replacement in abbrevs.items():
        q_lower = re.sub(pattern, replacement, q_lower)

    QUESTION_WORDS = (
        "bao nhiêu", "mấy", "như thế nào", "là gì", "ở đâu",
        "khi nào", "có không", "được không",
    )
    if any(w in q_lower for w in QUESTION_WORDS) and not q.endswith("?"):
        q = q + "?"
    return q


# ── Context Builder ───────────────────────────────────────────────────────────

class ContextBuilder:
    def __init__(self, retriever: Retriever):
        self._retriever = retriever
        self._pending: dict = {}

    def build(self, query: str, intent: Intent) -> str:
        if intent.intent_type == IntentType.GREETING:
            return "__GREETING__"
        if intent.intent_type == IntentType.OFF_TOPIC:
            return "__OFF_TOPIC__"

        handler = {
            IntentType.JSON_DIEM_CHUAN      : self._ctx_diem_chuan,
            IntentType.JSON_HOC_PHI         : self._ctx_hoc_phi,
            IntentType.JSON_CHI_TIEU_TO_HOP : self._ctx_chi_tieu_to_hop,
            IntentType.JSON_QUY_DOI_DIEM    : self._ctx_quy_doi_diem,
            IntentType.JSON_DAU_TRUOT       : self._ctx_dau_truot,
            IntentType.RAG_MO_TA_NGANH      : self._ctx_rag,
            IntentType.RAG_FAQ              : self._ctx_rag,
            IntentType.RAG_TRUONG_HOC_BONG  : self._ctx_rag,
            IntentType.UNKNOWN              : self._ctx_rag,
            IntentType.GREETING             : self._ctx_rag,
            IntentType.OFF_TOPIC            : self._ctx_rag,
        }
        return handler.get(intent.intent_type, self._ctx_rag)(query, intent)

    def _ctx_diem_chuan(self, query: str, intent: Intent) -> str:
        e       = intent.entities
        ten     = e.get("nganh", query)
        nam     = e.get("nam")
        pt      = e.get("phuong_thuc")
        q_lower = query.lower()

        SO_SANH_KEYWORDS = [
            "so sánh", "xu hướng", "qua các năm", "nhiều năm",
            "liên tiếp", "tăng", "giảm", "biến động", "3 năm",
            "các năm", "lịch sử", "từ năm",
        ]
        if any(kw in q_lower for kw in SO_SANH_KEYWORDS):
            result = get_lich_su_diem_chuan(ten)
        elif nam:
            result = get_diem_chuan(ten, nam=nam, phuong_thuc=pt)
        else:
            result = get_diem_chuan_moi_nhat(ten)

        ctx = fmt_diem_chuan(result)
        if not result["found"]:
            ctx += "\n\n" + self._retriever.retrieve_as_context(query)
        return ctx

    def _ctx_hoc_phi(self, query: str, intent: Intent) -> str:
        nganh  = intent.entities.get("nganh", "")

        # Thử theo ngành trước
        result = get_hoc_phi(nganh) if nganh else {"found": False}

        # Nếu không có ngành → thử match theo query
        if not result.get("found"):
            result = get_hoc_phi(query)

        # Nếu vẫn không match → lấy toàn bộ bảng học phí (đại học chính quy)
        if not result.get("found"):
            result = get_hoc_phi("")   # lấy tất cả

        return fmt_hoc_phi(result)

    def _ctx_chi_tieu_to_hop(self, query: str, intent: Intent) -> str:
        e        = intent.entities
        contexts = []
        if e.get("to_hop"):
            contexts.append(fmt_nganh_theo_to_hop(get_nganh_theo_to_hop(e["to_hop"])))
        if e.get("nganh"):
            contexts.append(fmt_chi_tieu_nganh(get_chi_tieu_nganh(e["nganh"])))
        if "2026" in query or "năm tới" in query.lower():
            contexts.append(fmt_chi_tieu_2026(get_chi_tieu_tong_2026()))
        if not contexts:
            contexts.append(self._retriever.retrieve_as_context(query))
        return "\n\n".join(contexts)

    def _ctx_quy_doi_diem(self, query: str, intent: Intent) -> str:
        e        = intent.entities
        contexts = []
        q_lower  = query.lower()
        diem     = e.get("diem")
        diem_30  = diem

        if diem:
            if "hsa" in q_lower or "năng lực" in q_lower:
                r = quy_doi_HSA(diem)
                contexts.append(fmt_quy_doi(r))
                if r["found"]: diem_30 = r["diem_quy_doi"]
            elif "tsa" in q_lower or "tư duy" in q_lower:
                r = quy_doi_TSA(diem)
                contexts.append(fmt_quy_doi(r))
                if r["found"]: diem_30 = r["diem_quy_doi"]
            elif "học bạ" in q_lower:
                r = quy_doi_KQHB(diem)
                contexts.append(fmt_quy_doi(r))
                if r["found"]: diem_30 = round(r["diem_quy_doi"] * 3, 2)

        kv = e.get("khu_vuc")
        dt = e.get("doi_tuong")

        if kv and diem_30:
            contexts.append(fmt_tinh_diem_uu_tien(tinh_diem_uu_tien(diem_30, kv, dt)))
        elif kv:
            r = get_diem_uu_tien_khu_vuc(kv)
            if r["found"]:
                contexts.append(
                    f"Mức điểm ưu tiên tối đa {r['ten']} ({r['ma']}): +{r['diem']} điểm\n"
                    f"Lưu ý: Nếu tổng điểm ≥ 22.5, điểm ưu tiên thực tế sẽ thấp hơn."
                )
        elif dt:
            r = get_diem_uu_tien_doi_tuong(dt)
            if r["found"]:
                contexts.append(
                    f"Mức điểm ưu tiên tối đa đối tượng {dt} (nhóm {r['nhom']}): +{r['diem']} điểm\n"
                    f"Lưu ý: Nếu tổng điểm ≥ 22.5, điểm ưu tiên thực tế sẽ thấp hơn."
                )

        if not contexts:
            contexts.append(self._retriever.retrieve_as_context(query))
        return "\n\n".join(contexts)

    def _ctx_dau_truot(self, query: str, intent: Intent) -> str:
        e       = intent.entities
        nganh   = e.get("nganh")
        diem    = e.get("diem")
        kv      = e.get("khu_vuc")
        dt      = e.get("doi_tuong")
        nam     = e.get("nam", 2024)
        q_lower = query.lower()

        if self._pending.get("waiting_for") == "phuong_thuc":
            if self._is_cancel(query):
                self._pending = {}
                return "__CANCELLED__"
            if nganh is None:  nganh = self._pending.get("nganh")
            if diem  is None:  diem  = self._pending.get("diem")
            if kv    is None:  kv    = self._pending.get("khu_vuc")
            self._pending = {}

        if not nganh or not diem:
            return (
                self._retriever.retrieve_as_context(query) + "\n\n"
                "Lưu ý: Để kiểm tra chính xác, tôi cần biết:\n"
                "  - Tên ngành bạn muốn xét tuyển\n"
                "  - Tổng điểm 3 môn (thang 30) hoặc điểm TSA/HSA\n"
                "  - Khu vực (KV1/KV2/KV2-NT/KV3) nếu có"
            )

        contexts  = []
        diem_30   = diem
        pt_filter = None

        if "tư duy" in q_lower or "tsa" in q_lower:
            r = quy_doi_TSA(diem)
            if r["found"]:
                diem_30 = r["diem_quy_doi"]
                pt_filter = "PT5"
                contexts.append(fmt_quy_doi(r))
        elif "năng lực" in q_lower or "hsa" in q_lower:
            r = quy_doi_HSA(diem)
            if r["found"]:
                diem_30 = r["diem_quy_doi"]
                pt_filter = "PT4"
                contexts.append(fmt_quy_doi(r))
        elif "học bạ" in q_lower or "pt2" in q_lower:
            r = quy_doi_KQHB(diem)
            if r["found"]:
                diem_30 = round(r["diem_quy_doi"] * 3, 2)
                pt_filter = "PT2"
                contexts.append(fmt_quy_doi(r) + f"\n  → Điểm 3 môn thang 30: {diem_30}")
        elif any(kw in q_lower for kw in ["thpt", "thi thpt", "pt3", "tốt nghiệp"]):
            pt_filter = "PT3"
        else:
            self._pending = {
                "waiting_for": "phuong_thuc",
                "nganh"      : nganh,
                "diem"       : diem,
                "khu_vuc"    : kv,
            }
            dc_all = get_diem_chuan(nganh, nam=2024)
            ctx_dc = ""
            if dc_all["found"]:
                ctx_dc = "\n\nĐiểm chuẩn tham khảo năm 2024:\n" + fmt_diem_chuan(dc_all)
            return (
                f"Bạn có {diem} điểm muốn xét tuyển ngành {nganh}.{ctx_dc}\n\n"
                f"__CLARIFY__ Bạn thi theo phương thức nào?\n"
                f"1. Thi THPT (PT3)\n"
                f"2. Đánh giá tư duy TSA (PT5)\n"
                f"3. Đánh giá năng lực HSA (PT4)\n"
                f"4. Xét học bạ (PT2)"
            )

        diem_xet = diem_30
        if kv:
            ut = tinh_diem_uu_tien(diem_30, kv, dt)
            contexts.append(fmt_tinh_diem_uu_tien(ut))
            if ut["found"]: diem_xet = ut["diem_xet_tuyen"]

        import json as _json, os as _os
        _dc_json_path = _os.path.join(
            _os.path.dirname(__file__), "..", "..", "data", "processed",
            "diem_chuan_2023_2024_2025.json"
        )
        try:
            dc = _json.loads(open(_dc_json_path, encoding="utf-8").read())
        except Exception:
            dc = []

        _dc_parts = []
        _dc_chung = get_diem_chuan(nganh, phuong_thuc="chung")
        _r2025    = None
        if _dc_chung["found"]:
            _r2025 = next(
                (r for r in _dc_chung["ket_qua"] if r["nam"] == 2025), None
            )
        if _r2025:
            _pts = ", ".join(_r2025.get("cac_phuong_thuc_ap_dung", []))
            _dc_parts.append(f"• 2025 (chung — áp dụng {_pts}): {_r2025['diem_chuan']} điểm")
            dc_val = _r2025["diem_chuan"]
        else:
            dc_val = None

        _PT_CODE_MAP = {"PT3": "PT3", "PT5": "PT5", "PT4": "PT4", "PT2": "PT2"}
        _pt_code = _PT_CODE_MAP.get(pt_filter) if pt_filter else None

        if _pt_code:
            for _year in [2024, 2023]:
                _r = next(
                    (r for r in dc if r.get("ten_nganh") == nganh
                     and r["nam"] == _year
                     and r.get("phuong_thuc_code") == _pt_code),
                    None
                )
                if _r:
                    _dc_parts.append(f"• {_year} ({_pt_code}): {_r['diem_chuan']} điểm")
                    if dc_val is None:
                        dc_val = _r["diem_chuan"]

        if _dc_parts:
            contexts.append(
                f"Điểm chuẩn ngành {nganh} (3 năm gần nhất):\n" + "\n".join(_dc_parts)
            )

        if dc_val is not None:
            diff = round(diem_xet - dc_val, 2)
            if _pt_code:
                _dc_2024 = next(
                    (r["diem_chuan"] for r in dc
                     if r.get("ten_nganh") == nganh and r["nam"] == 2024
                     and r.get("phuong_thuc_code") == _pt_code),
                    None
                )
                if _dc_2024:
                    diff_2024 = round(diem_xet - _dc_2024, 2)
                    diff_2025 = round(diem_xet - (_r2025["diem_chuan"] if _r2025 else dc_val), 2)
                    if diff_2025 >= 0 and diff_2024 >= 0:
                        nhan_xet = (
                            f"Khả năng cao — điểm xét {diem_xet} cao hơn "
                            f"cả 2025/chung ({_r2025['diem_chuan'] if _r2025 else '?'}) "
                            f"và {_pt_code}/2024 ({_dc_2024})"
                        )
                    elif diff_2025 >= 0 and diff_2024 < 0:
                        nhan_xet = (
                            f"Khả năng trung bình — điểm xét {diem_xet} "
                            f"cao hơn 2025/chung ({_r2025['diem_chuan'] if _r2025 else '?'}) "
                            f"nhưng thấp hơn {_pt_code}/2024 ({_dc_2024}, thiếu {abs(diff_2024)} điểm). "
                            f"Điểm 2026 chưa công bố — cần theo dõi thêm."
                        )
                    else:
                        nhan_xet = (
                            f"Khả năng thấp — điểm xét {diem_xet} "
                            f"thấp hơn cả 2025/chung ({_r2025['diem_chuan'] if _r2025 else '?'}, "
                            f"thiếu {abs(diff_2025)} điểm) lẫn {_pt_code}/2024 ({_dc_2024})"
                        )
                else:
                    nhan_xet = (
                        f"Điểm xét {diem_xet} so với 2025/chung "
                        f"({'cao hơn' if diff>=0 else 'thấp hơn'} {abs(diff)} điểm)"
                    )
            else:
                nhan_xet = (
                    f"Điểm xét {diem_xet} so với điểm chuẩn 2025/chung "
                    f"({'cao hơn' if diff>=0 else 'thấp hơn'} {abs(diff)} điểm). "
                    f"Điểm chuẩn 2026 chưa công bố."
                )
            contexts.append(f"Điểm xét tuyển của bạn: {diem_xet} điểm\n{nhan_xet}")
        else:
            _r_dt = kiem_tra_dau_truot(nganh, diem_xet, nam=nam, phuong_thuc=pt_filter)
            contexts.append(fmt_kiem_tra_dau_truot(_r_dt))

        if nam == 2026:
            contexts.append(
                "Lưu ý: Điểm chuẩn năm 2026 chưa công bố. "
                "Kết quả trên dựa trên điểm chuẩn các năm trước để tham khảo."
            )

        return "\n\n".join(ctx for ctx in contexts if ctx)

    def reset(self) -> None:
        self._pending = {}

    def _ctx_rag(self, query: str, intent: Intent) -> str:
        return self._retriever.retrieve_as_context(query, intent_type=intent.intent_type)

    def _is_cancel(self, text: str) -> bool:
        t = text.lower().strip().rstrip("!.?")
        return t in Chatbot._CANCEL_PHRASES or any(p in t for p in Chatbot._CANCEL_PHRASES)


# ── Chatbot ───────────────────────────────────────────────────────────────────

class Chatbot:
    """
    Chatbot tư vấn tuyển sinh HaUI — dùng Ollama làm LLM backend.

    Cách dùng:
        bot = Chatbot()
        bot = Chatbot(model="qwen2.5:1.5b")   # model nhẹ hơn

        # Non-streaming
        resp = bot.chat("Điểm chuẩn ngành CNTT năm 2024?")
        print(resp.answer)

        # Streaming (cho Chainlit)
        for token in bot.chat_stream("Ngành CNTT học gì?"):
            print(token, end="", flush=True)
    """

    _CANCEL_PHRASES = {
        "thôi", "bỏ qua", "không cần", "không muốn", "dừng lại",
        "cancel", "hủy", "không hỏi nữa", "bỏ đi", "thôi được",
        "không", "thôi không", "bỏ", "đủ rồi", "không cần nữa",
    }

    _OFF_TOPIC_REPLY = (
        "Xin lỗi, tôi chỉ có thể tư vấn về tuyển sinh HaUI — "
        "điểm chuẩn, học phí, ngành học, phương thức xét tuyển và các thông tin liên quan.\n\n"
        "Bạn có câu hỏi nào về tuyển sinh không?"
    )

    _GREETING_REPLIES = [
        "Xin chào! Tôi là trợ lý tư vấn tuyển sinh HaUI.\nBạn cần tư vấn gì về tuyển sinh không?",
        "Chào bạn! Tôi có thể giúp bạn tra cứu điểm chuẩn, học phí, ngành học.\nBạn muốn hỏi gì?",
        "Xin chào! Rất vui được hỗ trợ bạn.\nBạn đang quan tâm đến ngành nào tại HaUI?",
    ]
    _THANKS_REPLIES = [
        "Không có gì! Nếu cần thêm thông tin về tuyển sinh HaUI, tôi luôn sẵn sàng.",
        "Rất vui được giúp bạn! Hãy hỏi thêm nếu cần nhé.",
    ]
    _BYE_REPLIES = [
        "Tạm biệt! Chúc bạn thi tốt và đạt kết quả như mong muốn!",
        "Hẹn gặp lại! Chúc bạn may mắn trong kỳ xét tuyển!",
    ]

    def __init__(
        self,
        model    : str = OLLAMA_MODEL,
        base_url : str = OLLAMA_BASE_URL,
        retriever: Optional[Retriever] = None,
        use_hybrid: bool = True,
    ):
        # 1. Khởi tạo LLM — ưu tiên Groq nếu có API key, fallback Ollama
        if USE_GROQ:
            logger.info("Backend: Groq API")
            self._llm = GroqLLM()
        else:
            logger.info("Backend: Ollama local")
            self._llm = OllamaLLM(model=model, base_url=base_url)

        # 2. Khởi tạo Retriever
        if retriever:
            self._retriever = retriever
        else:
            self._retriever = (
                Retriever(use_reranker=True, use_bm25=True)
                if use_hybrid
                else Retriever(use_reranker=True, use_bm25=False)
            )

        # 3. Router — dùng lại embedder của Retriever, không load thêm model
        self._router = Router()
        try:
            self._router.init_embedder(self._retriever._embedder.embed_query)
            logger.info("Router: embedding classifier sẵn sàng")
        except Exception as e:
            logger.warning(f"Router embedding không khởi tạo được: {e}. Chỉ dùng rule-based.")

        # 4. Context builder và history
        self._ctx_builder = ContextBuilder(self._retriever)
        self._history    : list[Message] = []
        self._pending    : dict          = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_cancel(self, text: str) -> bool:
        t = text.lower().strip().rstrip("!.?")
        return t in self._CANCEL_PHRASES or any(p in t for p in self._CANCEL_PHRASES)

    def _pick_greeting_reply(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["cảm ơn", "thanks", "thank", "cám ơn"]):
            return random.choice(self._THANKS_REPLIES)
        if any(w in q for w in ["bye", "tạm biệt", "hẹn gặp"]):
            return random.choice(self._BYE_REPLIES)
        return random.choice(self._GREETING_REPLIES)

    def _build_user_prompt(self, query: str, context: str) -> str:
        return (
            f"THÔNG TIN THAM KHẢO:\n"
            f"{'─' * 40}\n"
            f"{context}\n"
            f"{'─' * 40}\n\n"
            f"CÂU HỎI: {query}"
        )

    def _trim_history(self) -> None:
        if len(self._history) > MAX_HISTORY * 2:
            self._history = self._history[-(MAX_HISTORY * 2):]

    def _try_fast_path(self, query: str) -> Optional[tuple[Intent, str]]:
        """GREETING / OFF_TOPIC — xử lý ngay không cần LLM."""
        from src.pipeline.router import _rule_match
        quick = _rule_match(query)
        if quick and quick[0] in (IntentType.GREETING, IntentType.OFF_TOPIC) \
                and quick[1] >= 0.65:
            intent = Intent(
                intent_type = quick[0],
                confidence  = quick[1],
                entities    = {},
                method      = "rule",
            )
            context = self._ctx_builder.build(query, intent)
            return intent, context
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> ChatResponse:
        p = LatencyProfiler()
        p.mark("start")

        user_message = _light_normalize(user_message)
        p.mark("normalize")

        # Cancel pending
        if self._pending and self._is_cancel(user_message):
            self._pending = {}
            answer = "Được rồi, tôi đã hủy. Bạn có muốn hỏi gì khác không?"
            self._history.append(Message("user", user_message))
            self._history.append(Message("assistant", answer))
            return ChatResponse(
                answer=answer, intent=IntentType.UNKNOWN,
                method="rule", context="", confidence=1.0,
            )

        # Fast-path GREETING / OFF_TOPIC
        fast = self._try_fast_path(user_message)
        if fast:
            intent, context = fast
            answer = self._pick_greeting_reply(user_message) \
                if context == "__GREETING__" else self._OFF_TOPIC_REPLY
            self._history.append(Message("user", user_message))
            self._history.append(Message("assistant", answer))
            p.mark("fast_done")
            p.report(query=user_message)
            return ChatResponse(
                answer=answer, intent=intent.intent_type,
                method=intent.method, context=context, confidence=intent.confidence,
            )

        # Classify + entity merge
        intent = self._router.classify(user_message)
        p.mark("classify")

        from src.pipeline.router import _extract_entities_rule
        for k, v in _extract_entities_rule(user_message).items():
            intent.entities.setdefault(k, v)

        # Build context
        context = self._ctx_builder.build(user_message, intent)
        p.mark("context")

        # Generate
        if context == "__CANCELLED__":
            answer = "Được rồi, tôi đã hủy. Bạn có muốn hỏi gì khác không?"
        elif context == "__GREETING__":
            answer = self._pick_greeting_reply(user_message)
        elif context == "__OFF_TOPIC__":
            answer = self._OFF_TOPIC_REPLY
        else:
            if "__CLARIFY__" in context:
                context = context.replace("__CLARIFY__", "")
            user_prompt = self._build_user_prompt(user_message, context)
            answer = self._llm.generate(
                system   = SYSTEM_PROMPT,
                history  = self._history[-MAX_HISTORY * 2:],
                user_msg = user_prompt,
            )
        p.mark("generate")
        p.report(query=user_message)

        self._history.append(Message("user", user_message))
        self._history.append(Message("assistant", answer))
        self._trim_history()

        return ChatResponse(
            answer     = answer,
            intent     = intent.intent_type,
            method     = intent.method,
            context    = context,
            confidence = intent.confidence,
        )

    def chat_stream(self, user_message: str) -> Iterator[str]:
        """Streaming version — yield từng token, dùng cho Chainlit."""
        p = LatencyProfiler()
        p.mark("start")

        user_message = _light_normalize(user_message)
        p.mark("normalize")

        # Fast-path GREETING / OFF_TOPIC
        fast = self._try_fast_path(user_message)
        if fast:
            intent, context = fast
            answer = self._pick_greeting_reply(user_message) \
                if context == "__GREETING__" else self._OFF_TOPIC_REPLY
            self._history.append(Message("user", user_message))
            self._history.append(Message("assistant", answer))
            p.mark("fast_done")
            p.report(query=user_message)
            yield answer
            return

        # Cancel pending
        if self._pending and self._is_cancel(user_message):
            self._pending = {}
            answer = "Được rồi, tôi đã hủy câu hỏi. Bạn có muốn hỏi gì khác không?"
            self._history.append(Message("user", user_message))
            self._history.append(Message("assistant", answer))
            yield answer
            return

        # Classify + entity merge
        intent = self._router.classify(user_message)
        p.mark("classify")

        from src.pipeline.router import _extract_entities_rule
        for k, v in _extract_entities_rule(user_message).items():
            intent.entities.setdefault(k, v)

        context = self._ctx_builder.build(user_message, intent)
        p.mark("context")

        # Trường hợp đặc biệt không cần generate
        if context == "__CANCELLED__":
            answer = "Được rồi, tôi đã hủy câu hỏi. Bạn có muốn hỏi gì khác không?"
            self._history.append(Message("user", user_message))
            self._history.append(Message("assistant", answer))
            yield answer
            return

        if "__CLARIFY__" in context:
            context = context.replace("__CLARIFY__", "")

        user_prompt = self._build_user_prompt(user_message, context)
        full_answer = ""
        first_token = True

        for token in self._llm.generate_stream(
            system   = SYSTEM_PROMPT,
            history  = self._history[-MAX_HISTORY * 2:],
            user_msg = user_prompt,
        ):
            if first_token:
                p.mark("first_token")
                p.report(query=user_message)
                first_token = False
            full_answer += token
            yield token

        self._history.append(Message("user", user_message))
        self._history.append(Message("assistant", full_answer))
        self._trim_history()

    def reset(self) -> None:
        """Reset toàn bộ session."""
        self._history.clear()
        self._pending = {}
        self._ctx_builder.reset()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("Khởi động HaUI Chatbot (Ollama + Qwen2.5)...")
    try:
        bot = Chatbot()
    except RuntimeError as e:
        print(f"\nLỗi: {e}")
        sys.exit(1)

    print("Sẵn sàng! Gõ 'exit' để thoát.\n")
    while True:
        try:
            q = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "thoát"):
            print("Tạm biệt!")
            break

        print("Bot: ", end="", flush=True)
        for token in bot.chat_stream(q):
            print(token, end="", flush=True)
        print("\n")