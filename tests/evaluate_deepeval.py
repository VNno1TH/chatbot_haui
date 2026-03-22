"""
evaluate_deepeval.py — Đánh giá RAG chatbot HaUI bằng DeepEval
Thay thế RAGAs, hỗ trợ Ollama local, không cần API key.

Metrics:
  - FaithfulnessMetric      : câu trả lời có trung thực với context?
  - AnswerRelevancyMetric   : câu trả lời có liên quan câu hỏi?
  - ContextualRecallMetric  : context có đủ thông tin không?

Cài đặt:
    pip install deepeval

Chạy:
    python tests/evaluate_deepeval.py --quick --save
    python tests/evaluate_deepeval.py --cat diem_chuan --save
    python tests/evaluate_deepeval.py --save
    python tests/evaluate_deepeval.py --stats
"""

import sys, os, json, time, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from haui_ragas_dataset import EVAL_DATASET, print_stats

JSON_INTENTS = {
    "JSON_DIEM_CHUAN", "JSON_HOC_PHI", "JSON_CHI_TIEU_TO_HOP",
    "JSON_QUY_DOI_DIEM", "JSON_DAU_TRUOT",
}


# ── Setup DeepEval với Ollama ──────────────────────────────────────────────────

def _setup_deepeval():
    """Cấu hình DeepEval dùng Ollama làm judge."""
    try:
        import deepeval
        print(f"  DeepEval version: {deepeval.__version__}")
    except ImportError:
        print("❌ Cần cài: pip install deepeval")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()

    ollama_url   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11435")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    # Kiểm tra Ollama
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5):
            pass
        print(f"  ✓ Ollama OK — {ollama_model} @ {ollama_url}")
    except Exception as e:
        print(f"  ❌ Ollama không kết nối: {e}")
        sys.exit(1)

    # Cấu hình DeepEval dùng Ollama qua OpenAI-compatible endpoint
    from deepeval.models import DeepEvalBaseLLM
    from openai import OpenAI

    class OllamaDeepEvalLLM(DeepEvalBaseLLM):
        """Wrapper Ollama cho DeepEval."""

        def __init__(self, model: str, base_url: str):
            self.model    = model
            self.base_url = base_url
            self._client  = OpenAI(
                base_url = base_url.rstrip("/") + "/v1",
                api_key  = "ollama",
                timeout  = 120,
            )

        def get_model_name(self) -> str:
            return self.model

        def load_model(self):
            return self._client

        def _extract_json(self, text: str) -> dict:
            """Trích JSON từ response kể cả khi có markdown code block."""
            import re
            # Thử parse trực tiếp
            try:
                return json.loads(text)
            except Exception:
                pass
            # Tìm trong ```json ... ```
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            # Tìm { ... } đầu tiên
            m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {}

        def _call_llm(self, prompt: str, use_json: bool = False) -> str:
            """Gọi Ollama với retry tối đa 3 lần."""
            for attempt in range(3):
                try:
                    kwargs = dict(
                        model       = self.model,
                        messages    = [{"role": "user", "content": prompt}],
                        temperature = 0,
                        max_tokens  = 1000,
                    )
                    if use_json:
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = self._client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1)

        def generate(self, prompt: str, schema=None):
            if schema:
                # Thêm hướng dẫn JSON rõ ràng vào prompt
                json_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Respond with ONLY a valid JSON object. "
                    + "No markdown, no explanation, just the JSON."
                )
                content = self._call_llm(json_prompt, use_json=True)
                data = self._extract_json(content)
                if data:
                    try:
                        return schema(**data)
                    except Exception:
                        # Thử từng field một
                        try:
                            fields = schema.__fields__ if hasattr(schema, '__fields__') else {}
                            filtered = {k: v for k, v in data.items() if k in fields}
                            return schema(**filtered)
                        except Exception:
                            pass
                # Retry không có json_object mode
                content2 = self._call_llm(json_prompt, use_json=False)
                data2 = self._extract_json(content2)
                if data2:
                    try:
                        return schema(**data2)
                    except Exception:
                        pass
                raise ValueError(f"Cannot parse schema from: {content[:200]}")
            else:
                return self._call_llm(prompt, use_json=False)

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

    llm = OllamaDeepEvalLLM(model=ollama_model, base_url=ollama_url)
    print(f"  ✓ DeepEval LLM judge: OllamaDeepEvalLLM → {ollama_model}")
    return llm


# ── Thu thập pipeline data ─────────────────────────────────────────────────────

def _collect_pipeline_data(dataset: list, verbose: bool = True) -> list:
    from src.pipeline.chatbot    import Chatbot
    from src.retrieval.retriever import Retriever

    print("  Khởi tạo pipeline chatbot...")
    retriever = Retriever(use_reranker=True, use_bm25=True)
    bot       = Chatbot(retriever=retriever)
    print("  ✓ Pipeline sẵn sàng\n")

    rows = []
    for i, item in enumerate(dataset, 1):
        if verbose:
            print(f"  [{i:03d}/{len(dataset)}] [{item['category'].upper():<20}] "
                  f"{item['question'][:50]}...")

        t0 = time.perf_counter()
        try:
            resp    = bot.chat(item["question"])
            answer  = resp.answer
            ctx_raw = resp.context or ""
            bot.reset()
        except Exception as e:
            answer  = f"ERROR: {e}"
            ctx_raw = ""

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Lấy contexts
        contexts = []
        try:
            if item["intent"] in JSON_INTENTS:
                if ctx_raw and ctx_raw not in ("__GREETING__", "__OFF_TOPIC__"):
                    parts = [p.strip() for p in ctx_raw.replace("__CLARIFY__","").split("---") if p.strip()]
                    contexts = parts[:4] if parts else [ctx_raw[:800]]
                else:
                    contexts = [answer]
            else:
                chunks = retriever.retrieve(item["question"], top_k=4)
                contexts = [c.text for c in chunks] if chunks else [ctx_raw[:800] or "No context."]
        except Exception:
            contexts = [answer]

        if not contexts:
            contexts = ["No context available."]

        kw_hit   = [k for k in item["keywords"] if k.lower() in answer.lower()]
        kw_miss  = [k for k in item["keywords"] if k.lower() not in answer.lower()]
        hit_rate = round(len(kw_hit)/len(item["keywords"]), 2) if item["keywords"] else 1.0

        if verbose:
            bar = "▓"*int(hit_rate*10) + "░"*(10-int(hit_rate*10))
            print(f"         {latency_ms:>5}ms | kw [{bar}] {hit_rate:.0%} | "
                  f"miss: {kw_miss if kw_miss else '—'}")

        rows.append({
            "id"           : item["id"],
            "category"     : item["category"],
            "question"     : item["question"],
            "answer"       : answer,
            "contexts"     : contexts,
            "ground_truth" : item["expected"],
            "keywords"     : item["keywords"],
            "keywords_hit" : kw_hit,
            "keywords_miss": kw_miss,
            "kw_hit_rate"  : hit_rate,
            "latency_ms"   : latency_ms,
            "intent"       : item["intent"],
        })
        time.sleep(0.05)

    return rows


# ── Chạy DeepEval metrics ──────────────────────────────────────────────────────

def _run_deepeval(rows: list, llm, verbose: bool = True) -> list:
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
    )

    # Khởi tạo metrics với Ollama judge
    faithfulness_m = FaithfulnessMetric(
        threshold = 0.5,
        model     = llm,
        verbose_mode = False,
    )
    relevancy_m = AnswerRelevancyMetric(
        threshold = 0.5,
        model     = llm,
        verbose_mode = False,
    )
    recall_m = ContextualRecallMetric(
        threshold = 0.5,
        model     = llm,
        verbose_mode = False,
    )

    print(f"\n  Đánh giá {len(rows)} câu với DeepEval metrics...")
    scored = []

    for i, r in enumerate(rows, 1):
        if verbose:
            print(f"  [{i:03d}/{len(rows)}] {r['question'][:50]}...", end=" ", flush=True)

        t0 = time.perf_counter()

        # Tạo test case
        test_case = LLMTestCase(
            input               = r["question"],
            actual_output       = r["answer"],
            expected_output     = r["ground_truth"],
            retrieval_context   = r["contexts"],
        )

        scores = {}
        try:
            faithfulness_m.measure(test_case)
            scores["faithfulness"] = round(faithfulness_m.score, 3)
        except Exception as e:
            scores["faithfulness"] = None
            if verbose: print(f"\n    ⚠ faithfulness: {e}")

        try:
            relevancy_m.measure(test_case)
            scores["answer_relevancy"] = round(relevancy_m.score, 3)
        except Exception as e:
            scores["answer_relevancy"] = None
            if verbose: print(f"\n    ⚠ answer_relevancy: {e}")

        # Context recall chỉ cho RAG intents
        if r["intent"] not in JSON_INTENTS:
            try:
                recall_m.measure(test_case)
                scores["context_recall"] = round(recall_m.score, 3)
            except Exception as e:
                scores["context_recall"] = None
                if verbose: print(f"\n    ⚠ context_recall: {e}")
        else:
            scores["context_recall"] = None

        elapsed = int((time.perf_counter()-t0)*1000)
        if verbose:
            f_str = f"{scores['faithfulness']:.2f}" if scores["faithfulness"] is not None else "N/A"
            r_str = f"{scores['answer_relevancy']:.2f}" if scores["answer_relevancy"] is not None else "N/A"
            c_str = f"{scores['context_recall']:.2f}" if scores["context_recall"] is not None else "N/A"
            print(f"faith={f_str} rel={r_str} recall={c_str} ({elapsed}ms)")

        scored.append({**r, **scores})
        time.sleep(0.1)

    return scored


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation(
    quick    : bool = False,
    category : str  = "",
    save     : bool = False,
) -> dict:
    print("=" * 70)
    print("  DeepEval RAG Evaluation — Chatbot Tuyển sinh HaUI")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    ds = EVAL_DATASET
    if category:
        ds = [d for d in ds if d["category"] == category]
        if not ds:
            valid = sorted(set(d["category"] for d in EVAL_DATASET))
            print(f"❌ Danh mục '{category}' không tồn tại. Hợp lệ: {valid}")
            return {}
        print(f"  Danh mục: {category} ({len(ds)} câu)")
    if quick:
        ds = ds[:15]
        print(f"  Quick mode: {len(ds)} câu")
    print(f"  Tổng: {len(ds)} câu\n")

    # Bước 1
    print("── Bước 1: Khởi tạo DeepEval + Ollama judge ──")
    llm = _setup_deepeval()
    print()

    # Bước 2
    print("── Bước 2: Thu thập câu trả lời từ pipeline ──")
    rows = _collect_pipeline_data(ds, verbose=True)
    avg_kw  = sum(r["kw_hit_rate"] for r in rows) / len(rows)
    avg_lat = sum(r["latency_ms"] for r in rows) // len(rows)
    print(f"\n  Keyword hit rate TB: {avg_kw:.1%}")
    print(f"  Latency TB         : {avg_lat}ms")

    # Bước 3
    print("\n── Bước 3: Chạy DeepEval metrics ──")
    scored = _run_deepeval(rows, llm, verbose=True)

    # Bước 4: Tổng hợp
    print("\n── Bước 4: Tổng hợp kết quả ──\n")

    def _avg(key):
        vals = [r[key] for r in scored if r.get(key) is not None]
        return round(sum(vals)/len(vals), 4) if vals else None

    avg_faith  = _avg("faithfulness")
    avg_rel    = _avg("answer_relevancy")
    avg_recall = _avg("context_recall")

    cat_kw = {}
    for r in scored:
        cat_kw.setdefault(r["category"], []).append(r["kw_hit_rate"])
    cat_kw_avg = {c: round(sum(v)/len(v), 3) for c, v in cat_kw.items()}

    THRESH = {"faithfulness": 0.80, "answer_relevancy": 0.75, "context_recall": 0.70}

    def _bar(score):
        return "▓"*int(score*20) + "░"*(20-int(score*20))

    def _status(score, key):
        if score is None: return "➖"
        t = THRESH.get(key, 0.75)
        return "✅" if score >= t else ("⚠️ " if score >= t-0.1 else "❌")

    def _fmt(score):
        return f"{score:.4f}" if score is not None else "  N/A "

    print("╔" + "═"*60 + "╗")
    print("║    KẾT QUẢ DEEPEVAL — CHATBOT HAUI                       ║")
    print("║    Judge: Ollama qwen2.5:7b (local, không cần API key)   ║")
    print("╠" + "═"*60 + "╣")
    print(f"║  {_status(avg_faith,'faithfulness')}  Faithfulness (Trung thực)      "
          f"{_fmt(avg_faith)}  [{_bar(avg_faith or 0)}] ║")
    print(f"║  {_status(avg_rel,'answer_relevancy')}  Answer Relevancy (Liên quan)  "
          f"{_fmt(avg_rel)}  [{_bar(avg_rel or 0)}] ║")
    print(f"║  {_status(avg_recall,'context_recall')}  Context Recall (Đầy đủ ctx)  "
          f"{_fmt(avg_recall)}  [{_bar(avg_recall or 0)}] ║")
    print(f"║  📊  Keyword Hit Rate            {avg_kw:.4f}                      ║")
    print(f"║  ⏱   Latency chatbot TB          {avg_lat}ms                         ║")
    print(f"║  📝  Số câu đánh giá             {len(scored)}                          ║")
    print("╚" + "═"*60 + "╝")

    print("\n── Keyword hit rate theo danh mục ──")
    for cat, kw in sorted(cat_kw_avg.items(), key=lambda x: -x[1]):
        bar    = "▓"*int(kw*10) + "░"*(10-int(kw*10))
        status = "✅" if kw >= 0.8 else ("⚠️ " if kw >= 0.6 else "❌")
        print(f"  {status} {cat:<25} {kw:.1%}  [{bar}]")

    worst = sorted(scored, key=lambda r: r["kw_hit_rate"])[:5]
    if worst:
        print("\n── Top 5 câu thiếu keywords nhất ──")
        for r in worst:
            f_s = f"{r['faithfulness']:.2f}" if r.get("faithfulness") is not None else "N/A"
            r_s = f"{r['answer_relevancy']:.2f}" if r.get("answer_relevancy") is not None else "N/A"
            print(f"  [{r['id']}] {r['question'][:55]}...")
            print(f"         miss    : {r['keywords_miss']}")
            print(f"         faith={f_s} rel={r_s}")
            print(f"         answer  : {r['answer'][:80]}...")

    result = {
        "timestamp"            : datetime.now().isoformat(),
        "judge_model"          : llm.model,
        "framework"            : "deepeval",
        "total_questions"      : len(scored),
        "category_filter"      : category or "all",
        "avg_faithfulness"     : avg_faith,
        "avg_answer_relevancy" : avg_rel,
        "avg_context_recall"   : avg_recall,
        "keyword_hit_rate"     : avg_kw,
        "avg_latency_ms"       : avg_lat,
        "per_category_kw"      : cat_kw_avg,
        "detail"               : [{
            "id"               : r["id"],
            "category"         : r["category"],
            "question"         : r["question"],
            "answer"           : r["answer"][:300],
            "kw_hit_rate"      : r["kw_hit_rate"],
            "keywords_miss"    : r["keywords_miss"],
            "latency_ms"       : r["latency_ms"],
            "faithfulness"     : r.get("faithfulness"),
            "answer_relevancy" : r.get("answer_relevancy"),
            "context_recall"   : r.get("context_recall"),
        } for r in scored],
    }

    if save:
        out_dir = ROOT / "tests" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag  = f"_{category}" if category else ""
        fj   = out_dir / f"eval_deepeval{tag}_{ts}.json"
        fc   = out_dir / f"eval_deepeval{tag}_{ts}.csv"
        fj.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ JSON: {fj}")
        try:
            import csv
            with open(fc, "w", newline="", encoding="utf-8-sig") as f:
                fields = ["id","category","question","kw_hit_rate","keywords_miss",
                          "latency_ms","faithfulness","answer_relevancy","context_recall","answer"]
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(result["detail"])
            print(f"✅ CSV : {fc}")
        except Exception as e:
            print(f"⚠ CSV lỗi: {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepEval RAG Evaluation — Ollama judge, không cần API key"
    )
    parser.add_argument("--quick",  action="store_true", help="Chạy 15 câu đầu")
    parser.add_argument("--cat",    type=str, default="", help="Chỉ 1 danh mục")
    parser.add_argument("--save",   action="store_true", help="Lưu JSON + CSV")
    parser.add_argument("--stats",  action="store_true", help="Thống kê dataset")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        run_evaluation(
            quick    = args.quick,
            category = args.cat,
            save     = args.save,
        )