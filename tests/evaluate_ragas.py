"""
evaluate_ragas.py
Đánh giá chatbot tuyển sinh HaUI bằng framework RAGAs chính thức.

Tái dụng bộ test từ evaluate_rag.py (v2 — 90 câu), bổ sung:
  - contexts  : các chunk được retrieve từ RAG pipeline
  - ground_truth: expected answer từ dataset gốc

Metrics RAGAs được dùng:
  - faithfulness       : câu trả lời có trung thực với context không?
  - answer_relevancy   : câu trả lời có liên quan câu hỏi không?
  - context_precision  : context retrieve có chính xác không?
  - context_recall     : context có đủ để trả lời không?

Cài đặt:
    pip install ragas langchain-anthropic langchain-community

Chạy:
    python tests/evaluate_ragas.py                    # tất cả 90 câu
    python tests/evaluate_ragas.py --quick            # 10 câu đầu (test nhanh)
    python tests/evaluate_ragas.py --cat diem_chuan   # 1 danh mục
    python tests/evaluate_ragas.py --save             # lưu kết quả
    python tests/evaluate_ragas.py --compare          # so sánh với kết quả cũ
"""

import sys, json, time, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Dataset — tái dụng từ evaluate_rag.py ────────────────────────────────────
# Import trực tiếp bằng path để tránh lỗi ModuleNotFoundError
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("evaluate_rag", ROOT / "tests" / "evaluate_rag.py")
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EVAL_DATASET = _mod.EVAL_DATASET


# ── RAGAs setup ───────────────────────────────────────────────────────────────

def _setup_ragas():
    """
    Khởi tạo RAGAs 0.4.x với Anthropic Claude + HuggingFace embeddings.
    ragas 0.4: dùng llm_factory / HuggingFaceEmbeddings từ ragas.embeddings
    """
    try:
        import ragas
        from ragas import evaluate
        print(f"RAGAs version: {ragas.__version__}")
    except ImportError:
        print("❌ Cần cài: pip install ragas")
        sys.exit(1)

    # ── Metrics — ragas 0.4 dùng class instance ───────────────────────────────
    try:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
        faithfulness       = Faithfulness()
        answer_relevancy   = AnswerRelevancy()
        context_precision  = ContextPrecision()
        context_recall     = ContextRecall()
    except ImportError:
        # fallback ragas cũ hơn dùng singleton
        try:
            from ragas.metrics.collections import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )
        except ImportError:
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # ── LLM — dùng ragas llm_factory với Anthropic ────────────────────────────
    try:
        import anthropic as _anth
        from ragas.llms import llm_factory

        # ragas 0.4 hỗ trợ anthropic qua litellm
        try:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            llm = llm_factory(
                "claude-haiku-4-5-20251001",
                run_config=None,
            )
            print("✓ RAGAs LLM: llm_factory (native)")
        except Exception:
            raise ImportError("llm_factory không hỗ trợ anthropic trực tiếp")

    except Exception:
        # Fallback: LangchainLLMWrapper
        try:
            from langchain_anthropic import ChatAnthropic
            from ragas.llms import LangchainLLMWrapper
            _llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                temperature=0,
                max_tokens=2048,
            )
            llm = LangchainLLMWrapper(_llm)
            print("✓ RAGAs LLM: LangchainLLMWrapper (fallback)")
        except ImportError:
            print("❌ Cần cài: pip install langchain-anthropic")
            sys.exit(1)

    # ── Embeddings ─────────────────────────────────────────────────────────────
    model_dir  = ROOT / "models" / "bge-m3"
    model_name = str(model_dir) if (model_dir / "config.json").exists() else "BAAI/bge-m3"

    try:
        # ragas 0.4 có HuggingFaceEmbeddings riêng
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmb
        embeddings = RagasHFEmb(model_name=model_name)
        print(f"✓ RAGAs Embeddings: ragas.embeddings.HuggingFaceEmbeddings")
    except (ImportError, Exception):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings as LCEmb
            from ragas.embeddings import LangchainEmbeddingsWrapper
            _emb   = LCEmb(model_name=model_name, model_kwargs={"device": "cpu"},
                           encode_kwargs={"normalize_embeddings": True})
            embeddings = LangchainEmbeddingsWrapper(_emb)
            print(f"✓ RAGAs Embeddings: LangchainEmbeddingsWrapper")
        except Exception as e:
            print(f"⚠ Không load được embeddings: {e} — answer_relevancy có thể không chạy")
            embeddings = None

    # ── Gán LLM + Embeddings vào metrics ──────────────────────────────────────
    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = llm
        if hasattr(m, "embeddings") and embeddings is not None:
            m.embeddings = embeddings

    return evaluate, metrics


# ── Thu thập data từ pipeline ─────────────────────────────────────────────────

def _collect_pipeline_data(dataset: list, verbose: bool = True) -> list:
    """
    Chạy pipeline chatbot để lấy:
      - answer    : câu trả lời thực tế
      - contexts  : các chunk được retrieve (dùng cho RAGAs)

    Returns:
        list of {question, answer, contexts, ground_truth, id, category, latency_ms}
    """
    from src.pipeline.chatbot  import Chatbot
    from src.pipeline.router   import IntentType
    from src.retrieval.retriever import Retriever

    print("Khởi tạo pipeline...")
    retriever = Retriever(use_reranker=False, use_bm25=True)
    bot       = Chatbot(retriever=retriever)
    print("✓ Sẵn sàng\n")

    rows = []
    for i, item in enumerate(dataset, 1):
        if verbose:
            print(f"[{i:02d}/{len(dataset)}] [{item['id']}] {item['question'][:55]}...")

        t0 = time.perf_counter()

        # ── Lấy answer từ chatbot ──────────────────────────────────────────
        try:
            resp   = bot.chat(item["question"])
            answer = resp.answer
            intent = resp.intent
            bot.reset()
        except Exception as e:
            answer = f"ERROR: {e}"
            intent = IntentType.UNKNOWN

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # ── Lấy contexts từ retriever ──────────────────────────────────────
        # RAGAs cần contexts là list[str] — các đoạn văn bản được dùng để trả lời
        contexts = []
        try:
            # Với JSON intents, context là kết quả tính toán — không phải RAG chunk
            # Vẫn lưu context string từ chatbot để RAGAs đánh giá faithfulness
            if hasattr(bot, "_ctx_builder"):
                from src.pipeline.router import _extract_entities_rule, _rule_match, _claude_classify
                import anthropic as _anth
                # Lấy context đã build từ last response
                last = getattr(bot, "_last_response", None)
                if last and last.context and last.context not in (
                    "__GREETING__", "__OFF_TOPIC__", "__CANCELLED__"
                ):
                    # Chia context thành các đoạn ngắn hơn cho RAGAs
                    ctx_text = last.context.replace("__CLARIFY__", "").strip()
                    # Split theo dấu phân cách ---
                    parts = [p.strip() for p in ctx_text.split("---") if p.strip()]
                    contexts = parts if parts else [ctx_text[:1000]]
                else:
                    # Retrieve trực tiếp cho câu hỏi này
                    chunks = retriever.retrieve(item["question"], top_k=3)
                    contexts = [c.text for c in chunks] if chunks else ["Không có context"]
        except Exception:
            contexts = [answer]  # Fallback: dùng answer làm context

        if not contexts:
            contexts = ["Không có thông tin context"]

        rows.append({
            "id"          : item["id"],
            "category"    : item["category"],
            "question"    : item["question"],
            "answer"      : answer,
            "contexts"    : contexts,
            "ground_truth": item["expected"],
            "latency_ms"  : latency_ms,
            "keywords_hit": [k for k in item["keywords"] if k.lower() in answer.lower()],
            "keywords_miss": [k for k in item["keywords"] if k.lower() not in answer.lower()],
        })

        if verbose:
            hit_rate = len(rows[-1]["keywords_hit"]) / len(item["keywords"]) if item["keywords"] else 1.0
            print(f"         ✓ {latency_ms}ms | keywords: {len(rows[-1]['keywords_hit'])}/{len(item['keywords'])} | intent: {intent.value if hasattr(intent,'value') else intent}")

        time.sleep(0.2)  # Tránh rate limit

    return rows


# ── Chạy RAGAs evaluation ─────────────────────────────────────────────────────

def run_ragas_evaluation(
    quick   : bool = False,
    save    : bool = False,
    category: str  = "",
    compare : bool = False,
    batch   : int  = 10,
) -> dict:
    print("=" * 70)
    print("  RAGAs Evaluation — Chatbot Tuyển sinh HaUI")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Lọc dataset ───────────────────────────────────────────────────────────
    ds = EVAL_DATASET
    if category:
        ds = [d for d in ds if d["category"] == category]
        print(f"Danh mục: {category} ({len(ds)} câu)")
    if quick:
        ds = ds[:10]
        print(f"Quick mode: {len(ds)} câu")
    print(f"Tổng: {len(ds)} câu hỏi\n")

    # ── Thu thập data từ pipeline ─────────────────────────────────────────────
    print("── Bước 1: Thu thập câu trả lời từ pipeline ──")
    rows = _collect_pipeline_data(ds, verbose=True)

    # ── Chạy RAGAs ────────────────────────────────────────────────────────────
    print("\n── Bước 2: Chạy RAGAs metrics ──")
    print("(Gọi Claude Haiku làm judge — có thể mất vài phút)\n")

    evaluate_fn, metrics = _setup_ragas()

    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        print("❌ Cần cài: pip install datasets")
        sys.exit(1)

    # RAGAs nhận HuggingFace Dataset
    # Tách RAG intents và JSON intents
    # context_precision/recall chỉ có nghĩa với RAG intents
    # (JSON intents dùng context là kết quả tính toán, không phải chunks)
    # JSON_INTENTS: các intent dùng JSON query, không phải RAG chunks
    # context_precision/recall không có nghĩa với các intent này
    JSON_INTENTS = {
        "JSON_DIEM_CHUAN", "JSON_HOC_PHI", "JSON_CHI_TIEU_TO_HOP",
        "JSON_QUY_DOI_DIEM", "JSON_DAU_TRUOT",
        "GREETING", "OFF_TOPIC",
    }
    rag_rows  = [r for r in rows if r.get("intent","UNKNOWN") not in JSON_INTENTS]
    all_rows  = rows  # faithfulness + answer_relevancy áp dụng cho tất cả

    print(f"  RAG intents: {len(rag_rows)}/{len(rows)} câu → dùng cho context metrics")
    print(f"  JSON intents: {len(rows)-len(rag_rows)}/{len(rows)} câu → chỉ faithfulness + relevancy")

    hf_all = HFDataset.from_list([
        {"question": r["question"], "answer": r["answer"],
         "contexts": r["contexts"], "ground_truth": r["ground_truth"]}
        for r in all_rows
    ])
    hf_rag = HFDataset.from_list([
        {"question": r["question"], "answer": r["answer"],
         "contexts": r["contexts"], "ground_truth": r["ground_truth"]}
        for r in rag_rows
    ]) if rag_rows else None

    # Phân nhóm metrics
    try:
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        metrics_all = [m for m in metrics if isinstance(m, (Faithfulness, AnswerRelevancy))]
        metrics_rag = [m for m in metrics if isinstance(m, (ContextPrecision, ContextRecall))]
    except Exception:
        metrics_all = metrics[:2]   # faithfulness, answer_relevancy
        metrics_rag = metrics[2:]   # context_precision, context_recall

    # Chạy từng batch để tránh timeout
    all_scores = {m.name: [] for m in metrics}
    metric_names = [m.name for m in metrics]

    print(f"Chạy {len(metrics_all)} metrics trên {len(all_rows)} câu (tất cả)...")
    for i in range(0, len(all_rows), batch):
        batch_data = hf_all.select(range(i, min(i + batch, len(all_rows))))
        print(f"  Batch {i//batch + 1}: câu {i+1}–{min(i+batch, len(all_rows))}...")
        try:
            result = evaluate_fn(batch_data, metrics=metrics_all)
            # ragas 0.4+: EvaluationResult — dùng to_pandas() để lấy scores
            try:
                df = result.to_pandas()
                for m_name in metric_names:
                    if m_name in df.columns:
                        vals = df[m_name].dropna().tolist()
                        all_scores[m_name].extend([float(v) for v in vals])
                        print(f"    {m_name}: {[round(v,3) for v in vals]}")
            except AttributeError:
                # ragas < 0.2: result là dict
                for m_name in metric_names:
                    val = result.get(m_name) if hasattr(result, "get") else None
                    if val is not None:
                        if hasattr(val, "__iter__") and not isinstance(val, float):
                            all_scores[m_name].extend(list(val))
                        else:
                            all_scores[m_name].append(float(val))
        except Exception as e:
            import traceback
            print(f"  ⚠ Batch lỗi: {e}")
            print("  Traceback:")
            traceback.print_exc()
            print("  → Thử chạy lại batch này với batch_size=1...")
            # Retry từng câu 1 để xác định câu nào gây lỗi
            for j in range(i, min(i + batch, len(rows))):
                try:
                    single = hf_data.select([j])
                    r2 = evaluate_fn(single, metrics=metrics)
                    try:
                        df2 = r2.to_pandas()
                        for m_name in metric_names:
                            if m_name in df2.columns:
                                vals = df2[m_name].dropna().tolist()
                                all_scores[m_name].extend([float(v) for v in vals])
                    except AttributeError:
                        for m_name in metric_names:
                            val = r2.get(m_name) if hasattr(r2, "get") else None
                            if val is not None:
                                all_scores[m_name].append(float(val) if isinstance(val, float) else float(list(val)[0]))
                    print(f"    câu {j+1} ✅")
                except Exception as e2:
                    print(f"    câu {j+1} ❌ {e2}")
                    # Điền 0.0 cho câu lỗi để không bỏ sót
                    for m_name in metric_names:
                        all_scores[m_name].append(0.0)

    # Chạy context_precision + context_recall chỉ trên RAG intents
    if hf_rag and metrics_rag:
        print(f"\nChạy context metrics trên {len(rag_rows)} RAG câu...")
        for i in range(0, len(rag_rows), batch):
            batch_data = hf_rag.select(range(i, min(i + batch, len(rag_rows))))
            print(f"  Batch {i//batch + 1}: câu {i+1}–{min(i+batch, len(rag_rows))}...")
            try:
                result = evaluate_fn(batch_data, metrics=metrics_rag)
                try:
                    df = result.to_pandas()
                    for m_name in [m.name for m in metrics_rag]:
                        if m_name in df.columns:
                            vals = df[m_name].dropna().tolist()
                            all_scores[m_name].extend([float(v) for v in vals])
                            print(f"    {m_name}: {[round(v,3) for v in vals]}")
                except AttributeError:
                    for m_name in [m.name for m in metrics_rag]:
                        val = result.get(m_name) if hasattr(result,"get") else None
                        if val is not None:
                            all_scores[m_name].append(float(val))
            except Exception as e2:
                import traceback; traceback.print_exc()
                for m_name in [m.name for m in metrics_rag]:
                    all_scores[m_name].append(0.0)

    # ── Kiểm tra kết quả ─────────────────────────────────────────────────────
    total_scored = sum(len(v) for v in all_scores.values())
    if total_scored == 0:
        print("\n❌ KHÔNG CÓ KẾT QUẢ RAGAs — tất cả batch đều lỗi.")
        print("   Gợi ý: kiểm tra ANTHROPIC_API_KEY và kết nối mạng.")
        print("   Vẫn hiển thị kết quả keyword-based bên dưới.")
    else:
        print(f"\n✅ Đã score {total_scored // len(metrics)} / {len(rows)} câu")

    # ── Tính điểm tổng kết ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KẾT QUẢ RAGAs EVALUATION")
    print("=" * 70)

    avg_scores = {}
    for m_name in metric_names:
        vals = all_scores[m_name]
        avg  = sum(vals) / len(vals) if vals else 0.0
        avg_scores[m_name] = round(avg, 4)

    # In bảng metrics chính
    print("\n📊 Metrics RAGAs (thang 0–1):\n")
    metric_explain = {
        "faithfulness"      : "Câu trả lời trung thực với context",
        "answer_relevancy"  : "Câu trả lời liên quan câu hỏi",
        "context_precision" : "Context retrieve chính xác",
        "context_recall"    : "Context đủ để trả lời",
    }
    ragas_avg = 0.0
    for m_name, score in avg_scores.items():
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        icon  = "✅" if score >= 0.7 else ("⚠️" if score >= 0.5 else "❌")
        label = metric_explain.get(m_name, m_name)
        print(f"  {icon} {m_name:<22} {bar} {score:.3f}  ({label})")
        ragas_avg += score
    ragas_avg /= len(avg_scores) if avg_scores else 1

    print(f"\n  {'─'*60}")
    print(f"  🏆 RAGAs Score Tổng hợp: {ragas_avg:.3f} / 1.000")

    # Keyword hit rate từ pipeline
    kw_total = sum(len(r["keywords_hit"]) + len(r["keywords_miss"]) for r in rows)
    kw_hit   = sum(len(r["keywords_hit"]) for r in rows)
    kw_rate  = kw_hit / kw_total if kw_total else 0
    print(f"  📌 Keyword Hit Rate     : {kw_rate:.3f} ({kw_hit}/{kw_total} keywords)")
    avg_latency = sum(r["latency_ms"] for r in rows) // len(rows)
    print(f"  ⚡ Latency TB           : {avg_latency}ms/câu")

    # Điểm theo danh mục
    print("\n📂 Điểm theo danh mục (keyword-based):\n")
    by_cat: dict[str, list] = {}
    for r in rows:
        kw = len(r["keywords_hit"]) / (len(r["keywords_hit"]) + len(r["keywords_miss"])) \
             if (r["keywords_hit"] or r["keywords_miss"]) else 1.0
        by_cat.setdefault(r["category"], []).append(kw)

    for cat, scores in sorted(by_cat.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        avg_cat = sum(scores) / len(scores)
        bar     = "█" * int(avg_cat * 10) + "░" * (10 - int(avg_cat * 10))
        icon    = "✅" if avg_cat >= 0.8 else ("⚠️" if avg_cat >= 0.5 else "❌")
        print(f"  {icon} {cat:<20} {bar} {avg_cat:.2f}  (n={len(scores)})")

    # So sánh với kết quả cũ (evaluate_rag.py)
    if compare:
        old_path = ROOT / "tests" / "eval_results.json"
        if old_path.exists():
            print("\n📈 So sánh với bộ test cũ (evaluate_rag.py):\n")
            old_data = json.loads(old_path.read_text(encoding="utf-8"))
            old_avg  = sum(d["score"] for d in old_data) / len(old_data)
            print(f"  Bộ test cũ (LLM judge)  : {old_avg:.3f}/1.0  ({len(old_data)} câu)")
            print(f"  RAGAs faithfulness       : {avg_scores.get('faithfulness', 0):.3f}/1.0")
            print(f"  RAGAs answer_relevancy   : {avg_scores.get('answer_relevancy', 0):.3f}/1.0")
            print(f"\n  → RAGAs đo lường đa chiều hơn (4 metrics vs 1 LLM judge)")
        else:
            print("\n⚠ Không tìm thấy eval_results.json để so sánh")

    # Câu trả lời kém nhất
    print("\n🔍 Top 5 câu keywords miss nhiều nhất:\n")
    worst = sorted(rows, key=lambda r: len(r["keywords_miss"]) / max(len(r["keywords_hit"]) + len(r["keywords_miss"]), 1), reverse=True)[:5]
    for r in worst:
        total_kw = len(r["keywords_hit"]) + len(r["keywords_miss"])
        rate     = len(r["keywords_hit"]) / total_kw if total_kw else 1.0
        print(f"  [{r['id']}] {r['question'][:50]}")
        print(f"         Keywords miss: {r['keywords_miss']} | hit rate: {rate:.0%}")

    # ── Lưu kết quả ───────────────────────────────────────────────────────────
    output = {
        "timestamp"     : datetime.now().isoformat(),
        "n_questions"   : len(rows),
        "ragas_scores"  : avg_scores,
        "ragas_avg"     : round(ragas_avg, 4),
        "keyword_rate"  : round(kw_rate, 4),
        "avg_latency_ms": avg_latency,
        "by_category"   : {cat: round(sum(s)/len(s), 4) for cat, s in by_cat.items()},
        "details"       : rows,
    }

    if save:
        out_path = ROOT / "tests" / "eval_ragas_results.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ Đã lưu: {out_path}")

    print("\n" + "=" * 70)
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAGAs Evaluation cho chatbot HaUI")
    ap.add_argument("--quick",   action="store_true", help="Chỉ chạy 10 câu đầu")
    ap.add_argument("--save",    action="store_true", help="Lưu kết quả JSON")
    ap.add_argument("--compare", action="store_true", help="So sánh với kết quả evaluate_rag.py cũ")
    ap.add_argument("--cat",     default="",
        help="Chạy 1 danh mục: truong/diem_chuan/hoc_phi/to_hop/tinh_toan/mo_ta_nganh/faq/dau_truot/edge")
    ap.add_argument("--batch",   type=int, default=10, help="Số câu mỗi batch RAGAs (default: 10)")
    args = ap.parse_args()

    result = run_ragas_evaluation(
        quick   = args.quick,
        save    = args.save,
        category= args.cat,
        compare = args.compare,
        batch   = args.batch,
    )

    print(f"\nRAGAs Score: {result['ragas_avg']:.3f}/1.000")
    print(f"Keyword Rate: {result['keyword_rate']:.3f}/1.000")