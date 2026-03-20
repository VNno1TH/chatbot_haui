"""
evaluate_ragas_full.py
Chạy đánh giá RAGAs đầy đủ cho chatbot tuyển sinh HaUI.

Dùng bộ test 152 câu từ haui_ragas_dataset.py — được xây dựng trực tiếp
từ toàn bộ file .md và .json trong dự án.

Metrics:
  - faithfulness       : câu trả lời có trung thực với context?
  - answer_relevancy   : câu trả lời có liên quan câu hỏi?
  - context_precision  : context retrieve có chính xác không? (chỉ RAG intents)
  - context_recall     : context có đủ để trả lời không?     (chỉ RAG intents)

Cài đặt:
    pip install ragas langchain-openai langchain-huggingface datasets
    # Không cần API key — Ollama local qua OpenAI-compatible endpoint

Chạy:
    python tests/evaluate_ragas_full.py                      # 152 câu
    python tests/evaluate_ragas_full.py --quick              # 15 câu đầu
    python tests/evaluate_ragas_full.py --cat diem_chuan     # 1 danh mục
    python tests/evaluate_ragas_full.py --save               # lưu JSON + CSV
    python tests/evaluate_ragas_full.py --cat edge --save    # danh mục edge

Kết quả lưu tại: tests/results/ragas_YYYYMMDD_HHMMSS.json
"""

import sys, os, json, time, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from haui_ragas_dataset import EVAL_DATASET, print_stats

# ── Các intents dùng JSON query (không phải RAG chunk) ────────────────────────
JSON_INTENTS = {
    "JSON_DIEM_CHUAN", "JSON_HOC_PHI", "JSON_CHI_TIEU_TO_HOP",
    "JSON_QUY_DOI_DIEM", "JSON_DAU_TRUOT",
}


# ── Setup RAGAs ───────────────────────────────────────────────────────────────

def _setup_ragas():
    """
    Khởi tạo RAGAs với Ollama làm judge qua OpenAI-compatible endpoint.

    Ollama >= 0.1.24 expose /v1/chat/completions — RAGAs dùng
    langchain-openai trỏ vào đó, không cần langchain-ollama.

    Cài đặt:
        pip install ragas langchain-openai langchain-huggingface datasets

    Kiểm tra Ollama hỗ trợ OpenAI endpoint:
        curl http://localhost:11435/v1/models
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        import ragas
        print(f"  RAGAs version: {ragas.__version__}")
        from ragas import evaluate
    except ImportError:
        print("❌ Cần cài: pip install ragas")
        sys.exit(1)

    ollama_url   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11435")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
    # Ollama OpenAI-compatible base URL: thêm /v1
    openai_base  = ollama_url.rstrip("/") + "/v1"

    # ── Kiểm tra Ollama có hỗ trợ /v1 endpoint không ──────────────────────────
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(f"{openai_base}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode()
            if ollama_model.split(":")[0] not in body:
                print(f"  ⚠ Model '{ollama_model}' không thấy trong /v1/models")
                print(f"    Kiểm tra: curl {openai_base}/models")
            else:
                print(f"  ✓ Ollama /v1 endpoint OK — model '{ollama_model}' sẵn sàng")
    except urllib.error.URLError as e:
        print(f"❌ Không kết nối Ollama tại {openai_base}: {e}")
        print(f"   Kiểm tra SSH tunnel: ssh -p 22156 -L 11435:localhost:11435 ...")
        sys.exit(1)

    # ── LLM judge: llm_factory (RAGAs 0.4.x) trỏ vào Ollama /v1 ─────────────
    llm = None
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory

        _client = OpenAI(
            base_url = openai_base,
            api_key  = "ollama",   # Ollama không check key, cần string bất kỳ
            timeout  = 120,
        )
        llm = llm_factory(ollama_model, client=_client)
        print(f"  ✓ RAGAs LLM judge: llm_factory → {ollama_model}")
        print(f"    Endpoint: {openai_base}")
    except ImportError:
        print("❌ Cần cài: pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Khởi tạo LLM judge thất bại: {e}")
        sys.exit(1)

    # ── Embeddings: dùng HuggingFaceEmbeddings của RAGAs 0.4.x ──────────────
    # RAGAs embedding_factory không hỗ trợ Ollama trực tiếp
    # → dùng HuggingFaceEmbeddings với model local (đã có sẵn từ Retriever)
    embeddings = None
    try:
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
        local_model = ROOT / "models" / "multilingual-e5-small"
        model_name  = str(local_model) if (local_model / "config.json").exists() \
                      else "intfloat/multilingual-e5-small"
        embeddings  = RagasHFEmbeddings(model_name=model_name)
        print(f"  ✓ RAGAs Embeddings: HuggingFace multilingual-e5-small (local)")
    except Exception as e:
        print(f"  ⚠ Embeddings lỗi: {e} → answer_relevancy sẽ bị bỏ qua")

    # ── Metrics (RAGAs 0.4.x) ─────────────────────────────────────────────────
    try:
        from ragas.metrics.collections import (Faithfulness, AnswerRelevancy,
                                               ContextPrecision, ContextRecall)
    except ImportError:
        from ragas.metrics import (Faithfulness, AnswerRelevancy,
                                   ContextPrecision, ContextRecall)

    faithfulness_m      = Faithfulness(llm=llm)
    answer_relevancy_m  = AnswerRelevancy(llm=llm, embeddings=embeddings) \
                          if embeddings else None
    context_precision_m = ContextPrecision(llm=llm)
    context_recall_m    = ContextRecall(llm=llm)

    metrics = [m for m in [
        faithfulness_m,
        answer_relevancy_m,
        context_precision_m,
        context_recall_m,
    ] if m is not None]
    print(f"  ✓ RAGAs metrics: {[m.name for m in metrics]}")

    return evaluate, metrics


# ── Thu thập answers và contexts từ chatbot pipeline ─────────────────────────

def _collect_pipeline_data(dataset: list, verbose: bool = True) -> list:
    """
    Chạy từng câu hỏi qua pipeline chatbot, thu thập:
      - answer    : câu trả lời thực tế
      - contexts  : danh sách đoạn văn bản dùng để trả lời
      - latency   : thời gian phản hồi
    """
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
                  f"{item['question'][:52]}...")

        t0 = time.perf_counter()

        # ── Lấy câu trả lời ───────────────────────────────────────────────────
        try:
            t_chat  = time.perf_counter()
            resp    = bot.chat(item["question"])
            answer  = resp.answer
            ctx_raw = resp.context or ""
            t_chat_ms = int((time.perf_counter() - t_chat) * 1000)
            bot.reset()
        except Exception as e:
            answer    = f"ERROR: {e}"
            ctx_raw   = ""
            t_chat_ms = 0

        latency_ms = int((time.perf_counter() - t0) * 1000)
        if True:
            print(f"         → chat(): {t_chat_ms}ms")

        # ── Lấy contexts ──────────────────────────────────────────────────────
        # JSON intents: context là chuỗi kết quả tính toán → dùng làm context
        # RAG intents : retrieve trực tiếp để lấy chunks
        contexts = []
        try:
            if item["intent"] in JSON_INTENTS:
                # JSON context: chia theo --- hoặc dòng trống
                if ctx_raw and ctx_raw not in ("__GREETING__", "__OFF_TOPIC__"):
                    parts = [p.strip() for p in ctx_raw.replace("__CLARIFY__","").split("---") if p.strip()]
                    contexts = parts[:4] if parts else [ctx_raw[:800]]
                else:
                    contexts = [answer]   # fallback
            else:
                # RAG context: retrieve từ ChromaDB
                chunks = retriever.retrieve(item["question"], top_k=4)
                if chunks:
                    contexts = [c.text for c in chunks]
                elif ctx_raw and ctx_raw not in ("__GREETING__", "__OFF_TOPIC__"):
                    contexts = [ctx_raw[:800]]
                else:
                    contexts = ["Không tìm thấy thông tin liên quan."]
        except Exception:
            contexts = [answer]

        if not contexts:
            contexts = ["Không có thông tin context."]

        # ── Tính keyword hit rate ──────────────────────────────────────────────
        kw_hit  = [k for k in item["keywords"] if k.lower() in answer.lower()]
        kw_miss = [k for k in item["keywords"] if k.lower() not in answer.lower()]

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
            "kw_hit_rate"  : round(len(kw_hit) / len(item["keywords"]), 2) if item["keywords"] else 1.0,
            "latency_ms"   : latency_ms,
            "intent"       : item["intent"],
        })

        if verbose:
            hit = rows[-1]["kw_hit_rate"]
            bar = "▓" * int(hit * 10) + "░" * (10 - int(hit * 10))
            print(f"         {latency_ms:>5}ms | kw [{bar}] {hit:.0%} | "
                  f"miss: {kw_miss if kw_miss else '—'}")

        time.sleep(0.15)   # tránh rate limit Groq

    return rows


# ── Chạy RAGAs evaluation ─────────────────────────────────────────────────────

def run_evaluation(
    quick      : bool = False,
    category   : str  = "",
    save       : bool = False,
    batch_size : int  = 10,
) -> dict:
    print("=" * 70)
    print("  RAGAs Full Evaluation — Chatbot Tuyển sinh HaUI")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Lọc dataset ───────────────────────────────────────────────────────────
    ds = EVAL_DATASET
    if category:
        ds = [d for d in ds if d["category"] == category]
        if not ds:
            print(f"❌ Không tìm thấy danh mục '{category}'")
            valid = sorted(set(d["category"] for d in EVAL_DATASET))
            print(f"   Danh mục hợp lệ: {valid}")
            return {}
        print(f"  Danh mục: {category} ({len(ds)} câu)")

    if quick:
        ds = ds[:15]
        print(f"  Quick mode: {len(ds)} câu")

    print(f"  Tổng: {len(ds)} câu hỏi\n")

    # ── Bước 1: Setup ─────────────────────────────────────────────────────────
    print("── Bước 1: Khởi tạo RAGAs ──")
    evaluate_fn, metrics = _setup_ragas()
    print()

    # ── Bước 2: Thu thập data ─────────────────────────────────────────────────
    print("── Bước 2: Thu thập câu trả lời từ pipeline ──")
    rows = _collect_pipeline_data(ds, verbose=True)

    # ── Tổng kết keyword hit rate ─────────────────────────────────────────────
    avg_kw = sum(r["kw_hit_rate"] for r in rows) / len(rows)
    print(f"\n  Keyword hit rate trung bình: {avg_kw:.1%}")
    print(f"  Latency trung bình: {sum(r['latency_ms'] for r in rows)//len(rows)}ms\n")

    # ── Bước 3: Chạy RAGAs ───────────────────────────────────────────────────
    print("── Bước 3: Chạy RAGAs metrics (Claude Haiku làm judge) ──\n")

    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        print("❌ Cần cài: pip install datasets")
        sys.exit(1)

    # Phân tách RAG intents (cần context metrics) và JSON intents
    rag_rows  = [r for r in rows if r["intent"] not in JSON_INTENTS]
    json_rows = [r for r in rows if r["intent"] in JSON_INTENTS]
    print(f"  RAG rows : {len(rag_rows)} câu → faithfulness, relevancy, precision, recall")
    print(f"  JSON rows: {len(json_rows)} câu → faithfulness, relevancy\n")

    try:
        from ragas.metrics.collections import (Faithfulness, AnswerRelevancy,
                                               ContextPrecision, ContextRecall)
    except ImportError:
        from ragas.metrics import (Faithfulness, AnswerRelevancy,
                                   ContextPrecision, ContextRecall)
    m_rag = [m for m in metrics if isinstance(m, (ContextPrecision, ContextRecall))]

    def _hf(row_list):
        data = [{
            "user_input"         : r["question"],
            "response"           : r["answer"],
            "retrieved_contexts" : r["contexts"],
            "reference"          : r["ground_truth"],
        } for r in row_list]
        try:
            from ragas import EvaluationDataset
            return EvaluationDataset.from_list(data)
        except (ImportError, Exception):
            return HFDataset.from_list([{
                "question"    : r["question"],
                "answer"      : r["answer"],
                "contexts"    : r["contexts"],
                "ground_truth": r["ground_truth"],
            } for r in row_list])

    scores_all = {}   # faithfulness, answer_relevancy
    scores_rag = {}   # context_precision, context_recall

    def _run_batch(hf_data, metric_list, label=""):
        result_scores = {m.name: [] for m in metric_list}
        total = len(hf_data)
        for i in range(0, total, batch_size):
            end   = min(i + batch_size, total)
            batch = hf_data.select(range(i, end))
            print(f"  {label} Batch {i//batch_size + 1}: câu {i+1}–{end}...", end=" ", flush=True)
            try:
                result = evaluate_fn(batch, metrics=metric_list)
                # Xử lý cả ragas cũ và mới
                try:
                    df = result.to_pandas()
                    for m in metric_list:
                        if m.name in df.columns:
                            vals = df[m.name].dropna().tolist()
                            result_scores[m.name].extend([float(v) for v in vals])
                    avgs = {m.name: round(sum(result_scores[m.name][-len(batch):]) /
                                         len(batch), 3)
                            for m in metric_list if result_scores[m.name]}
                    print(f"✓ {avgs}")
                except AttributeError:
                    for m in metric_list:
                        val = result.get(m.name) if hasattr(result, "get") else None
                        if val is not None:
                            result_scores[m.name].append(float(val))
                    print("✓")
            except Exception as e:
                print(f"⚠ Lỗi batch: {e}")
                # Retry từng câu một
                for j in range(i, end):
                    try:
                        single = hf_data.select([j])
                        r      = evaluate_fn(single, metrics=metric_list)
                        try:
                            df = r.to_pandas()
                            for m in metric_list:
                                if m.name in df.columns:
                                    vals = df[m.name].dropna().tolist()
                                    result_scores[m.name].extend([float(v) for v in vals])
                        except AttributeError:
                            pass
                    except Exception:
                        pass
        return result_scores

    # Chạy trên tất cả rows (faithfulness + relevancy)
    if rows:
        print("  [All rows] Chạy faithfulness + answer_relevancy...")
        scores_all = _run_batch(_hf(rows), m_all, "[all]")
        print()

    # Chạy trên RAG rows (context_precision + context_recall)
    if rag_rows:
        print("  [RAG rows] Chạy context_precision + context_recall...")
        scores_rag = _run_batch(_hf(rag_rows), m_rag, "[rag]")
        print()

    # ── Bước 4: Tổng hợp kết quả ─────────────────────────────────────────────
    print("── Bước 4: Tổng hợp kết quả ──\n")

    all_scores = {**scores_all, **scores_rag}
    avg_scores = {}
    for m_name, vals in all_scores.items():
        if vals:
            avg_scores[m_name] = round(sum(vals) / len(vals), 4)

    # Tính keyword hit rate theo danh mục
    cat_kw = {}
    for r in rows:
        cat = r["category"]
        if cat not in cat_kw:
            cat_kw[cat] = []
        cat_kw[cat].append(r["kw_hit_rate"])
    cat_kw_avg = {c: round(sum(v)/len(v), 3) for c, v in cat_kw.items()}

    # ── In báo cáo ────────────────────────────────────────────────────────────
    print("╔" + "═" * 50 + "╗")
    print("║         KẾT QUẢ ĐÁNH GIÁ RAGAs CHATBOT HaUI         ║")
    print("╠" + "═" * 50 + "╣")

    METRIC_VI = {
        "faithfulness"      : "Độ trung thực (Faithfulness)",
        "answer_relevancy"  : "Độ liên quan (Answer Relevancy)",
        "context_precision" : "Độ chính xác context",
        "context_recall"    : "Độ đầy đủ context",
    }
    METRIC_THRESHOLD = {
        "faithfulness"      : 0.85,
        "answer_relevancy"  : 0.80,
        "context_precision" : 0.75,
        "context_recall"    : 0.75,
    }

    for m_name, score in avg_scores.items():
        label    = METRIC_VI.get(m_name, m_name)
        thresh   = METRIC_THRESHOLD.get(m_name, 0.75)
        status   = "✅" if score >= thresh else ("⚠️ " if score >= thresh - 0.1 else "❌")
        bar      = "▓" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"║  {status} {label:<28} {score:.4f}  [{bar}] ║")

    print(f"║  📊 Keyword hit rate (avg)       {avg_kw:.4f}                  ║")
    print(f"║  ⏱  Latency trung bình           "
          f"{sum(r['latency_ms'] for r in rows)//len(rows)}ms                     ║")
    print(f"║  📝 Số câu đánh giá              {len(rows)}                      ║")
    print("╚" + "═" * 50 + "╝")

    print("\n── Keyword hit rate theo danh mục ──")
    for cat, kw in sorted(cat_kw_avg.items(), key=lambda x: -x[1]):
        bar    = "▓" * int(kw * 10) + "░" * (10 - int(kw * 10))
        status = "✅" if kw >= 0.8 else ("⚠️ " if kw >= 0.6 else "❌")
        print(f"  {status} {cat:<25} {kw:.1%}  [{bar}]")

    # ── Câu trả lời có keyword miss nhiều nhất ────────────────────────────────
    worst = sorted(rows, key=lambda r: r["kw_hit_rate"])[:5]
    if worst:
        print("\n── Top 5 câu trả lời thiếu keywords nhất ──")
        for r in worst:
            print(f"  [{r['id']}] {r['question'][:55]}...")
            print(f"         miss: {r['keywords_miss']}")
            print(f"         answer[:80]: {r['answer'][:80]}...")

    # ── Lưu kết quả ───────────────────────────────────────────────────────────
    result = {
        "timestamp"     : datetime.now().isoformat(),
        "total_questions": len(rows),
        "category_filter": category or "all",
        "ragas_scores"  : avg_scores,
        "keyword_hit_rate": avg_kw,
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) // len(rows),
        "per_category_kw": cat_kw_avg,
        "detail"        : [{
            "id"          : r["id"],
            "category"    : r["category"],
            "question"    : r["question"],
            "answer"      : r["answer"][:300],
            "kw_hit_rate" : r["kw_hit_rate"],
            "keywords_miss": r["keywords_miss"],
            "latency_ms"  : r["latency_ms"],
        } for r in rows],
    }

    if save:
        out_dir = ROOT / "tests" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        cat_tag  = f"_{category}" if category else ""
        out_file = out_dir / f"ragas_full{cat_tag}_{ts}.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ Đã lưu kết quả: {out_file}")

        # Lưu thêm CSV cho dễ đọc
        try:
            import csv
            csv_file = out_dir / f"ragas_full{cat_tag}_{ts}.csv"
            with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "id", "category", "question", "kw_hit_rate",
                    "keywords_miss", "latency_ms", "answer"
                ])
                writer.writeheader()
                writer.writerows(result["detail"])
            print(f"✅ Đã lưu CSV: {csv_file}")
        except Exception as e:
            print(f"⚠ Không lưu được CSV: {e}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAs Evaluation — Chatbot Tuyển sinh HaUI (152 câu)"
    )
    parser.add_argument("--quick",  action="store_true",
                        help="Chỉ chạy 15 câu đầu (test nhanh)")
    parser.add_argument("--cat",    type=str, default="",
                        help="Chỉ chạy 1 danh mục (vd: diem_chuan, hoc_phi, edge...)")
    parser.add_argument("--save",   action="store_true",
                        help="Lưu kết quả JSON + CSV vào tests/results/")
    parser.add_argument("--batch",  type=int, default=10,
                        help="Số câu mỗi batch gọi RAGAs (mặc định 10)")
    parser.add_argument("--stats",  action="store_true",
                        help="Chỉ in thống kê dataset, không chạy eval")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        run_evaluation(
            quick      = args.quick,
            category   = args.cat,
            save       = args.save,
            batch_size = args.batch,
        )