"""
evaluate_ragas_ollama.py — RAGAs 0.4.x + Ollama local
Gọi trực tiếp metric.score(**kwargs) thay vì evaluate()
vì evaluate() có bug validation với non-OpenAI providers.

Signatures đúng của từng metric:
  Faithfulness.ascore (user_input, response, retrieved_contexts)
  ContextPrecision.ascore(user_input, reference, retrieved_contexts)
  ContextRecall.ascore  (user_input, retrieved_contexts, reference)

Chạy:
    python tests/evaluate_ragas_ollama.py --quick --save
    python tests/evaluate_ragas_ollama.py --cat diem_chuan --save
    python tests/evaluate_ragas_ollama.py --save
    python tests/evaluate_ragas_ollama.py --stats
"""

import sys, os, json, time, argparse, warnings, math
warnings.filterwarnings("ignore")

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


def _setup_metrics():
    from dotenv import load_dotenv
    load_dotenv()

    import ragas
    print(f"  RAGAs version: {ragas.__version__}")

    ollama_url   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11435")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    openai_base  = ollama_url.rstrip("/") + "/v1"

    # Kiểm tra litellm proxy port 4000
    import urllib.request
    for url, name in [
        ("http://localhost:4000/health", "litellm proxy :4000"),
        (f"{openai_base}/models", f"Ollama :{ollama_url.split(':')[-1]}"),
    ]:
        try:
            with urllib.request.urlopen(url, timeout=5):
                print(f"  ✓ {name} OK")
                judge_url = "http://localhost:4000" if "4000" in url else openai_base
                break
        except Exception:
            print(f"  ⚠ {name} không kết nối")
            judge_url = None

    if not judge_url:
        print("  ❌ Không kết nối được judge LLM")
        sys.exit(1)

    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        base_url = judge_url if "4000" in judge_url else openai_base,
        api_key  = "ollama",
        timeout  = 120,
    )
    llm = llm_factory(ollama_model, client=client)
    print(f"  ✓ LLM judge: {ollama_model} @ {judge_url}")

    from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall
    f  = Faithfulness(llm=llm)
    cp = ContextPrecision(llm=llm)
    cr = ContextRecall(llm=llm)

    print(f"  ✓ Metrics: faithfulness, context_precision, context_recall")
    return f, cp, cr


def _score_one(f, cp, cr, row: dict) -> dict:
    """
    Gọi trực tiếp metric.score() với đúng kwargs theo signature.
    Faithfulness : user_input, response, retrieved_contexts
    ContextPrecision: user_input, reference, retrieved_contexts
    ContextRecall   : user_input, retrieved_contexts, reference
    """
    q   = row["question"]
    ans = row["answer"]
    ctx = row["contexts"]
    ref = row["ground_truth"]

    scores = {}

    # Faithfulness
    try:
        r = f.score(user_input=q, response=ans, retrieved_contexts=ctx)
        scores["faithfulness"] = round(float(r.value), 3) if not math.isnan(r.value) else None
    except Exception as e:
        scores["faithfulness"] = None
        print(f"\n    ⚠ faithfulness: {e}")

    # ContextPrecision (chỉ RAG intents)
    if row["intent"] not in JSON_INTENTS:
        try:
            r = cp.score(user_input=q, reference=ref, retrieved_contexts=ctx)
            scores["context_precision"] = round(float(r.value), 3) if not math.isnan(r.value) else None
        except Exception as e:
            scores["context_precision"] = None
            print(f"\n    ⚠ context_precision: {e}")

        # ContextRecall (chỉ RAG intents)
        try:
            r = cr.score(user_input=q, retrieved_contexts=ctx, reference=ref)
            scores["context_recall"] = round(float(r.value), 3) if not math.isnan(r.value) else None
        except Exception as e:
            scores["context_recall"] = None
            print(f"\n    ⚠ context_recall: {e}")
    else:
        scores["context_precision"] = None
        scores["context_recall"]    = None

    return scores


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
            contexts = ["No context."]

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


def run_evaluation(
    quick    : bool = False,
    category : str  = "",
    save     : bool = False,
) -> dict:
    print("=" * 70)
    print("  RAGAs + Ollama Evaluation — Chatbot Tuyển sinh HaUI")
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

    print("── Bước 1: Khởi tạo RAGAs metrics ──")
    f, cp, cr = _setup_metrics()
    print()

    print("── Bước 2: Thu thập câu trả lời từ pipeline ──")
    rows = _collect_pipeline_data(ds, verbose=True)
    avg_kw  = sum(r["kw_hit_rate"] for r in rows) / len(rows)
    avg_lat = sum(r["latency_ms"] for r in rows) // len(rows)
    print(f"\n  Keyword hit rate TB: {avg_kw:.1%}")
    print(f"  Latency TB         : {avg_lat}ms")

    print("\n── Bước 3: Chạy RAGAs metrics (gọi trực tiếp score()) ──\n")
    scored = []
    for i, row in enumerate(rows, 1):
        print(f"  [{i:03d}/{len(rows)}] {row['question'][:50]}...", end=" ", flush=True)
        t0 = time.perf_counter()

        scores = _score_one(f, cp, cr, row)
        elapsed = int((time.perf_counter()-t0)*1000)

        f_s  = f"{scores['faithfulness']:.2f}"      if scores["faithfulness"]      is not None else "N/A"
        cp_s = f"{scores['context_precision']:.2f}" if scores["context_precision"] is not None else "N/A"
        cr_s = f"{scores['context_recall']:.2f}"    if scores["context_recall"]    is not None else "N/A"
        print(f"faith={f_s} prec={cp_s} recall={cr_s} ({elapsed}ms)")

        scored.append({**row, **scores})
        time.sleep(0.05)

    print("\n── Bước 4: Tổng hợp kết quả ──\n")

    def _avg(key):
        vals = [r[key] for r in scored if r.get(key) is not None]
        return round(sum(vals)/len(vals), 4) if vals else None

    avg_faith = _avg("faithfulness")
    avg_prec  = _avg("context_precision")
    avg_rec   = _avg("context_recall")

    cat_kw = {}
    for r in scored:
        cat_kw.setdefault(r["category"], []).append(r["kw_hit_rate"])
    cat_kw_avg = {c: round(sum(v)/len(v), 3) for c, v in cat_kw.items()}

    def _bar(s): return "▓"*int((s or 0)*20) + "░"*(20-int((s or 0)*20))
    def _st(s, t): 
        if s is None: return "➖"
        return "✅" if s >= t else ("⚠️ " if s >= t-0.1 else "❌")
    def _fmt(s): return f"{s:.4f}" if s is not None else "  N/A "

    print("╔" + "═"*62 + "╗")
    print("║    KẾT QUẢ RAGAs + Ollama — CHATBOT HaUI                  ║")
    print("╠" + "═"*62 + "╣")
    print(f"║  {_st(avg_faith,0.80)}  Faithfulness          {_fmt(avg_faith)}  [{_bar(avg_faith)}] ║")
    print(f"║  {_st(avg_prec, 0.75)}  Context Precision     {_fmt(avg_prec)}  [{_bar(avg_prec)}]  ║")
    print(f"║  {_st(avg_rec,  0.70)}  Context Recall        {_fmt(avg_rec)}  [{_bar(avg_rec)}]  ║")
    print(f"║  📊  Keyword Hit Rate   {avg_kw:.4f}                          ║")
    print(f"║  ⏱   Latency TB         {avg_lat}ms                             ║")
    print(f"║  📝  Số câu             {len(scored)}                                ║")
    print("╚" + "═"*62 + "╝")

    print("\n── Keyword hit rate theo danh mục ──")
    for cat, kw in sorted(cat_kw_avg.items(), key=lambda x: -x[1]):
        bar    = "▓"*int(kw*10) + "░"*(10-int(kw*10))
        status = "✅" if kw >= 0.8 else ("⚠️ " if kw >= 0.6 else "❌")
        print(f"  {status} {cat:<25} {kw:.1%}  [{bar}]")

    worst = sorted(scored, key=lambda r: r["kw_hit_rate"])[:5]
    if worst:
        print("\n── Top 5 câu thiếu keywords nhất ──")
        for r in worst:
            print(f"  [{r['id']}] {r['question'][:55]}...")
            print(f"         miss  : {r['keywords_miss']}")
            print(f"         faith={r.get('faithfulness')} prec={r.get('context_precision')} recall={r.get('context_recall')}")
            print(f"         answer: {r['answer'][:80]}...")

    result = {
        "timestamp"         : datetime.now().isoformat(),
        "framework"         : "ragas_direct",
        "judge_model"       : os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
        "total_questions"   : len(scored),
        "category_filter"   : category or "all",
        "avg_faithfulness"  : avg_faith,
        "avg_context_precision": avg_prec,
        "avg_context_recall": avg_rec,
        "keyword_hit_rate"  : avg_kw,
        "avg_latency_ms"    : avg_lat,
        "per_category_kw"   : cat_kw_avg,
        "detail"            : [{
            "id"               : r["id"],
            "category"         : r["category"],
            "question"         : r["question"],
            "answer"           : r["answer"][:300],
            "kw_hit_rate"      : r["kw_hit_rate"],
            "keywords_miss"    : r["keywords_miss"],
            "latency_ms"       : r["latency_ms"],
            "faithfulness"     : r.get("faithfulness"),
            "context_precision": r.get("context_precision"),
            "context_recall"   : r.get("context_recall"),
        } for r in scored],
    }

    if save:
        out_dir = ROOT / "tests" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{category}" if category else ""
        fj  = out_dir / f"eval_ragas_ollama{tag}_{ts}.json"
        fc  = out_dir / f"eval_ragas_ollama{tag}_{ts}.csv"
        fj.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ JSON: {fj}")
        try:
            import csv
            with open(fc, "w", newline="", encoding="utf-8-sig") as f_csv:
                fields = ["id","category","question","kw_hit_rate","keywords_miss",
                          "latency_ms","faithfulness","context_precision","context_recall","answer"]
                w = csv.DictWriter(f_csv, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(result["detail"])
            print(f"✅ CSV : {fc}")
        except Exception as e:
            print(f"⚠ CSV lỗi: {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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