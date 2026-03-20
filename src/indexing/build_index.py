"""
build_index.py  (v2)
Pipeline: đọc .md → chunk → embed (multilingual-e5-small) → lưu ChromaDB

Cách dùng:
    python src/indexing/build_index.py           # build (skip chunks đã có)
    python src/indexing/build_index.py --reset   # xóa DB cũ rồi build lại từ đầu
    python src/indexing/build_index.py --test    # chỉ test query
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from src.indexing.chunker  import MarkdownChunker
from src.indexing.embedder import Embedder

# ── Cấu hình ─────────────────────────────────────────────────────────────────

BASE_DIR        = ROOT_DIR
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma_db"

# Đọc từ env — mặc định multilingual-e5-small (local CPU)
# Để dùng bge-m3 trên GPU cloud: thêm EMBEDDING_MODEL=BAAI/bge-m3 vào .env
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

MD_DIRS = [
    PROCESSED_DIR / "nganh",
    PROCESSED_DIR,
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_md_files(dirs: list[Path]) -> list[Path]:
    seen, files = set(), []
    for d in dirs:
        if not d.exists():
            print(f"  ⚠ Thư mục không tồn tại: {d}")
            continue
        pattern = "**/*.md" if d.name == "nganh" else "*.md"
        for f in sorted(d.glob(pattern)):
            if f.resolve() not in seen:
                seen.add(f.resolve())
                files.append(f)
    return files


def print_section(title: str):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build(reset: bool = False):
    t0 = time.time()

    print_section("1. Khởi tạo model & vector store")
    embedder = Embedder(
        model_name      = EMBEDDING_MODEL,
        vectorstore_dir = VECTORSTORE_DIR,
        collection_name = "haui_tuyen_sinh",
    )

    if reset:
        print("  ⚠ --reset: Xóa toàn bộ collection cũ...")
        embedder.delete_collection()
        embedder = Embedder(
            model_name      = EMBEDDING_MODEL,
            vectorstore_dir = VECTORSTORE_DIR,
            collection_name = "haui_tuyen_sinh",
        )

    print_section("2. Thu thập file .md")
    md_files = collect_md_files(MD_DIRS)
    print(f"  Tìm thấy {len(md_files)} file .md")

    print_section("3. Chunking")
    chunker    = MarkdownChunker()
    all_chunks = []
    ok, fail   = 0, 0
    for f in md_files:
        try:
            chunks = chunker.chunk_file(f)
            all_chunks.extend(chunks)
            rel = f.relative_to(BASE_DIR)
            print(f"  ✓ {str(rel):<65} → {len(chunks):>3} chunks")
            ok += 1
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
            fail += 1
    print(f"\n  Tổng: {len(all_chunks)} chunks từ {ok} file ({fail} lỗi)")

    if not all_chunks:
        print("\n❌ Không có chunks — dừng lại.")
        return

    print_section("4. Embedding & lưu ChromaDB")
    added = embedder.add_chunks(all_chunks, skip_existing=not reset)

    print_section("5. Kết quả")
    stats   = embedder.get_stats()
    elapsed = time.time() - t0
    print(f"  Model         : {EMBEDDING_MODEL}")
    print(f"  Thời gian     : {elapsed:.1f}s")
    print(f"  Chunks mới    : {added}")
    print(f"  Tổng chunks   : {stats['total_chunks']}")
    print(f"  Tổng file     : {stats['total_files']}")
    print(f"  Tổng ngành    : {stats['total_nganh']}")
    print(f"  Loại tài liệu : {stats['loai_types']}")
    print(f"\n✅ Build xong! VectorDB: {VECTORSTORE_DIR}")


def test_query():
    embedder = Embedder(
        model_name      = EMBEDDING_MODEL,
        vectorstore_dir = VECTORSTORE_DIR,
    )
    TEST_QUERIES = [
        "Ngành công nghệ thông tin HaUI học gì?",
        "Điều kiện xét tuyển ngành kế toán là gì?",
        "Ngành robot và trí tuệ nhân tạo ra làm việc ở đâu?",
        "Học phí ngành điện tử bao nhiêu?",
        "Hướng dẫn đăng ký xét tuyển năm 2025",
        "Chính sách học bổng HaUI",
    ]
    print_section("Test query")
    for q in TEST_QUERIES:
        import time as _t
        t0      = _t.perf_counter()
        results = embedder.query(q, n_results=3)
        ms      = (_t.perf_counter() - t0) * 1000
        print(f"\n❓ {q}  ({ms:.0f}ms)")
        for i, r in enumerate(results):
            label = r["metadata"].get("ten_nganh") or r["metadata"].get("source", "")
            section = r["metadata"].get("section", "")
            print(f"   {i+1}. [{r['score']:.3f}] {label} — {section}")
            print(f"        {r['text'][:120].strip()}...")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Xóa DB cũ và build lại")
    parser.add_argument("--test",  action="store_true", help="Chỉ test query")
    args = parser.parse_args()

    if args.test:
        test_query()
    else:
        build(reset=args.reset)
        test_query()