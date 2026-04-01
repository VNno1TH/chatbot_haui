"""
PATCH 4 — Script chẩn đoán: chạy để xác định vấn đề TRƯỚC khi sửa code
Lưu thành: diagnose.py ở root project, chạy: python diagnose.py
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
VECTORSTORE_DIR = ROOT_DIR / "data" / "vectorstore" / "chroma_db"

def check_files():
    print("=" * 60)
    print("KIỂM TRA 1: File .md trong PROCESSED_DIR")
    print("=" * 60)
    if not PROCESSED_DIR.exists():
        print(f"  ✗ PROCESSED_DIR không tồn tại: {PROCESSED_DIR}")
        return

    all_md = list(PROCESSED_DIR.glob("**/*.md"))
    print(f"  Tổng file .md tìm thấy: {len(all_md)}")

    # Tìm file gioi_thieu_truong
    gt_files = [f for f in all_md if "gioi_thieu" in f.name.lower()]
    if gt_files:
        for f in gt_files:
            print(f"  ✓ FOUND: {f.relative_to(ROOT_DIR)}")
    else:
        print("  ✗ KHÔNG TÌM THẤY file gioi_thieu_truong.md!")
        print("    → Copy file này vào thư mục data/processed/")
        print(f"    → Đường dẫn đề nghị: {PROCESSED_DIR / 'gioi_thieu_truong.md'}")

    # Liệt kê tất cả
    print(f"\n  Tất cả file .md:")
    for f in sorted(all_md):
        print(f"    {f.relative_to(ROOT_DIR)}")


def check_index():
    print("\n" + "=" * 60)
    print("KIỂM TRA 2: ChromaDB có chunk từ gioi_thieu_truong không")
    print("=" * 60)

    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection("haui_tuyen_sinh")
        total = col.count()
        print(f"  Tổng docs trong ChromaDB: {total}")

        # Lấy sample metadata
        sample = col.get(limit=min(total, 500), include=["metadatas"])
        sources = set(m.get("source", "") for m in sample["metadatas"])
        loai_set = set(m.get("loai", "") for m in sample["metadatas"])

        print(f"\n  Các file đã được index:")
        for s in sorted(sources):
            count = sum(1 for m in sample["metadatas"] if m.get("source") == s)
            print(f"    {s}: {count} chunks")

        print(f"\n  Các loại tài liệu (loai): {sorted(loai_set)}")

        gt_sources = [s for s in sources if "gioi_thieu" in s.lower()]
        if gt_sources:
            print(f"\n  ✓ gioi_thieu_truong đã được index: {gt_sources}")
        else:
            print("\n  ✗ KHÔNG CÓ chunk nào từ gioi_thieu_truong trong ChromaDB!")
            print("    → Nguyên nhân 1: File chưa được copy vào PROCESSED_DIR")
            print("    → Nguyên nhân 2: build_index.py chưa được chạy sau khi thêm file")
            print("    → FIX: Chạy python src/indexing/build_index.py --reset")

    except Exception as e:
        print(f"  ✗ Lỗi kết nối ChromaDB: {e}")


def check_query():
    print("\n" + "=" * 60)
    print("KIỂM TRA 3: Query test cho câu hỏi cơ cấu tổ chức")
    print("=" * 60)

    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    OLLAMA_EMBED    = os.environ.get("OLLAMA_EMBED_MODEL", "")

    try:
        from src.indexing.embedder import Embedder
        embedder = Embedder(
            model_name=EMBEDDING_MODEL,
            vectorstore_dir=VECTORSTORE_DIR,
        )

        test_queries = [
            "HaUI có bao nhiêu trường trực thuộc",
            "cơ cấu tổ chức đào tạo HaUI",
            "trường thành viên Đại học Công nghiệp Hà Nội",
            "HaUI gồm những khoa nào",
        ]

        THRESHOLD_OK    = 0.55   # score >= này → kết quả tốt
        THRESHOLD_WARN  = 0.45   # score >= này → kết quả chấp nhận được

        for q in test_queries:
            results = embedder.query(q, n_results=3)
            if not results:
                print(f"  ✗ NO RESULT — '{q}'")
                continue
            top = results[0]
            score = top["score"]
            source = top["metadata"].get("source", "?")
            text_preview = top["text"][:80].strip()

            if score >= THRESHOLD_OK:
                status = "✓ OK  "
            elif score >= THRESHOLD_WARN:
                status = "⚠ WARN"
            else:
                status = "✗ BAD "

            print(f"  {status} [{score:.3f}] '{q[:45]}'")
            print(f"           → {source}: {text_preview}...")

            # Cờ đỏ: score thấp VÀ source không phải gioi_thieu
            if score < THRESHOLD_WARN and "gioi_thieu" not in source:
                print(f"           ⚠ CẢNH BÁO: score thấp, sai nguồn → context sẽ MISS!")

    except Exception as e:
        print(f"  ✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()


def print_fix_plan():
    print("\n" + "=" * 60)
    print("KẾ HOẠCH SỬA THEO THỨ TỰ ƯU TIÊN")
    print("=" * 60)
    print("""
  Bước 1 (QUAN TRỌNG NHẤT — làm ngay):
    • Copy gioi_thieu_truong.md vào data/processed/
    • Chạy: python src/indexing/build_index.py --reset
    • Chạy lại: python diagnose.py để kiểm tra

  Bước 2 (Sửa code — sau khi xác nhận data đã vào index):
    • Áp dụng patch_2_system_prompt.py → sửa SYSTEM_PROMPT trong chatbot.py
    • Áp dụng patch_3_retriever.py → mở rộng FACTUAL_KEYWORDS, thêm synonyms

  Bước 3 (Sửa build_index.py để không bao giờ bị sót nữa):
    • Áp dụng patch_1_build_index.py → collect_md_files() dùng **/*.md everywhere
    • Thêm verify_critical_chunks() vào cuối hàm build()

  Test cuối:
    • Chạy chatbot, hỏi: "HaUI có bao nhiêu trường trực thuộc?"
    • Kết quả đúng phải là: "HaUI có 5 Trường và 4 Khoa..."
    • Không được có cụm: "Theo thông tin tôi được đào tạo"
""")


if __name__ == "__main__":
    check_files()
    check_index()
    check_query()
    print_fix_plan()