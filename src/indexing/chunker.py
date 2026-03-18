"""
chunker.py
Chia file .md thành chunks có metadata, phục vụ embedding vào ChromaDB.

Chiến lược:
  1. Dùng MarkdownHeaderTextSplitter chia theo ## header
  2. Nếu chunk nào vẫn > MAX_CHUNK_CHARS → chia tiếp bằng RecursiveCharacterTextSplitter
  3. Gắn metadata từ YAML frontmatter vào mỗi chunk
"""

import re
from pathlib import Path
from typing import Any

import frontmatter                                          # pip install python-frontmatter
from langchain_text_splitters import (                     # pip install langchain-text-splitters
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ── Cấu hình ─────────────────────────────────────────────────────────────────

# Header dùng để chia chunk cấp 1
HEADERS_TO_SPLIT = [
    ("#",  "h1"),
    ("##", "h2"),
    ("###","h3"),
]

# Ngưỡng ký tự tối đa 1 chunk (≈ 500-600 token tiếng Việt)
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP   = 150

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Loại bỏ khoảng trắng thừa, giữ nguyên nội dung."""
    # Xóa nhiều dòng trống liên tiếp → 1 dòng trống
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Xóa khoảng trắng cuối dòng
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()


def _extract_frontmatter(filepath: Path) -> tuple[dict, str]:
    """
    Tách YAML frontmatter và nội dung markdown.
    Trả về (metadata_dict, markdown_content).
    """
    post = frontmatter.load(str(filepath))
    meta = dict(post.metadata)
    content = _clean_text(post.content)
    return meta, content


def _normalize_metadata(meta: dict, filepath: Path) -> dict:
    """
    Chuẩn hóa metadata — đảm bảo các field quan trọng luôn có.
    ChromaDB chỉ chấp nhận str/int/float/bool.
    """
    return {
        "source"      : filepath.name,
        "source_path" : str(filepath),
        "loai"        : str(meta.get("loai", "unknown")),
        "truong_khoa" : str(meta.get("truong_khoa", "")),
        "ten_nganh"   : str(meta.get("ten_nganh", "")),
        "ma_nganh"    : str(meta.get("ma_nganh", "")),
        "nam"         : str(meta.get("nam", "2025")),
        "nguon"       : str(meta.get("url", meta.get("nguon", ""))),
    }


def _add_section_context(chunks: list[dict], base_meta: dict) -> list[dict]:
    """
    Với mỗi chunk, thêm field 'section' (từ header ## của chunk đó)
    và 'nganh_context' để tăng chất lượng retrieval.
    """
    result = []
    for i, chunk in enumerate(chunks):
        meta = base_meta.copy()

        # Lấy tên section từ header metadata của LangChain
        section = (
            chunk["metadata"].get("h2")
            or chunk["metadata"].get("h3")
            or chunk["metadata"].get("h1")
            or "general"
        )
        meta["section"]  = str(section)
        meta["chunk_id"] = i

        # Prefix ngữ cảnh vào text để cải thiện embedding
        # (kỹ thuật của bge-m3: thêm "passage: " với văn bản cần embed)
        ten_nganh = base_meta.get("ten_nganh", "")
        if ten_nganh:
            context_prefix = f"Ngành {ten_nganh} — {section}\n"
        else:
            context_prefix = f"{section}\n"

        full_text = context_prefix + chunk["page_content"].strip()

        result.append({
            "text"    : full_text,
            "metadata": meta,
        })
    return result


# ── Chunker chính ─────────────────────────────────────────────────────────────

class MarkdownChunker:
    """
    Chia file .md thành danh sách chunks kèm metadata.

    Cách dùng:
        chunker = MarkdownChunker()
        chunks  = chunker.chunk_file(Path("nganh_cong_nghe_thong_tin.md"))
        # chunks = [{"text": "...", "metadata": {...}}, ...]
    """

    def __init__(
        self,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        chunk_overlap  : int = CHUNK_OVERLAP,
    ):
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT,
            strip_headers=False,   # giữ header trong text để bge-m3 hiểu ngữ cảnh
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_chars,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def chunk_file(self, filepath: Path) -> list[dict[str, Any]]:
        """
        Đọc 1 file .md → trả về list chunks.
        Mỗi chunk: {"text": str, "metadata": dict}
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Không tìm thấy: {filepath}")

        # 1. Tách frontmatter + content
        fm_meta, content = _extract_frontmatter(filepath)
        base_meta = _normalize_metadata(fm_meta, filepath)

        # 2. Chia theo markdown header
        header_chunks = self.header_splitter.split_text(content)

        # 3. Fallback: chia tiếp chunk nào quá dài
        final_raw_chunks = []
        for hchunk in header_chunks:
            if len(hchunk.page_content) <= MAX_CHUNK_CHARS:
                final_raw_chunks.append({
                    "page_content": hchunk.page_content,
                    "metadata"    : hchunk.metadata,
                })
            else:
                sub_docs = self.fallback_splitter.create_documents(
                    [hchunk.page_content],
                    metadatas=[hchunk.metadata],
                )
                for sub in sub_docs:
                    final_raw_chunks.append({
                        "page_content": sub.page_content,
                        "metadata"    : sub.metadata,
                    })

        # 4. Gắn metadata đầy đủ + context prefix
        chunks = _add_section_context(final_raw_chunks, base_meta)

        # 5. Lọc chunk rỗng
        chunks = [c for c in chunks if len(c["text"].strip()) > 30]

        return chunks

    def chunk_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Chunk toàn bộ file .md trong 1 thư mục.
        recursive=True → quét cả thư mục con.
        """
        pattern = "**/*.md" if recursive else "*.md"
        md_files = sorted(directory.glob(pattern))

        if not md_files:
            print(f"  ⚠ Không tìm thấy file .md trong {directory}")
            return []

        all_chunks = []
        for f in md_files:
            try:
                chunks = self.chunk_file(f)
                all_chunks.extend(chunks)
                print(f"  ✓ {f.name:<50} → {len(chunks):>3} chunks")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")

        return all_chunks


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Dùng: python chunker.py <đường_dẫn_file_hoặc_thư_mục>")
        sys.exit(1)

    target = Path(sys.argv[1])
    chunker = MarkdownChunker()

    if target.is_file():
        chunks = chunker.chunk_file(target)
        print(f"\nTổng: {len(chunks)} chunks\n")
        for i, c in enumerate(chunks[:3]):   # in thử 3 chunk đầu
            print(f"─── Chunk {i} ───")
            print(f"Section : {c['metadata']['section']}")
            print(f"Ngành   : {c['metadata']['ten_nganh']}")
            print(f"Text    :\n{c['text'][:300]}...")
            print()

    elif target.is_dir():
        chunks = chunker.chunk_directory(target)
        print(f"\nTổng: {len(chunks)} chunks từ {target}")
    else:
        print(f"Không tìm thấy: {target}")