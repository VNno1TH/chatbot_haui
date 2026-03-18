"""
embedder.py  (v2 — model nhẹ hơn)
Thay bge-m3 (570MB, dim=1024) → multilingual-e5-small (118MB, dim=384).

Lý do:
  - Nhanh hơn ~4x khi encode trên CPU
  - RAM giảm ~450MB → còn nhiều hơn cho Qwen LLM
  - Chất lượng tiếng Việt vẫn tốt (trained trên 100 ngôn ngữ)
  - Prefix giống bge-m3: "query: " / "passage: "

LƯU Ý: Phải rebuild ChromaDB sau khi đổi model.
  python src/indexing/build_index.py --reset
"""

import os
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Cấu hình ─────────────────────────────────────────────────────────────────

# Model mới — nhẹ hơn 5x, hỗ trợ tiếng Việt tốt
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_COLLECTION = "haui_tuyen_sinh"
EMBED_BATCH_SIZE   = 64     # tăng batch size vì model nhẹ hơn

# Tắt HuggingFace online check khi đã có local
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    """
    Quản lý embedding model và ChromaDB collection.

    Cách dùng:
        embedder = Embedder()
        embedder.add_chunks(chunks)
        results = embedder.query("Ngành CNTT học gì?", n_results=5)
    """

    def __init__(
        self,
        model_name     : str  = DEFAULT_MODEL_NAME,
        vectorstore_dir: Path = Path("data/vectorstore/chroma_db"),
        collection_name: str  = DEFAULT_COLLECTION,
        device         : str  = "cpu",
    ):
        self.collection_name = collection_name

        # Load embedding model
        print(f"Đang load model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"  ✓ Model loaded — dimension: {dim}")

        # Khởi tạo ChromaDB
        vectorstore_dir = Path(vectorstore_dir)
        vectorstore_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path    = str(vectorstore_dir),
            settings= Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name    = collection_name,
            metadata= {"hnsw:space": "cosine"},
        )
        print(f"  ✓ ChromaDB '{collection_name}' — {self.collection.count()} docs")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed văn bản (chunks) — dùng prefix 'passage: '."""
        prefixed = [f"passage: {t}" for t in texts]
        vectors  = self.model.encode(
            prefixed,
            batch_size          = EMBED_BATCH_SIZE,
            show_progress_bar   = True,
            normalize_embeddings= True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed câu hỏi của user — dùng prefix 'query: '."""
        vector = self.model.encode(
            f"query: {query}",
            normalize_embeddings= True,
        )
        return vector.tolist()

    # ── CRUD ChromaDB ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_chunk_id(text: str, metadata: dict) -> str:
        raw = f"{metadata.get('source', '')}::{metadata.get('chunk_id', 0)}::{text[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def add_chunks(self, chunks: list[dict[str, Any]], skip_existing: bool = True) -> int:
        if not chunks:
            return 0

        texts     = [c["text"]     for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids       = [self._make_chunk_id(t, m) for t, m in zip(texts, metadatas)]

        if skip_existing:
            existing = set(self.collection.get(ids=ids)["ids"])
            new_idx  = [i for i, cid in enumerate(ids) if cid not in existing]
            if not new_idx:
                print(f"  ↷ Tất cả {len(chunks)} chunks đã tồn tại, bỏ qua")
                return 0
            texts     = [texts[i]     for i in new_idx]
            metadatas = [metadatas[i] for i in new_idx]
            ids       = [ids[i]       for i in new_idx]
            print(f"  → Thêm {len(new_idx)} chunks mới (bỏ qua {len(chunks)-len(new_idx)} đã có)")

        print(f"  Đang embed {len(texts)} chunks...")
        embeddings = self._embed_passages(texts)

        CHROMA_BATCH = 500
        for start in range(0, len(texts), CHROMA_BATCH):
            end = start + CHROMA_BATCH
            self.collection.upsert(
                ids        = ids[start:end],
                documents  = texts[start:end],
                embeddings = embeddings[start:end],
                metadatas  = metadatas[start:end],
            )

        print(f"  ✓ Đã lưu {len(texts)} chunks — tổng: {self.collection.count()}")
        return len(texts)

    def query(
        self,
        query_text: str,
        n_results : int  = 5,
        where     : dict = None,
    ) -> list[dict[str, Any]]:
        query_vec = self.embed_query(query_text)
        kwargs = dict(
            query_embeddings = [query_vec],
            n_results        = n_results,
            include          = ["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        output  = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text"    : doc,
                "metadata": meta,
                "score"   : round(1 - dist, 4),
            })
        return output

    def delete_collection(self):
        self.chroma_client.delete_collection(self.collection_name)
        print(f"✓ Đã xóa collection '{self.collection_name}'")

    def get_stats(self) -> dict:
        count  = self.collection.count()
        sample = self.collection.get(limit=min(count, 1000), include=["metadatas"])
        sources, loai_set, nganh_set, truong_set = set(), set(), set(), set()
        for m in sample["metadatas"]:
            sources.add(m.get("source", ""))
            loai_set.add(m.get("loai", ""))
            if m.get("ten_nganh"):  nganh_set.add(m["ten_nganh"])
            if m.get("truong_khoa"): truong_set.add(m["truong_khoa"])
        return {
            "total_chunks": count,
            "total_files" : len(sources),
            "loai_types"  : sorted(loai_set),
            "total_nganh" : len(nganh_set),
            "total_truong": len(truong_set),
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    embedder = Embedder()

    queries = [
        "Ngành công nghệ thông tin ra làm gì?",
        "Học phí ngành cơ khí bao nhiêu?",
        "Hướng dẫn đăng ký xét tuyển",
    ]
    for q in queries:
        t0      = time.perf_counter()
        results = embedder.query(q, n_results=3)
        ms      = (time.perf_counter() - t0) * 1000
        print(f"\n❓ {q}  ({ms:.0f}ms)")
        for r in results:
            label = r["metadata"].get("ten_nganh") or r["metadata"].get("source", "")
            print(f"   [{r['score']:.3f}] {label} — {r['text'][:100].strip()}...")

    print("\nThống kê:")
    for k, v in embedder.get_stats().items():
        print(f"  {k}: {v}")