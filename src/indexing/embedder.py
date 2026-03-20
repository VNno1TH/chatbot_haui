"""
embedder.py  (v3 — hỗ trợ Ollama embedding trên GPU cloud)

Hai chế độ, tự động chọn qua biến môi trường:

  Chế độ 1 — Local CPU (mặc định, không cần cấu hình thêm):
      EMBEDDING_MODEL=intfloat/multilingual-e5-small
      → SentenceTransformer load model về máy, encode trên CPU

  Chế độ 2 — Ollama GPU cloud (khuyến nghị, CPU máy nhàn hơn):
      OLLAMA_BASE_URL=http://localhost:11435
      OLLAMA_EMBED_MODEL=bge-m3
      → Gọi HTTP đến Ollama /v1/embeddings, encode trên GPU cloud

Khi dùng chế độ 2, phải rebuild ChromaDB một lần:
    ollama pull bge-m3           # trên GPU cloud
    python src/indexing/build_index.py --reset
"""

import os
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

# ── Cấu hình ─────────────────────────────────────────────────────────────────

DEFAULT_COLLECTION = "haui_tuyen_sinh"
EMBED_BATCH_SIZE   = 64

# Chế độ Ollama: bật khi có OLLAMA_EMBED_MODEL trong env
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "")   # rỗng = dùng local

# Chế độ local: model HuggingFace
DEFAULT_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Tắt HuggingFace online check khi đã có local
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ── Ollama Embedding Backend ──────────────────────────────────────────────────

class _OllamaEmbedder:
    """
    Gọi Ollama /v1/embeddings qua HTTP — encode trên GPU cloud.
    Dùng OpenAI-compatible endpoint (Ollama >= 0.1.24).
    """

    def __init__(self, base_url: str, model: str):
        self._url   = base_url.rstrip("/") + "/v1/embeddings"
        self._model = model
        self._dim   = self._detect_dim()
        print(f"  Ollama embedding endpoint: {self._url}")

    def _detect_dim(self) -> int:
        vec = self._call(["test"])[0]
        return len(vec)

    def _call(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({
            "model": self._model,
            "input": texts,
        }).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                return [item["embedding"] for item in data["data"]]
        except urllib.error.HTTPError as e:
            body = e.read().decode() if hasattr(e, "read") else ""
            raise RuntimeError(
                f"Ollama embedding lỗi HTTP {e.code} tại {self._url}\n"
                f"Response: {body}\n"
                f"Kiểm tra: ollama list | grep {self._model}"
            )
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Không kết nối Ollama tại {self._url}: {e}\n"
                f"Kiểm tra SSH tunnel đang chạy."
            )

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        all_vecs = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            all_vecs.extend(self._call(batch))
        return all_vecs

    def encode_query(self, query: str) -> list[float]:
        return self._call([query])[0]

    @property
    def dimension(self) -> int:
        return self._dim


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    """
    Quản lý embedding model và ChromaDB collection.

    Tự động chọn backend:
      - Có OLLAMA_EMBED_MODEL trong .env → dùng Ollama GPU cloud
      - Không có → load SentenceTransformer local trên CPU

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

        # ── Chọn backend embedding ────────────────────────────────────────────
        if OLLAMA_EMBED_MODEL:
            # Chế độ Ollama — encode trên GPU cloud, không tốn CPU local
            print(f"Embedding backend: Ollama GPU cloud")
            print(f"  URL  : {OLLAMA_BASE_URL}/v1/embeddings")
            print(f"  Model: {OLLAMA_EMBED_MODEL}")
            self._backend = _OllamaEmbedder(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL)
            self.model    = None   # không dùng SentenceTransformer
            print(f"  ✓ Ollama embedder OK — dimension: {self._backend.dimension}")
        else:
            # Chế độ local — load SentenceTransformer trên CPU
            from sentence_transformers import SentenceTransformer
            print(f"Embedding backend: local CPU")
            print(f"  Model: {model_name}")
            self._st      = SentenceTransformer(model_name, device=device)
            self._backend = None
            self.model    = self._st   # giữ để tương thích với router.py
            dim = self._st.get_sentence_embedding_dimension()
            print(f"  ✓ Model loaded — dimension: {dim}")

        # ── Khởi tạo ChromaDB ─────────────────────────────────────────────────
        vectorstore_dir = Path(vectorstore_dir)
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path     = str(vectorstore_dir),
            settings = Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name     = collection_name,
            metadata = {"hnsw:space": "cosine"},
        )
        print(f"  ✓ ChromaDB '{collection_name}' — {self.collection.count()} docs")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed văn bản (chunks)."""
        if self._backend:
            # Ollama: gửi text thẳng, không cần prefix
            return self._backend.encode_passages(texts)
        else:
            # Local SentenceTransformer: dùng prefix 'passage: '
            prefixed = [f"passage: {t}" for t in texts]
            vectors  = self._st.encode(
                prefixed,
                batch_size           = EMBED_BATCH_SIZE,
                show_progress_bar    = True,
                normalize_embeddings = True,
            )
            return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed câu hỏi của user."""
        if self._backend:
            return self._backend.encode_query(query)
        else:
            vector = self._st.encode(
                f"query: {query}",
                normalize_embeddings = True,
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