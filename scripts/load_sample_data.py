from pathlib import Path
import argparse
import json
import logging
from typing import List, Dict
from tqdm import tqdm
import numpy as np

"""
File: scripts/load_sample_data.py

Small utility to load plain-text (txt, md) sample documents, chunk them,
create embeddings with sentence-transformers, and store a FAISS index
plus metadata ready for a retrieval-augmented generation demo.

Usage:
    python load_sample_data.py --data-dir ./sample_data --out-dir ./embeddings

Dependencies:
    pip install sentence-transformers faiss-cpu tqdm

If faiss is not available the script will save embeddings as a .npy file
and metadata as meta.jsonl in the output directory.
"""

# optional dependency: faiss
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAS_FAISS = False

# required dependency: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit(
        "Missing dependency: sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from e


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_text_files(data_dir: Path) -> Dict[str, str]:
    files = {}
    for p in sorted(data_dir.glob("**/*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                text = p.read_text(encoding="latin-1")
            files[str(p.relative_to(data_dir))] = text
    return files


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple word-based sliding window chunker.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_corpus(files: Dict[str, str], chunk_size: int, overlap: int):
    docs = []
    for path, text in files.items():
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            docs.append(
                {
                    "source": path,
                    "chunk_id": i,
                    "text": c,
                }
            )
    return docs


def embed_documents(model_name: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    return embeddings.astype("float32")


def save_outputs(out_dir: Path, docs: List[Dict], embeddings: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logging.info("Saved metadata to %s", meta_path)

    if HAS_FAISS:
        dim = embeddings.shape[1]
        # normalize for cosine similarity (inner product on normalized vectors)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        idx_path = out_dir / "index.faiss"
        faiss.write_index(index, str(idx_path))
        logging.info("Saved FAISS index to %s (n=%d, dim=%d)", idx_path, index.ntotal, dim)
    else:
        npy_path = out_dir / "embeddings.npy"
        np.save(str(npy_path), embeddings)
        logging.info("FAISS not available: saved embeddings to %s", npy_path)


def parse_args():
    p = argparse.ArgumentParser(description="Load sample data and build embeddings/index")
    p.add_argument("--data-dir", type=Path, default=Path("./sample_data"), help="Directory with sample text files")
    p.add_argument("--out-dir", type=Path, default=Path("./embeddings"), help="Output directory for index and metadata")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    p.add_argument("--chunk-size", type=int, default=500, help="Chunk size in words")
    p.add_argument("--overlap", type=int, default=50, help="Chunk overlap in words")
    return p.parse_args()


def main():
    args = parse_args()
    logging.info("Loading files from %s", args.data_dir)
    files = read_text_files(args.data_dir)
    if not files:
        logging.error("No .txt or .md files found in %s", args.data_dir)
        return
    logging.info("Found %d files", len(files))
    docs = build_corpus(files, chunk_size=args.chunk_size, overlap=args.overlap)
    logging.info("Built %d chunks", len(docs))
    texts = [d["text"] for d in docs]
    embeddings = embed_documents(args.model, texts)
    save_outputs(args.out_dir, docs, embeddings)
    logging.info("Done.")


if __name__ == "__main__":
    main()