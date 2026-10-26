"""建索引：解析 → 分块 → 向量化 → 写入向量库 + 关键词索引。

一次建好，之后检索直接读索引，不用重新跑模型。

用法：
    python -m src.index                 # 用 config 里的默认分块参数
    python -m src.index --max-chars 1000  # 做对比实验时改参数重建
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle

import config
from src import chunk as chunk_mod
from src import parse as parse_mod

CHUNKS_FILE = config.DATA_INDEX / "chunks.jsonl"
BM25_FILE = config.DATA_INDEX / "bm25.pkl"
COLLECTION_NAME = "chunks"

_embedder = None


# ---------------------------------------------------------------- 向量模型
def get_embedder():
    """懒加载 bge-m3。第一次调用会下载模型（约 1.1GB），之后就快了。"""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"[index] 加载向量模型 {config.EMBED_MODEL} ...")
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """把文本编码成向量。normalize_embeddings=True 配合 cosine 距离使用。"""
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------- 中文分词
def tokenize(text: str) -> list[str]:
    """BM25 用的分词。

    jieba 切词 + 去掉单字和标点。单字（"的""了"）对 BM25 没信息量，
    留着反而会稀释关键词的权重。
    """
    import jieba

    tokens = []
    for tok in jieba.lcut(text.lower()):
        tok = tok.strip()
        if len(tok) < 2:
            continue
        if not any(ch.isalnum() for ch in tok):
            continue
        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------- 建索引
def build_index(pages: list[dict] | None = None,
                max_chars: int | None = None,
                overlap: int | None = None,
                heading_aware: bool = True) -> list[dict]:
    """完整建索引流程。返回切好的块列表。"""
    config.DATA_INDEX.mkdir(parents=True, exist_ok=True)

    pages = pages if pages is not None else parse_mod.parse_all()
    chunks = chunk_mod.build_chunks(
        pages, max_chars=max_chars, overlap=overlap, heading_aware=heading_aware
    )
    print(f"[index] 分块统计：{chunk_mod.chunk_stats(chunks)}")

    # --- 1. 向量库（Chroma 嵌入式，不需要 Docker）---
    import chromadb

    client = chromadb.PersistentClient(path=str(config.DATA_INDEX))
    # 重建前先删掉旧集合，避免追加造成重复
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    print(f"[index] 向量化 {len(chunks)} 个块 ...")
    vectors = embed_texts([c["text"] for c in chunks])

    # 分批写入，避免一次性请求过大
    batch = 256
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in part],
            embeddings=vectors[i:i + batch],
            documents=[c["text"] for c in part],
            metadatas=[{
                "source": c["source"],
                "page": c["page"],
                "heading": c["heading"],
                "order": c["order"],
            } for c in part],
        )
    print(f"[index] 向量库写入完成：{collection.count()} 条")

    # --- 2. 关键词索引 ---
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(c["body"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks]}, f)
    print(f"[index] BM25 索引写入完成：{BM25_FILE.name}")

    # --- 3. 块原文存盘（评测和界面展示都要用）---
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[index] 块数据写入完成：{CHUNKS_FILE.name}")

    # 索引变了，之前缓存的集合句柄必须作废，否则后续检索仍指向旧数据
    get_collection.cache_clear()

    return chunks


# ---------------------------------------------------------------- 读索引
def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"索引不存在：{CHUNKS_FILE}\n请先运行：python -m src.index"
        )
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bm25():
    if not BM25_FILE.exists():
        raise FileNotFoundError(
            f"BM25 索引不存在：{BM25_FILE}\n请先运行：python -m src.index"
        )
    with open(BM25_FILE, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["ids"]


@functools.lru_cache(maxsize=1)
def get_collection():
    """获取向量集合。

    ⚠️ 必须缓存：PersistentClient 每次新建都要重新打开 sqlite 与 HNSW 索引，
    在检索路径上反复创建会显著拖慢每次问答（实测能拖到几秒级）。
    缓存后整个进程只建一次。
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(config.DATA_INDEX))
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="建索引")
    ap.add_argument("--max-chars", type=int, default=None, help="每块最大字符数")
    ap.add_argument("--overlap", type=int, default=None, help="块间重叠字符数")
    ap.add_argument("--no-heading", action="store_true",
                    help="关闭章节感知分块（做基线对比用）")
    args = ap.parse_args()

    build_index(
        max_chars=args.max_chars,
        overlap=args.overlap,
        heading_aware=not args.no_heading,
    )
