"""混合检索（★ 自研点 2 / 5 / 6 / 7）

为什么不能只用向量检索：
  - 向量擅长"语义相近"，但对课件的专有名词、公式编号、章节号不敏感。
    学生问"定理 3.2 的条件"，向量检索经常召回一堆语义相似但不对的段落。
  - BM25 擅长精确关键词匹配，但不懂同义表述（"梯度下降" vs "最速下降法"）。

本模块的四层设计，每一层都对应一类具体失败：

  第 1 层｜双路召回 + RRF 融合（自研点 2）
      向量懂语义、BM25 懂词面，两路都召回，用 RRF 按排名融合。
      为什么不用分数加权：向量相似度是 0~1 的余弦值，BM25 分数无上界，
      量纲不同，直接相加要么被量纲大的那路主导，要么得反复调归一化。
      RRF 只用排名，天然免调参。

  第 2 层｜查询扩展（自研点 5）
      学生提问用口语（"学了率调太大"），课件写的是术语（"学习率过大"）。
      词面对不上时两路都会失手。先用本地大模型把问题改写成多个检索式，
      分别召回再融合，这是"条件题、对比题"提升最明显的一层。

  第 3 层｜交叉编码器重排（自研点 2）
      把"问题+段落"拼在一起过一遍网络，精度远高于双塔，代价是慢，
      所以只对召回的几十条做。

  第 4 层｜MMR 去冗余（自研点 7）
      章节内相邻块高度重叠，Top-5 可能是同一段话的不同切法，
      白白浪费上下文预算。MMR 在相关性和多样性之间折中。

另外，检索用小块（精准），送给大模型时扩成整章（完整）——见 attach_context。
"""

from __future__ import annotations

import functools

import config
from src import index as index_mod

_reranker = None


# ---------------------------------------------------------------- 静态数据
@functools.lru_cache(maxsize=1)
def _load_static():
    """块数据和 BM25 索引只加载一次。"""
    return index_mod.load_chunks(), index_mod.load_bm25()


@functools.lru_cache(maxsize=1)
def _by_id() -> dict[str, dict]:
    chunks, _ = _load_static()
    return {c["id"]: c for c in chunks}


@functools.lru_cache(maxsize=1)
def _section_texts() -> dict[tuple[str, str], str]:
    """把每个章节的所有块拼成完整正文，供"小到大检索"使用。

    键是 (文件名, 章节标题)；没有标题的块按页分组，避免整篇糊成一团。
    """
    chunks, _ = _load_static()
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in chunks:
        key = (c["source"], c.get("heading") or f"__第{c['page']}页")
        groups.setdefault(key, []).append(c)

    out: dict[tuple[str, str], str] = {}
    for key, items in groups.items():
        items.sort(key=lambda x: x.get("order", 0))
        text = "\n".join(i["body"] for i in items)
        out[key] = text[:config.PARENT_MAX_CHARS]
    return out


def _section_key(chunk: dict) -> tuple[str, str]:
    return (chunk["source"], chunk.get("heading") or f"__第{chunk['page']}页")


def clear_caches() -> None:
    """重建索引后必须调用。

    否则块数据、章节正文这些缓存的还是旧内容，
    界面会继续按旧索引回答，而且不报错——这种"静默不一致"最难排查。
    """
    for fn in (_load_static, _by_id, _section_texts):
        fn.cache_clear()


# ---------------------------------------------------------------- 第 1 层：双路召回
def vector_search(query: str, k: int | None = None) -> list[tuple[str, float]]:
    """向量召回。返回 [(chunk_id, 相似度)]，相似度越大越相关。"""
    k = k or config.TOP_K_VECTOR
    collection = index_mod.get_collection()
    query_vec = index_mod.embed_texts([query])[0]
    res = collection.query(query_embeddings=[query_vec], n_results=k)

    ids = res["ids"][0]
    distances = res["distances"][0]
    # Chroma 用 cosine 距离（越小越近），转成相似度方便阅读
    return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]


def bm25_search(query: str, k: int | None = None) -> list[tuple[str, float]]:
    """关键词召回。返回 [(chunk_id, BM25 分数)]，分数越大越相关。"""
    k = k or config.TOP_K_BM25
    _, (bm25, ids) = _load_static()
    tokens = index_mod.tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    # 过滤 0 分：BM25 给 0 分表示一个关键词都没匹配上，属于纯噪声
    return [(cid, float(score)) for cid, score in ranked[:k] if score > 0]


def rrf_fuse(result_lists: list[list[tuple[str, float]]], k: int | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：按排名融合多路召回结果。

    公式：score(d) = Σ 1 / (k + rank_i(d))

    为什么用 RRF 而不是把两路分数加权相加：
      向量相似度是 0~1 的余弦值，BM25 分数是无上界的实数，两者量纲完全不同，
      直接加权要么得反复调归一化、要么被量纲大的那一路主导。
      RRF 只用排名，天然免调参，是工业界多路召回合并的常用做法。

    k 越小，越偏向各路排名靠前的结果（默认 60 是原论文推荐值）。

    启用查询扩展后，这里会同时融合"向量×N 个查询"和"BM25×N 个查询"，
    扩展查询命中的段落在多路里都出现时排名会被自然推高。
    """
    k = k or config.RRF_K
    fused: dict[str, float] = {}
    for results in result_lists:
        for rank, (cid, _score) in enumerate(results, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------- 第 2 层：查询扩展
_EXPANSION_PROMPT = """你是检索助手。学生用口语提问，但课件里用的是书面专业术语。
请把下面的问题改写成 {n} 个不同表述的检索查询，用来在课件中查找相关内容。

要求：
1. 补充问题中隐含的专业术语（例如"跑得太慢"→"收敛速度 迭代次数"）
2. 每个查询占一行，不要编号、不要标点开头、不要任何解释
3. 只输出查询本身，不要输出其他文字

问题：{question}"""


def expand_query(question: str, n: int | None = None) -> list[str]:
    """用本地大模型把问题改写成多个检索式。

    这是"用 AI 改善检索"而不是"用 AI 生成答案"——即使模型答得一般，
    它补充术语的能力也足以让召回率明显提升。

    失败时静默返回空列表（不影响主流程），所以不需要 try 之外的处理。
    """
    n = n or config.QUERY_EXPANSION_N
    try:
        import ollama

        resp = ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user",
                       "content": _EXPANSION_PROMPT.format(n=n, question=question)}],
            options={"temperature": 0.3},   # 改写需要一点多样性
        )
        lines = resp["message"]["content"].strip().splitlines()
    except Exception as exc:
        print(f"[retrieve] 查询扩展失败，回退到原始问题：{exc}")
        return []

    out: list[str] = []
    for line in lines:
        line = line.strip().lstrip("-·•0123456789.、) ").strip()
        # 过滤掉模型偶尔多写的解释性长句
        if 2 <= len(line) <= 60 and line != question and line not in out:
            out.append(line)
    return out[:n]


# ---------------------------------------------------------------- 第 3 层：重排
def get_reranker():
    """懒加载重排模型。约 1.1GB，显存紧张时可改到 CPU（见函数内注释）。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        print(f"[retrieve] 加载重排模型 {config.RERANK_MODEL} ...")
        # 显存不够时取消下面这行的注释，让重排跑在 CPU 上（慢一些但稳）
        # _reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """对一批 (查询, 文本) 对打分，返回 0~1 相关性概率。

    把激活函数的处理集中在这里，避免每个调用点都踩一遍"重复 sigmoid"的坑
    （详见 rerank 的注释）。答案自检模块也复用这个函数。
    """
    if not pairs:
        return []
    raw = [float(s) for s in get_reranker().predict(pairs)]
    # 正常情况下 predict() 已返回概率；
    # 万一换成未内置激活函数的模型，输出是任意实数 logits，这时才需要 sigmoid。
    if raw and (min(raw) < 0.0 or max(raw) > 1.0):
        import math
        raw = [1.0 / (1.0 + math.exp(-s)) for s in raw]
    return raw


def rerank(query: str, candidates: list[tuple[str, float]], top_k: int | None = None) -> list[tuple[str, float]]:
    """交叉编码器重排。

    向量检索是"双塔"结构（问题和文档分别编码，再算相似度），快但精度有限。
    重排模型把"问题+文档"拼在一起过一遍网络，精度高得多，代价是慢——
    所以只对召回的几十条做粗排，再用它精选前几条。
    返回 [(chunk_id, 相关性概率 0~1)]。

    ⚠️ 踩过的坑（务必不要再犯）：
    bge-reranker-v2-m3 的 num_labels=1，sentence-transformers 会**自动**套用
    Sigmoid 激活（CrossEncoder.activation_fn == Sigmoid），所以 predict() 返回的
    已经是 0~1 的相关性概率。如果这里再手动套一层 sigmoid，
    0.98 会被压成 0.73、0.001 会被抬成 0.50，
    真实区分度（0.983 vs 0.00002，相差 6 万倍）被彻底抹平，
    拒答阈值随之失效——所有问题看起来都是"中等相关"。
    激活函数的统一处理已抽到 score_pairs()。
    """
    top_k = top_k or config.TOP_K_FINAL
    if not candidates:
        return []

    by_id = _by_id()
    pairs = [(query, by_id.get(cid, {}).get("body", "")) for cid, _ in candidates]
    scored = [(cid, s) for (cid, _), s in zip(candidates, score_pairs(pairs))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------- 第 4 层：MMR 去冗余
def mmr_select(query: str,
               candidates: list[tuple[str, float]],
               k: int,
               lam: float | None = None) -> list[tuple[str, float]]:
    """最大边际相关性（MMR）：在相关性和多样性之间折中挑出 k 条。

    贪心地逐条选择，每次选"与问题最相关、同时与已选内容最不重复"的那一条：
        score = λ · rel(d) − (1−λ) · max_{s∈已选} sim(d, s)

    为什么需要：章节感知分块会产生大量内容重叠的相邻块，
    如果 Top-5 全落在同一段话上，就白白浪费了上下文预算，
    大模型能看到的有效信息反而变少。

    取不到向量时静默退化为按相关性取前 k 条。
    """
    lam = config.MMR_LAMBDA if lam is None else lam
    if len(candidates) <= k:
        return candidates

    try:
        collection = index_mod.get_collection()
        ids = [cid for cid, _ in candidates]
        got = collection.get(ids=ids, include=["embeddings"])
        emb_by_id = {cid: vec for cid, vec in zip(got["ids"], got["embeddings"])}
        query_vec = index_mod.embed_texts([query])[0]
    except Exception as exc:
        print(f"[retrieve] MMR 取向量失败，退化为按相关性取前 {k} 条：{exc}")
        return candidates[:k]

    def dot(a, b) -> float:
        return sum(x * y for x, y in zip(a, b))

    selected: list[tuple[str, float]] = []
    pool = list(candidates)
    while pool and len(selected) < k:
        best, best_score = None, float("-inf")
        for cid, rel in pool:
            vec = emb_by_id.get(cid)
            if vec is None:
                continue
            if selected:
                dup = max(dot(vec, emb_by_id[sid]) for sid, _ in selected
                          if emb_by_id.get(sid) is not None)
            else:
                dup = dot(vec, query_vec)
            score = lam * rel - (1 - lam) * dup
            if score > best_score:
                best, best_score = (cid, rel), score
        if best is None:
            break
        selected.append(best)
        pool.remove(best)

    return selected


# ---------------------------------------------------------------- 小到大检索
def attach_context(hits: list[dict]) -> list[dict]:
    """给每条命中补上它所在章节的完整正文。

    检索要准，所以用小块；但生成答案时只有那一小块往往缺少必要上下文
    （比如结论在块里、前提条件在上一段），所以送给大模型时扩成整章。
    这样"检索精度"和"上下文完整度"就不必二选一。
    """
    if not config.USE_PARENT_EXPANSION:
        for hit in hits:
            hit["context"] = hit["text"]
        return hits

    sections = _section_texts()
    by_id = _by_id()
    for hit in hits:
        chunk = by_id.get(hit["id"])
        full = sections.get(_section_key(chunk)) if chunk else None
        hit["context"] = full or hit["text"]
    return hits


# ---------------------------------------------------------------- 对外主函数
def hybrid_search(query: str,
                  use_vector: bool = True,
                  use_bm25: bool = True,
                  use_rerank: bool = True,
                  use_mmr: bool | None = None,
                  use_expansion: bool | None = None,
                  top_k: int | None = None,
                  return_query_variants: bool = False):
    """完整检索流程。返回带元数据的候选列表（已按相关性排序）。

    这些开关是为了做对比实验和消融分析：
        hybrid_search(q, use_mmr=False, use_expansion=False)  → 第 1 层（基线）
        hybrid_search(q, use_expansion=False)                 → 第 1+3 层
        hybrid_search(q, use_mmr=False)                       → 第 1+2+3 层
        hybrid_search(q)                                      → 完整方案
    报告里的消融表就靠这些开关跑出来——每一层各自带来多少提升，一目了然。
    """
    top_k = top_k or config.TOP_K_FINAL
    use_mmr = config.USE_MMR if use_mmr is None else use_mmr
    use_expansion = config.USE_QUERY_EXPANSION if use_expansion is None else use_expansion

    # --- 查询扩展：原始问题 + 若干改写查询 ---
    queries = [query]
    if use_expansion:
        extra = expand_query(query)
        if extra:
            queries.extend(extra)
            print(f"[retrieve] 查询扩展：{query} → {extra}")

    # --- 多路召回 ---
    result_lists: list[list[tuple[str, float]]] = []
    for q in queries:
        if use_vector:
            result_lists.append(vector_search(q))
        if use_bm25:
            result_lists.append(bm25_search(q))

    if not result_lists:
        return []

    fused = rrf_fuse(result_lists) if len(result_lists) > 1 else result_lists[0]

    # --- 重排 ---
    if use_rerank:
        # 启用 MMR 时多留一些候选，让去冗余有挑选余地
        pool_size = max(top_k * 3, config.MMR_CANDIDATES) if use_mmr else top_k
        ranked = rerank(query, fused, top_k=pool_size)
    else:
        ranked = fused[:max(top_k * 3, config.MMR_CANDIDATES) if use_mmr else top_k]

    # --- 去冗余 ---
    if use_mmr and use_rerank:      # 没有重排分数时 MMR 的"相关性"项不可比，故不启用
        ranked = mmr_select(query, ranked, top_k)
    else:
        ranked = ranked[:top_k]

    # --- 组装结果 ---
    by_id = _by_id()
    out = []
    for rank, (cid, score) in enumerate(ranked, start=1):
        c = by_id.get(cid)
        if c is None:
            continue
        out.append({
            "rank": rank,
            "id": cid,
            "score": score,
            "text": c["body"],
            "source": c["source"],
            "page": c["page"],
            "heading": c["heading"],
        })
    attach_context(out)

    if return_query_variants:
        return out, queries
    return out


def top_score(results: list[dict]) -> float:
    """最高相关性分数。answer.py 用它判断要不要拒答。"""
    return results[0]["score"] if results else 0.0


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "梯度下降的收敛条件是什么"
    print(f"问题：{q}\n")
    hits = hybrid_search(q)
    for hit in hits:
        src = f"{hit['source']} 第{hit['page']}页"
        head = f" / {hit['heading']}" if hit["heading"] else ""
        expanded = len(hit.get("context", "")) - len(hit["text"])
        print(f"[{hit['rank']}] {hit['score']:.3f}  {src}{head}"
              f"（上下文 +{max(0, expanded)} 字）")
        print(f"    {hit['text'][:110].replace(chr(10), ' ')}")
