"""答案自检与逐句溯源（★ 自研点 8 —— 本作品针对"AI 幻觉"的核心设计）

# 要解决的问题

RAG 系统普遍存在的假安全感：答案末尾挂着 [1][2]，看起来有据可依，
但**模型标注了引用，不等于这句话真的被原文支持**。
"答案带出处"和"答案内容被出处支持"是两件事，只有后者才叫可验证。

这一点在课程问答场景里危害尤其大：学生没有能力判断"课件第 3 页"
是否真的支持那个结论，反而会因为"有出处"而更相信它。

# 做法：生成后逐句回查

1. 把生成的答案拆成一条条**断言**（按句末标点切分，去掉引用标记）
2. 对每条断言，在检索到的原文里找**支持它的具体句子**
   - 先用向量相似度从原文句子里粗筛出少量候选（快）
   - 再用交叉编码器精算相关性（准）
   —— 两阶段和检索链路的设计思路一致，复用同一个重排模型
3. 最高分低于阈值 → 判定"课件未直接支持"，在界面上明确标记

# 为什么这样设计

- **可量化**：可以统计"无依据断言率"，这就把"抗幻觉"从口号变成了指标，
  报告里能写"自检机制在评测集上拦截了 X% 的无依据断言"
- **零新增依赖**：复用已有的 bge-m3 与 bge-reranker，是"一模型两用"
- **可解释**：不是黑盒打分，而是直接给出支持它的那句原文——人能复核

# 与已有工作的关系（报告里要如实说明）

逐句归因（attribution）与自检（self-verification）在学术界已有研究。
本模块的工作不是提出新算法，而是**用开源组件把它落地成一个可运行、
可量化的完整闭环**，并针对课程课件这类长文本、术语密集的语料做了适配
（两阶段筛选、句子切分、阈值校准）。这属于"在现有技术上的显著改进"。
"""

from __future__ import annotations

import re

import config
from src import index as index_mod
from src import retrieve

# 句末标点：中英文都覆盖
_SENT_END_RE = re.compile(r"(?<=[。！？；;!?])")
# 答案里的引用标记，比较断言时要去掉（否则 [1] 会干扰匹配）
_CITE_RE = re.compile(r"\[\d+\]")
# 断言里这些开头说明它不是知识性结论，不需要核验
_SKIP_PREFIXES = ("建议", "综上", "注意", "提示", "例如可以", "如果需要")

# 常见的 LaTeX 命令拍平成纯文本，用于显示与匹配
_LATEX_MAP = {
    r"\equiv": "≡", r"\times": "×", r"\cdot": "·", r"\pm": "±",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\partial": "∂", r"\int": "∫", r"\sum": "∑", r"\infty": "∞",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\mu": "μ",
    r"\pi": "π", r"\theta": "θ", r"\lambda": "λ",
}
_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_SPACING_RE = re.compile(r"\\[,;:!]|\\left|\\right|\\displaystyle")


def flatten_math(text: str) -> str:
    """把 LaTeX 公式拍平成可读纯文本。

    ⚠️ 这里踩过一个坑：最初的做法是"把 $...$ 整段删掉"，
    结果断言变成"全微分方程是指形如   的方程  。"——关键的公式内容全没了。
    对数学类课件来说，公式本身就是断言的核心，
    删掉公式等于把要核验的内容挖空，匹配质量也跟着下降。
    正确做法是保留内容、只去掉标记。
    """
    # 去掉定界符，但保留里面的公式内容
    text = re.sub(r"\$\$(.*?)\$\$", r" \1 ", text, flags=re.S)
    text = re.sub(r"\$([^$]*)\$", r" \1 ", text)
    # 常见命令拍平
    for cmd, ch in _LATEX_MAP.items():
        text = text.replace(cmd, ch)
    # \frac{a}{b} → a/b
    for _ in range(3):        # 处理嵌套
        text, n = _FRAC_RE.subn(r"\1/\2", text)
        if not n:
            break
    text = _SPACING_RE.sub("", text)
    text = _CMD_RE.sub("", text)      # 残留的未知命令直接去掉
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """把一段文本切成句子，去掉空白过短的片段。"""
    out = []
    for s in _SENT_END_RE.split(text):
        s = s.strip()
        if len(s) >= config.VERIFY_MIN_SENT_CHARS:
            out.append(s)
    return out


def split_claims(answer: str) -> list[str]:
    """把答案拆成需要核验的断言。

    先把公式拍平成纯文本、去掉引用标记，再切句。
    """
    text = _CITE_RE.sub(" ", flatten_math(answer))

    claims = []
    for s in split_sentences(text):
        s = s.strip("　 、，,")
        if len(s) < config.VERIFY_MIN_SENT_CHARS:
            continue
        if any(s.startswith(p) for p in _SKIP_PREFIXES):
            continue
        claims.append(s)
    return claims


def _candidate_sentences(hits: list[dict], max_per_section: int = 30) -> list[dict]:
    """从检索结果里抽出候选证据句，带上它们的出处信息。

    ⚠️ 必须用 hit["context"]（整章上下文）而不是 hit["text"]（小块）：
    大模型是看着整章上下文生成答案的，如果核验时只在小块里找证据，
    模型从同一章后半部分正确得出的结论就会被误判成"无依据"。
    实测这个 bug 让无依据率虚高（人工抽查发现 18 条误报集中在这里）。
    因此核验的搜索范围必须与生成时的可见范围一致。
    """
    out = []
    for hit in hits:
        body = hit.get("context") or hit["text"]
        sents = split_sentences(body)[:max_per_section]
        for s in sents:
            out.append({
                "text": s,
                "source": hit["source"],
                "page": hit["page"],
                "heading": hit.get("heading", ""),
                "chunk_rank": hit["rank"],
            })
    return out


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def verify_answer(answer: str,
                  hits: list[dict],
                  threshold: float | None = None) -> list[dict]:
    """对答案逐句核验。返回每条断言的核验结果。

    返回结构：
        [{
            "claim":      断言原文,
            "supported":  是否找到支持证据,
            "best_score": 支持句的最高相关性,
            "evidence":   [{"text","source","page","heading","score"}, ...]
        }, ...]

    找不到候选证据时（例如拒答、或检索结果为空）返回空列表。
    """
    if threshold is None:
        threshold = config.VERIFY_THRESHOLD or config.MIN_RERANK_SCORE

    claims = split_claims(answer)
    candidates = _candidate_sentences(hits)
    if not claims or not candidates:
        return []

    # --- 两阶段筛选 ---
    # 第一阶段：向量相似度粗筛。一次性编码所有候选句，避免每条断言重复编码。
    cand_vecs = index_mod.embed_texts([c["text"] for c in candidates], batch_size=32)
    claim_vecs = index_mod.embed_texts(claims, batch_size=32)

    results = []
    for claim, cvec in zip(claims, claim_vecs):
        sims = [_dot(cvec, cv) for cv in cand_vecs]
        top_idx = sorted(range(len(candidates)), key=lambda i: -sims[i])[:config.VERIFY_CANDIDATES]

        # 第二阶段：交叉编码器精算
        pairs = [(claim, candidates[i]["text"]) for i in top_idx]
        scores = retrieve.score_pairs(pairs)

        ranked = sorted(zip(top_idx, scores), key=lambda x: -x[1])
        best_score = ranked[0][1] if ranked else 0.0

        evidence = []
        for i, sc in ranked:
            if sc >= threshold:
                c = candidates[i]
                evidence.append({
                    "text": c["text"], "source": c["source"], "page": c["page"],
                    "heading": c["heading"], "score": round(sc, 4),
                })

        results.append({
            "claim": claim,
            "supported": best_score >= threshold,
            "best_score": round(best_score, 4),
            "evidence": evidence[:2],      # 最多给两条证据，界面上够用了
        })

    return results


def summarize(checks: list[dict]) -> dict:
    """汇总核验结果，用于界面提示和指标统计。"""
    if not checks:
        return {"total": 0, "unsupported": 0, "unsupported_rate": 0.0}
    bad = [c for c in checks if not c["supported"]]
    return {
        "total": len(checks),
        "unsupported": len(bad),
        "unsupported_rate": round(len(bad) / len(checks), 3),
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "什么是全微分方程"
    from src import answer as answer_mod

    result = answer_mod.answer_question(q, self_check=False)
    print("答案：", result["answer"][:200])
    print()
    checks = verify_answer(result["answer"], result["hits"])
    s = summarize(checks)
    print(f"核验：{s['total']} 条断言，{s['unsupported']} 条无依据"
          f"（无依据率 {s['unsupported_rate']:.1%}）")
    for c in checks:
        mark = "✅" if c["supported"] else "⚠️"
        print(f"  {mark} [{c['best_score']:.3f}] {c['claim'][:60]}")
        for e in c["evidence"]:
            print(f"        ← {e['source']} 第{e['page']}页：{e['text'][:50]}")
