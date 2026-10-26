"""生成答案 + 引用溯源 + 拒答（★ 自研点 3）

这个模块体现作品的两个核心主张：

1. **答案可追溯**：提示词强制模型用 [1][2] 标注依据，并把编号映射回课件页码。
   学生不是在"听一个 AI 说话"，而是在"看课件哪一页支持这个结论"。

2. **不知道就说不知道**：检索到的最高相关性分数低于阈值时直接拒答。
   通用聊天机器人在课程问答场景最大的问题不是答得不好，而是**自信地编造**。
   拒答能力是本作品与"套壳通用模型"最本质的区别。
"""

from __future__ import annotations

import re

import config
from src import retrieve

# 模型偶尔不守规矩，用这个兜底把 [n] 提出来
_CITE_RE = re.compile(r"\[(\d+)\]")

# 模型表达"课件里没有"时常见的说法。命中任意一条即判定为拒答。
# 这份词表可以根据你们的实际运行结果继续补充——属于正常的适配工作。
_REFUSAL_MARKERS = (
    "课件中没有", "课件里没有", "课件未包含", "课件不包含",
    "没有相关内容", "未提及", "没有提到", "并未涉及",
    "无法回答", "无法从课件", "无法确定", "无法为您",
    "不在本课程范围", "超出课件范围",
)


def looks_like_refusal(text: str) -> bool:
    """判断模型的回答是不是在表达"课件里没有相关内容"。

    只看开头一小段：如果模型先说了"课件中没有相关内容"再补充别的，
    开头就足以判定；而正文中间偶然出现"无法确定"之类措辞的正常回答不会被误判。
    """
    head = text.strip()[:150]
    return any(marker in head for marker in _REFUSAL_MARKERS)


def normalize_math(text: str) -> str:
    """把模型输出的 LaTeX 分隔符统一成 Streamlit 能渲染的形式。

    ⚠️ 踩过的坑：提示词里明确要求"行内公式用 $...$"，但 Qwen 系列模型
    训练时的习惯极顽固，仍会输出 \\(...\\) 和 \\[...\\] 格式。
    Streamlit 的 Markdown 只渲染 $...$ 和 $$...$$，于是 \\equiv 这类符号
    就以原始文本露出来，公式显示一团糟。
    提示词管不住的，就在代码里做确定性转换——渲染前统一替换。
    """
    return (text
            .replace("\\[", "$$")
            .replace("\\]", "$$")
            .replace("\\(", "$")
            .replace("\\)", "$"))

SYSTEM_PROMPT = """你是一位严谨的课程助教。你的唯一知识来源是下面提供的课件片段。

必须遵守的规则：
1. 只根据课件片段回答，不得使用课件之外的任何知识，不得凭常识补充或推测。
2. 每一条结论后面必须标注依据编号，例如：[1]、[2]。可以一条结论引用多个来源，写作 [1][3]。
3. 如果课件片段不足以回答这个问题，直接回答"课件中没有相关内容，无法回答"，不要尝试猜测，也不要给"一般来说"这类泛泛的回答。
4. 用中文回答，条理清晰，涉及定义、公式、条件时保持与课件完全一致。
5. 数学公式必须用 LaTeX 书写：行内公式用 $...$ 包裹，独立公式用 $$...$$。
   课件片段中的公式是从 PDF 提取的，常有符号缺失、乱码或写法不规范（例如 \\equiv、分数变成斜杠）。
   你要根据上下文把它改写成规范的 LaTeX（如 $\\\\equiv$、$\\\\frac{a}{b}$），不要原样照抄损坏的写法。
   纯文字描述的简单运算（如 x dx + y dy = 0）可以保持文字形式，不必强行加 $。
6. 如果课件片段之间有冲突，指出冲突并说明分别出自哪一条来源。

作答格式：
- 先给结论，再给必要解释。
- 结尾不需要重复来源列表，编号在正文中标注即可。"""

USER_TEMPLATE = """课件片段：

{context}

---
学生的问题：{question}"""


class LLMUnavailable(RuntimeError):
    """Ollama 没启动或模型没拉取时抛出，界面上要给人话提示。"""


# ---------------------------------------------------------------- 提示词组装
def build_context(hits: list[dict], max_chars: int | None = None) -> str:
    """把检索结果拼成带编号的上下文。

    编号顺序 = 检索排名顺序，所以 [1] 永远是相关性最高的那条。

    两个关键处理：

    1. **用 hit["context"]（命中所属章节的完整正文）而不是 hit["text"]（命中的那一小块）**
       这是"小到大检索"：检索靠小块保证精准，生成靠整章保证上下文完整。
       否则经常出现"结论找到了，但它成立的前提在上一段"的情况，模型只能答一半。

    2. **按章节去重**
       一个小到大扩展后，同一章节的多个命中会得到完全相同的 context 文本；
       如果原样拼进去，同样的内容会重复占好几份上下文预算。
       这里跳过已经出现过的 context，把预算留给真正不同的章节。
       这比在检索阶段用 MMR 去冗余更好：既省了预算，又不动引用排序。
    """
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    parts: list[str] = []
    seen: set[str] = set()
    used = 0

    for i, hit in enumerate(hits, start=1):
        loc = f"{hit['source']} 第{hit['page']}页"
        if hit.get("heading"):
            loc += f"（{hit['heading']}）"
        body = hit.get("context") or hit["text"]

        # 同一章节的内容只出现一次，避免重复占用上下文预算
        key = body.strip()
        if key in seen:
            continue

        block = f"[{i}] 出处：{loc}\n{body}"
        if used + len(block) > max_chars:
            # 超预算时不再整块丢弃，而是把这一块截断塞进去——
            # 否则上下文只剩前两三条，后面的命中白召回了
            remain = max_chars - used
            if remain > 200:
                parts.append(block[:remain] + "\n……（后续内容因长度限制省略）")
                seen.add(key)
            break

        seen.add(key)
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


# ---------------------------------------------------------------- 调用模型
def call_llm(question: str, context: str, model: str | None = None) -> str:
    """调用本地 Ollama。失败时抛 LLMUnavailable，由上层给出可操作的提示。"""
    import ollama

    model = model or config.LLM_MODEL
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
            ],
            options={"temperature": config.TEMPERATURE},
        )
    except Exception as exc:
        raise LLMUnavailable(
            f"无法调用本地模型（{model}）。请确认：\n"
            f"1) 已安装并启动 Ollama\n"
            f"2) 已执行过 ollama pull {model}\n"
            f"原始错误：{exc}"
        ) from exc

    return resp["message"]["content"].strip()


def call_llm_raw(prompt: str, model: str | None = None,
                 temperature: float = 0.0) -> str:
    """单轮裸提示词调用（核验修订等内部环节用）。失败同样抛 LLMUnavailable。"""
    import ollama

    model = model or config.LLM_MODEL
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
    except Exception as exc:
        raise LLMUnavailable(str(exc)) from exc
    return resp["message"]["content"].strip()


# ---------------------------------------------------------------- 核验后的自动修订
# 这是"生成 → 逐句核验 → 自动修订 → 复验"闭环的第二环。
# 只标记问题不算完：发现无依据结论后，把它们连同课件依据喂回模型，
# 要求只改这些结论——改完再复验一遍，无依据数下降才采纳。
_REVISION_PROMPT = """你之前生成了一份课程问答答案，其中部分结论经与课件原文逐句比对，发现课件中没有直接支持。

请修订这份答案：
1. 对下面列出的每条"无依据结论"：删除它，或改写成与给出的课件依据严格一致的说法；课件依据为"无"的直接删除
2. 已经有依据的结论保持原样，不要改写
3. 保留原有的引用编号 [n]，不得引入课件之外的任何新知识
4. 只做必要的最小修改，输出修订后的完整答案，不要输出任何解释

原答案：
{answer}

需要修订的无依据结论：
{claims}"""


def revise_answer(text: str, checks: list[dict], model: str | None = None) -> str:
    """根据核验结果修订答案。只处理无依据的结论。"""
    bad = [c for c in checks if not c["supported"]]
    if not bad:
        return text

    lines = []
    for i, c in enumerate(bad, start=1):
        if c["evidence"]:
            e = c["evidence"][0]
            ev = f"课件依据（{e['source']} 第{e['page']}页）：{e['text'][:100]}"
        else:
            ev = "课件依据：无（课件中找不到任何支持，直接删除这条）"
        lines.append(f"{i}. 结论：{c['claim']}\n   {ev}")

    prompt = _REVISION_PROMPT.format(answer=text, claims="\n".join(lines))
    return call_llm_raw(prompt, model=model, temperature=0.0)


# ---------------------------------------------------------------- 对外主函数
def _map_citations(text: str, hits: list[dict]) -> list[dict]:
    """把答案里的 [n] 还原成课件出处；一条都没有时挂上最相关的那条。"""
    cites = []
    for m in _CITE_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= len(hits) and not any(c["index"] == idx for c in cites):
            h = hits[idx - 1]
            cites.append({
                "index": idx,
                "source": h["source"],
                "page": h["page"],
                "heading": h.get("heading", ""),
                "score": h["score"],
            })
    if not cites and hits:
        h = hits[0]
        cites.append({
            "index": 1, "source": h["source"], "page": h["page"],
            "heading": h.get("heading", ""), "score": h["score"],
        })
    return cites


def answer_question(question: str,
                    top_k: int | None = None,
                    force_answer: bool = False,
                    retrieval_kwargs: dict | None = None,
                    self_check: bool | None = None,
                    self_revision: bool | None = None) -> dict:
    """完整问答流程（含核验-修订闭环）。

    参数
    ----
    question       : 学生的问题
    force_answer   : True 时跳过拒答判断（调试用）
    retrieval_kwargs: 透传给 hybrid_search，用于做对比实验
    self_check     : 是否对答案做逐句核验（默认取 config.USE_SELF_CHECK）
    self_revision  : 核验发现无依据结论时是否自动修订（默认取 config.USE_SELF_REVISION）

    返回
    ----
    {
      "question", "answer", "refused", "top_score",
      "citations": [{"index":1, "source", "page", "heading", "score"}],
      "hits": [...],
      "checks": [...],        # 最终答案的逐句核验结果
      "check_summary": {...}, # 最终核验汇总
      "revision": {...} | None
      # 触发过自动修订时：{"before": 修订前汇总, "after": 修订后汇总}
      # 没有（或修订无效被回退）时为 None
    }
    """
    hits = retrieve.hybrid_search(question, top_k=top_k, **(retrieval_kwargs or {}))
    best = retrieve.top_score(hits)

    # --- 拒答判断：相关性太低，说明课件里没有相关内容 ---
    if not hits or (best < config.MIN_RERANK_SCORE and not force_answer):
        return {
            "question": question,
            "answer": "课件中没有相关内容，无法回答。建议换个问法，或者确认这个知识点是否在本课程范围内。",
            "refused": True,
            "top_score": best,
            "citations": [],
            "hits": hits,
            "checks": [],
            "check_summary": {"total": 0, "unsupported": 0, "unsupported_rate": 0.0},
            "revision": None,
        }

    context = build_context(hits)
    text = normalize_math(call_llm(question, context))

    # --- 第二级拒答判断：看模型自己的回答是否表明"课件里没有" ---
    # 为什么需要这一级：相关性分数只能反映"检索到的内容与问题像不像",
    # 无法覆盖"检索到了相关段落、但段落里其实没有答案"这种情况。
    # 实测中模型会遵守提示词规则主动说"课件中没有相关内容"，
    # 但如果程序不识别这句话，就会把它当成正常答案、还附上一条不相干的出处，
    # 学生看到的就会是一个"挂着出处却什么也没回答"的怪结果。
    if looks_like_refusal(text):
        return {
            "question": question,
            "answer": text,
            "refused": True,
            "top_score": best,
            "citations": [],      # 拒答时不给出处，避免误导
            "hits": hits,
            "checks": [],
            "check_summary": {"total": 0, "unsupported": 0, "unsupported_rate": 0.0},
            "revision": None,
        }

    # --- 逐句核验（★ 自研点 8）：把"有引用"变成"引用真被原文支持" ---
    checks: list[dict] = []
    check_summary = {"total": 0, "unsupported": 0, "unsupported_rate": 0.0}
    do_check = config.USE_SELF_CHECK if self_check is None else self_check
    if do_check:
        try:
            from src import verify as verify_mod
            checks = verify_mod.verify_answer(text, hits)
            check_summary = verify_mod.summarize(checks)
        except Exception as exc:
            # 核验是增强能力，失败不应该让整个问答失败
            print(f"[answer] 答案核验失败（不影响作答）：{exc}")

    # --- 核验后的自动修订（★ 闭环的第二环）---
    # 发现无依据结论 → 连同课件依据喂回模型，只改这些结论 → 复验 →
    # 无依据数下降才采纳，否则回退原答案（防止越改越差）。
    revision_info = None
    do_revise = config.USE_SELF_REVISION if self_revision is None else self_revision
    if do_check and do_revise and check_summary["unsupported"] > 0:
        try:
            from src import verify as verify_mod
            revised = revise_answer(text, checks)
            revised = normalize_math(revised) if revised else ""
            if revised and not looks_like_refusal(revised):
                checks2 = verify_mod.verify_answer(revised, hits)
                s2 = verify_mod.summarize(checks2)
                before_count = check_summary["unsupported"]
                if s2["total"] > 0 and s2["unsupported"] < before_count:
                    revision_info = {"before": check_summary, "after": s2}
                    text, checks, check_summary = revised, checks2, s2
                    print(f"[answer] 自动修订：无依据结论 {before_count} → {s2['unsupported']}")
                else:
                    print("[answer] 修订未减少无依据结论，已回退原答案")
        except Exception as exc:
            print(f"[answer] 自动修订失败（保留原答案）：{exc}")

    cites = _map_citations(text, hits)

    return {
        "question": question,
        "answer": text,
        "refused": False,
        "top_score": best,
        "citations": cites,
        "hits": hits,
        "checks": checks,
        "check_summary": check_summary,
        "revision": revision_info,
    }


def format_citations(citations: list[dict]) -> str:
    """把引用列表拼成一行出处文本，界面上和导出时都用。"""
    if not citations:
        return ""
    seen, parts = set(), []
    for c in citations:
        key = (c["source"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"[{c['index']}] {c['source']} 第{c['page']}页")
    return "　".join(parts)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "什么是梯度下降"
    result = answer_question(q)
    print(f"问题：{result['question']}")
    print(f"最高相关性：{result['top_score']:.3f}"
          f"（拒答阈值 {config.MIN_RERANK_SCORE}）")
    print(f"是否拒答：{'是' if result['refused'] else '否'}")
    print("-" * 60)
    print(result["answer"])
    if result["citations"]:
        print("-" * 60)
        print("出处：" + format_citations(result["citations"]))
