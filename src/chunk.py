"""分块（★ 自研点 1）

朴素做法：把整页文本按固定长度硬切。
问题：会把一个公式和它的成立前提切到两个块里——检索命中了，答案却答错。
这类"检索到了但答不对"的失败，很难通过换模型解决，只能靠分块策略。

本模块做法：
  1. 识别中文教材常见的章节标题，标题跟着它的正文走，不单独成块
  2. 标题路径记进块的元数据，检索命中后可以告诉用户"来自第 3.2 节"
  3. 正文按句末标点切分并保留重叠，尽量不切断一个完整的论述

★ 你们要做的实验：把「是否按标题感知切分」× 「块大小 400/700/1000」做对比，
  用评测集的 Recall@5 决定最终参数。这是报告里"创新性与方案设计"最实在的论据。
"""

from __future__ import annotations

import hashlib
import re

import config

# 中文教材里常见的标题形态
_HEADING_PATTERNS = [
    r"^第[一二三四五六七八九十百零\d]+[章节讲部分课]",
    r"^[一二三四五六七八九十]+[、.．]",
    r"^\d+(?:\.\d+){0,3}[、.．\s]",
    r"^【.{2,24}】",
    r"^[（(][一二三四五六七八九十\d]+[)）]",
]
_HEADING_RE = re.compile("|".join(_HEADING_PATTERNS))

# 句末标点：切分时优先在这些位置断开
_SENT_END_RE = re.compile(r"(?<=[。！？；;!?])")


def is_heading(line: str) -> bool:
    """判断一行是不是标题。

    经验规则：匹配标题形态，且长度不超过 40 字（太长的多半是正文）。
    """
    line = line.strip()
    if not line or len(line) > 40:
        return False
    return bool(_HEADING_RE.match(line))


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """把一段长文本按句子边界切成多块，块间保留 overlap 个字符的重叠。"""
    sentences = [s for s in _SENT_END_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        # 单句就超长（比如一整页公式），只能硬切
        if len(sent) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(sent), max_chars - overlap):
                chunks.append(sent[i:i + max_chars])
            continue

        if len(buf) + len(sent) <= max_chars:
            buf += sent
        else:
            chunks.append(buf)
            # 用上一块的尾部做重叠，保证跨块的论述不被割断
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = tail + sent
    if buf.strip():
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def _make_id(source: str, page: int, idx: int, text: str) -> str:
    """稳定的块 ID：同一份课件重新建索引时不会产生重复块。"""
    raw = f"{source}|{page}|{idx}|{text[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _merge_tiny(chunks: list[dict], min_chars: int, max_chars: int) -> list[dict]:
    """收尾清理：把过短的碎块并进相邻块。

    为什么还要单独做一遍：构建时虽然合并过一次，但那时只能往前看，
    遇到"文档第一行的标题""某章开头就是半句话"这类情况前面没有块可并，
    于是留下一个十几个字的碎块。它会污染块统计，也会在检索调试页里显得奇怪。
    """
    # 第一遍：往后并（处理开头碎块）
    forward: list[dict] = []
    carry: dict | None = None
    for c in chunks:
        if carry is not None and carry["source"] == c["source"] \
                and len(carry["body"]) + len(c["body"]) <= max_chars * 2:
            merged = dict(c)
            merged["body"] = carry["body"] + "\n" + c["body"]
            merged["text"] = merged["body"]
            # 出处取碎块所在位置（内容实际开始的地方）
            merged["page"] = carry["page"]
            merged["order"] = carry["order"]
            merged["id"] = carry["id"]
            carry = None
            forward.append(merged)
            continue
        if carry is not None:
            forward.append(carry)
            carry = None
        if len(c["body"]) < min_chars:
            carry = c            # 先攒着，看看能不能并进下一块
        else:
            forward.append(c)
    if carry is not None:
        forward.append(carry)

    # 第二遍：往前并（处理落在末尾的碎块）
    out: list[dict] = []
    for c in forward:
        if (len(c["body"]) < min_chars and out
                and out[-1]["source"] == c["source"]
                and len(out[-1]["body"]) + len(c["body"]) <= max_chars * 2):
            out[-1]["body"] = out[-1]["body"] + "\n" + c["body"]
            out[-1]["text"] = out[-1]["text"] + "\n" + c["body"]
            continue
        out.append(c)
    return out


def build_chunks(pages: list[dict],
                 max_chars: int | None = None,
                 overlap: int | None = None,
                 min_chars: int | None = None,
                 heading_aware: bool = True) -> list[dict]:
    """把解析出来的页级文本切成检索用的块。

    参数
    ----
    pages        : parse.parse_all() 的输出
    max_chars    : 每块最大字符数
    overlap      : 相邻块重叠字符数
    min_chars    : 小于此长度的块并入前一块
    heading_aware: 是否启用章节标题感知（关闭 = 退化成分块基线，用于对比实验）

    返回
    ----
    [{"id", "text", "source", "page", "heading", "order"}, ...]
    其中 text 会把标题路径拼在最前面——这样检索时"第3章 神经网络"这类
    关键词也能命中块内容。
    """
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    min_chars = min_chars or config.CHUNK_MIN_CHARS

    chunks: list[dict] = []
    heading = ""   # 跨页延续：一个章节标题会一直生效到下一个标题出现

    for page in pages:
        source, page_no = page["source"], page["page"]
        lines = [ln.strip() for ln in page["text"].splitlines()]

        buf: list[str] = []
        order_in_page = 0

        def flush(current_heading: str) -> None:
            """把缓冲区结算成块。"""
            nonlocal buf, order_in_page
            if not buf:
                return
            body = "\n".join(buf).strip()
            buf = []
            if not body:
                return

            pieces = _split_long_text(body, max_chars, overlap) or [body]
            for piece in pieces:
                # 太短的碎片并进上一块，避免产生"只有半句话"的无效块
                if (len(piece) < min_chars and chunks
                        and chunks[-1]["source"] == source
                        and len(chunks[-1]["text"]) + len(piece) <= max_chars * 2):
                    chunks[-1]["text"] += "\n" + piece
                    continue

                prefix = f"【{current_heading}】\n" if (heading_aware and current_heading) else ""
                chunks.append({
                    "id": _make_id(source, page_no, order_in_page, piece),
                    "text": prefix + piece,
                    "body": piece,          # 不含标题前缀的原文，用于对照实验
                    "source": source,
                    "page": page_no,
                    "heading": current_heading if heading_aware else "",
                    "order": order_in_page,
                })
                order_in_page += 1

        for line in lines:
            if not line:
                continue
            if heading_aware and is_heading(line):
                flush(heading)          # 遇到新标题，先把上一节结算掉
                heading = line[:60]
                buf.append(line)
            else:
                buf.append(line)
        flush(heading)

    before = len(chunks)
    chunks = _merge_tiny(chunks, min_chars, max_chars)
    if before != len(chunks):
        print(f"[chunk] 碎块合并：{before} → {len(chunks)} 块")

    print(f"[chunk] {len(pages)} 页 → {len(chunks)} 块"
          f"（heading_aware={heading_aware}, max_chars={max_chars}, overlap={overlap}）")
    return chunks


def chunk_stats(chunks: list[dict]) -> dict:
    """块的统计信息，写报告和做对比实验时直接引用。"""
    lens = [len(c["body"]) for c in chunks]
    lens.sort()
    if not lens:
        return {"count": 0}
    return {
        "count": len(lens),
        "min": lens[0],
        "median": lens[len(lens) // 2],
        "max": lens[-1],
        "avg": round(sum(lens) / len(lens), 1),
    }


if __name__ == "__main__":
    from src import parse

    demo_pages = parse.parse_all()
    demo = build_chunks(demo_pages)
    print(chunk_stats(demo))
    if demo:
        print("-" * 60)
        print(demo[0]["text"][:400])
