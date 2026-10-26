"""文档解析：把 PDF / PPTX / DOCX 统一成带页码的文本。

全项目通用的数据结构（后面所有模块都吃这个格式）：
    {"source": "第3章-神经网络.pdf", "page": 12, "text": "..."}

page 的含义：PDF 是页码，PPT 是第几页幻灯片，Word 统一记 1。
保留 page 是为了让答案能标注"出自第几页"，这是本作品的核心卖点之一。

⚠️ PDF 解析用 pdfplumber（MIT），刻意不用 PyMuPDF（AGPL-3.0）。
"""

from __future__ import annotations

import re
from pathlib import Path

import config

SUPPORTED_SUFFIXES = {".pdf", ".pptx", ".docx"}


# ---------------------------------------------------------------- 文本清洗
# 页眉页脚是检索噪声的大敌：同一句话在几十页重复出现，
# 会把 BM25 的分数带偏。下面这组规则能干掉大部分。
_PAGE_NUMBER_ONLY = re.compile(r"^\s*[-—]?\s*\d{1,4}\s*[-—]?\s*$")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t\u3000]{2,}")


def clean_text(text: str) -> str:
    """去掉纯页码行、压缩多余空白。"""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _PAGE_NUMBER_ONLY.match(line):
            continue
        lines.append(_MULTI_SPACE.sub(" ", line))
    return _MULTI_BLANK.sub("\n\n", "\n".join(lines)).strip()


# ---------------------------------------------------------------- 各格式解析
def parse_pdf(path: Path) -> list[dict]:
    import pdfplumber

    pages: list[dict] = []
    total = 0
    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"source": path.name, "page": page_no, "text": text})

    # 扫描版 PDF（整页是图片、没有文字层）在课件里很常见。
    # 它的表现很隐蔽：不报错、只是页数少，最后检索效果差却找不到原因。
    # 所以这里主动提示，省得白白排查半天。
    if total and len(pages) < total * 0.3:
        print(f"[parse] ⚠️ {path.name}：{total} 页里只有 {len(pages)} 页提到文字，"
              f"很可能是扫描版（图片型）PDF。本工具不处理这种情况，"
              f"建议换用原始 PPT，或用带 OCR 的工具先转成文字版。")
    return pages


def parse_pptx(path: Path) -> list[dict]:
    from pptx import Presentation

    slides: list[dict] = []
    prs = Presentation(str(path))
    for slide_no, slide in enumerate(prs.slides, start=1):
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text)
            # 表格内容也常常是考点（比如对比表格）
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = clean_text("\n".join(parts))
        if text:
            slides.append({"source": path.name, "page": slide_no, "text": text})
    return slides


def parse_docx(path: Path) -> list[dict]:
    import docx

    doc = docx.Document(str(path))
    text = clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    return [{"source": path.name, "page": 1, "text": text}] if text else []


_PARSERS = {".pdf": parse_pdf, ".pptx": parse_pptx, ".docx": parse_docx}


def parse_document(path: Path) -> list[dict]:
    """解析单个文件。不支持的格式返回空列表（不抛异常，方便批量跑）。"""
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        print(f"[parse] 跳过不支持的文件：{path.name}")
        return []
    try:
        pages = parser(path)
    except Exception as exc:  # 单个文件损坏不该让整批解析失败
        print(f"[parse] 解析失败 {path.name}：{exc}")
        return []
    print(f"[parse] {path.name} → {len(pages)} 页/页幻灯片")
    return pages


def _load_private_patterns() -> list[str]:
    """加载本地隐私过滤关键词（学校名、教师名、邮箱域名等）。

    规则文件 data/private_filter.txt 不进版本库——它本身含有学校信息。
    没有这个文件就不过滤，功能完全可迁移到任何学校。
    """
    import config

    f = config.PRIVATE_FILTER_FILE
    if not f.exists():
        return []
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def filter_private_pages(pages: list[dict]) -> list[dict]:
    """剔除含学校/教师身份信息的页面（比赛匿名性条款的自动化保障）。

    为什么放在解析层而不是界面层：一旦这类页进了索引，
    检索、引用、演示视频里都会露出来，事后删页要重建整条链路；
    在源头拦掉最干净。
    """
    patterns = _load_private_patterns()
    if not patterns or not pages:
        return pages

    kept, dropped = [], []
    for p in pages:
        if any(pat in p["text"] for pat in patterns):
            dropped.append(p["page"])
        else:
            kept.append(p)
    if dropped:
        print(f"[parse] 隐私过滤：已剔除 {len(dropped)} 页含学校/教师信息的页面"
              f"（第 {', '.join(map(str, dropped))} 页），规则见 data/private_filter.txt")
    return kept


def parse_all(raw_dir: Path | None = None) -> list[dict]:
    """解析 data/raw/ 下所有支持的文档，按文件名排序保证结果可复现。"""
    raw_dir = raw_dir or config.DATA_RAW
    if not raw_dir.exists():
        raise FileNotFoundError(f"课件目录不存在：{raw_dir}（请把课件放进去）")

    files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"{raw_dir} 里没有找到课件。支持格式：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    pages: list[dict] = []
    for path in files:
        pages.extend(parse_document(path))
    pages = filter_private_pages(pages)
    print(f"[parse] 共 {len(files)} 个文件，{len(pages)} 页，"
          f"{sum(len(p['text']) for p in pages)} 字")
    return pages


if __name__ == "__main__":
    for item in parse_all()[:3]:
        print("-" * 60)
        print(f"{item['source']} 第 {item['page']} 页")
        print(item["text"][:300])
