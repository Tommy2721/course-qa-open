"""把《出题分工表》转换成评测集 questions.jsonl。

支持两种输入（自动识别后缀）：
    python eval/csv_to_jsonl.py 出题分工表.xlsx          # 队友直接交的 Excel
    python eval/csv_to_jsonl.py 出题分工表.csv           # 或导出的 CSV
    加 --apply  才会覆盖 eval/questions.jsonl（默认只写预览文件并打印检查结果）

转换器会自动处理：
- 多页标注："16；24" / "16;24" / "16、24" → 拆成多个 gold 出处
- 行首的示例行 / 编号缺失行 → 按内容判断，示例行自动跳过
- 引用了不存在的课件文件名 → 跳过并警告（防止旧课件残留混进评测）
- 核对结果异常（非"已核对/对/是"）→ 跳过并警告，等核对完再转
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT_PREVIEW = HERE / "questions_from_table.jsonl"
OUT_FINAL = HERE / "questions.jsonl"

SEP_RE = re.compile(r"[；;、,，\s]+")          # 多页分隔符
VALID_TYPES = ("定义题", "条件题", "对比题", "跨页题", "术语题", "无答案题")


def read_rows(table: Path) -> list[dict]:
    if table.suffix.lower() == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(table, data_only=True)
        rows = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                rows.append([str(c).strip() if c is not None else "" for c in row])
        return rows
    import csv
    with open(table, encoding="utf-8-sig") as f:
        return [[c.strip() for c in row] for row in csv.reader(f)]


def to_rows_of_dict(rows: list[list[str]]) -> list[dict]:
    """把表格行按表头映射成字典（兼容不同列顺序，按列名找）。"""
    header_idx = 0
    for i, row in enumerate(rows):
        if any("题目" in c for c in row):
            header_idx = i
            break
    header = rows[header_idx]
    cols = {}
    for ci, name in enumerate(header):
        if "题目" in name:
            cols["question"] = ci
        elif "文件名" in name:
            cols["source"] = ci
        elif name == "页码":
            cols["page"] = ci
        elif "题型" in name:
            cols["category"] = ci
        elif "核对结果" in name:
            cols["checked"] = ci
    out = []
    for row in rows[header_idx + 1:]:
        if not any(row):
            continue
        out.append({k: (row[ci] if ci < len(row) else "") for k, ci in cols.items()})
    return out


def norm_name(s: str) -> str:
    """文件名归一化：去空格、统一各种横线字符。队友手打文件名常有不可见差异。"""
    return re.sub(r"[\s－–—―‐]+", "-", s)


def main(apply: bool = False, table: Path | None = None) -> None:
    table = table or (HERE / "出题分工表.csv")
    if not table.exists():
        print(f"[table] 没找到 {table}")
        return

    raw_dir = HERE.parent / "data" / "raw"
    raw_names = {p.name for p in raw_dir.iterdir()} if raw_dir.exists() else set()
    raw_norm = {norm_name(n): n for n in raw_names}

    def resolve(src: str) -> str | None:
        """表格里的文件名 → 真实课件文件名（精确匹配失败后按讲次编号模糊匹配）。"""
        n = norm_name(src)
        if n in raw_norm:
            return raw_norm[n]
        m = re.search(r"(\d+)\s*(?:-|\.pptx|\.pdf|\.docx|$)", n)
        if m:
            num = m.group(1)
            for normed, real in raw_norm.items():
                if num in normed and re.search(rf"{num}\s*-", normed):
                    return real
        return None

    items, skipped = [], []
    seen_questions = set()
    for row in to_rows_of_dict(read_rows(table)):
        q = row.get("question", "")
        if not q or q.startswith(("例", "编号")):
            continue                                   # 示例行 / 表头
        if q in seen_questions:
            skipped.append((q, "重复题"))
            continue
        src = row.get("source", "")
        cat = row.get("category", "")
        checked = row.get("checked", "")

        item = {"question": q, "gold": [], "type": "answerable", "category": cat}
        if "无答案" in cat:
            item["type"] = "unanswerable"
        else:
            real = resolve(src)
            if real is None:
                skipped.append((q, f"课件文件名对不上任何一份真实课件：{src or '(空)'}"))
                continue
            pages = [p for p in SEP_RE.split(row.get("page", "")) if p.isdigit()]
            if not pages:
                skipped.append((q, "页码没填或格式不对"))
                continue
            item["gold"] = [{"source": real, "page": int(p)} for p in pages]
        if checked and not any(k in checked for k in ("已核对", "对", "是")):
            skipped.append((q, f"还没交叉核对（{checked}）——核对完再转"))
            continue
        seen_questions.add(q)
        items.append(item)

    print(f"[table] 有效题目 {len(items)} 道"
          f"（可回答 {sum(1 for i in items if i['type'] == 'answerable')}，"
          f"无答案 {sum(1 for i in items if i['type'] == 'unanswerable')}）")
    cats: dict[str, int] = {}
    for i in items:
        cats[i["category"]] = cats.get(i["category"], 0) + 1
    print("[table] 题型分布:", "，".join(f"{k}{v}道" for k, v in cats.items()) or "无")
    for q, why in skipped:
        print(f"  ⚠️ 跳过「{q[:24]}」：{why}")

    if apply:
        # 合并模式（★ 不是覆盖）：保留现有评测集，只把表格里的新题补进去。
        # 按题目文本去重——现有题和表格题撞题时保留现有的（已核对过）。
        existing = []
        if OUT_FINAL.exists():
            with open(OUT_FINAL, encoding="utf-8") as f:
                existing = [json.loads(l) for l in f if l.strip().startswith("{")]
        have = {e["question"].strip() for e in existing}
        added = 0
        for item in items:
            if item["question"].strip() in have:
                continue
            existing.append(item)
            have.add(item["question"].strip())
            added += 1
        with open(OUT_FINAL, "w", encoding="utf-8") as f:
            for item in existing:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[table] 合并完成：新增 {added} 道，评测集现在共 {len(existing)} 道 → {OUT_FINAL}")
    else:
        with open(OUT_PREVIEW, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[table] 已写入预览 {OUT_PREVIEW}（--apply 未加，questions.jsonl 未被改动）")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--apply"]
    table = Path(args[0]) if args else None
    main(apply="--apply" in sys.argv, table=table)
