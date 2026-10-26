"""学习分析：薄弱知识点识别与复习清单（★ 自研点 4，差异化功能）

这是本作品区别于"一个能问课程的聊天窗口"的地方。

观察：学生反复问的，就是他没懂的。
做法：把每次提问记录下来，从命中的块里抽取知识点标签（章节标题 + 关键词），
      按出现频次排序，"被反复问到却每次都靠检索才能答上来"的知识点就是薄弱点。

产出：一份可以直接导出拿去复习的清单：
      「第 3.2 节 反向传播 —— 你问了 5 次」

★ 你们要做的改进：目前是词频统计，可以升级成按时间衰减加权
  （最近的提问权重更高，反映当前复习进度）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import config


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load() -> dict:
    if not config.PROFILE_FILE.exists():
        return {"records": []}
    try:
        with open(config.PROFILE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # 文件损坏时不要让整个应用挂掉，重建即可
        return {"records": []}


def _save(data: dict) -> None:
    config.DATA_INDEX.mkdir(parents=True, exist_ok=True)
    with open(config.PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record(question: str, hits: list[dict], refused: bool = False,
           check_summary: dict | None = None) -> None:
    """记录一次提问。每次问答后调用。

    check_summary 是逐句核验的汇总，记下来是为了能在学习分析里
    汇总"累计核验了多少条结论、其中多少条无依据"——
    这组数字是作品抗幻觉能力的直接证据。
    """
    data = _load()
    cs = check_summary or {}
    data["records"].append({
        "time": _now(),
        "question": question,
        "refused": refused,
        # 只记命中的章节标题，不存原文，控制文件体积
        "headings": [h.get("heading", "") for h in hits[:3] if h.get("heading")],
        "keywords": _keywords(question),
        "claims": int(cs.get("total", 0)),
        "unsupported": int(cs.get("unsupported", 0)),
    })
    _save(data)


def _keywords(text: str, topn: int = 6) -> list[str]:
    """从提问里抽关键词。用 jieba 的 TF-IDF 抽取，比词频更准。"""
    try:
        import jieba.analyse

        return jieba.analyse.extract_tags(text, topK=topn)
    except Exception:
        return []


def weak_points(min_count: int = 2, topn: int = 10) -> list[dict]:
    """汇总薄弱知识点：被问到 >= min_count 次的章节 / 关键词。

    返回 [{"label", "count", "last_time", "type"}]，按次数降序。
    """
    data = _load()
    stats: dict[str, dict] = {}

    for rec in data["records"]:
        for kind, values in (("heading", rec.get("headings", [])),
                             ("keyword", rec.get("keywords", []))):
            for value in values:
                key = f"{kind}:{value}"
                item = stats.setdefault(key, {
                    "label": value, "type": kind, "count": 0, "last_time": "",
                })
                item["count"] += 1
                if rec.get("time", "") > item["last_time"]:
                    item["last_time"] = rec["time"]

    items = [v for v in stats.values() if v["count"] >= min_count]
    items.sort(key=lambda x: (-x["count"], x["label"]))
    return items[:topn]


def summary() -> dict:
    """总体统计，界面上做概览卡片。"""
    data = _load()
    records = data["records"]
    claims = sum(r.get("claims", 0) for r in records)
    unsupported = sum(r.get("unsupported", 0) for r in records)
    return {
        "total": len(records),
        "refused": sum(1 for r in records if r.get("refused")),
        "weak": weak_points(),
        # 抗幻觉指标：累计核验的结论数、其中课件未直接支持的条数
        "claims": claims,
        "unsupported": unsupported,
        "unsupported_rate": round(unsupported / claims, 3) if claims else 0.0,
    }


def export_markdown() -> str:
    """导出复习清单（Markdown），学生可以直接存下来当复习提纲。"""
    s = summary()
    lines = [
        "# 我的薄弱知识点复习清单",
        "",
        f"累计提问 {s['total']} 次，其中 {s['refused']} 次课件中没有相关内容。",
    ]
    if s.get("claims"):
        lines.append(
            f"共核验 {s['claims']} 条结论，其中 {s['unsupported']} 条课件未直接支持"
            f"（无依据率 {s['unsupported_rate']:.1%}）。"
        )
    lines.append("")
    if not s["weak"]:
        lines.append("暂时还没有明显薄弱点——继续提问，问得越多分析越准。")
        return "\n".join(lines)

    lines.append("## 需要重点复习的知识点")
    lines.append("")
    lines.append("| 知识点 | 类型 | 提问次数 | 最近一次 |")
    lines.append("|---|---|---|---|")
    for item in s["weak"]:
        kind = "章节" if item["type"] == "heading" else "关键词"
        lines.append(
            f"| {item['label']} | {kind} | {item['count']} | {item['last_time'][:10]} |"
        )
    lines.append("")
    lines.append("> 提问次数越多，说明这个知识点掌握得越不牢。")
    return "\n".join(lines)


def reset() -> None:
    """清空记录（界面上留个按钮，演示前重置）。"""
    _save({"records": []})


if __name__ == "__main__":
    print(export_markdown())
