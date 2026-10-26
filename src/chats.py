"""会话历史持久化（新建对话时自动归档，可随时查看/删除）。

存在 data/index/chats.json，跨重启保留。
存的是"精简版"消息：assistant 消息只保留展示所需的字段，
不存检索中间结果，避免文件无限膨胀。
"""

from __future__ import annotations

import json
import time
import uuid

import config

CHAT_FILE = config.DATA_INDEX / "chats.json"


def _load() -> list[dict]:
    if not CHAT_FILE.exists():
        return []
    try:
        with open(CHAT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save(data: list[dict]) -> None:
    config.DATA_INDEX.mkdir(parents=True, exist_ok=True)
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def _compact_result(result: dict) -> dict:
    """assistant 消息里的 result 只留展示要用的字段。"""
    hits = [{
        "rank": h["rank"], "score": h["score"], "text": h["text"][:800],
        "source": h["source"], "page": h["page"], "heading": h.get("heading", ""),
    } for h in result.get("hits", [])]
    return {
        "answer": result.get("answer", ""),
        "refused": result.get("refused", False),
        "top_score": result.get("top_score", 0.0),
        "citations": result.get("citations", []),
        "hits": hits,
        # 逐句核验结果也存下来，否则从历史恢复后核验标记就没了
        "checks": result.get("checks", []),
        "check_summary": result.get("check_summary",
                                    {"total": 0, "unsupported": 0, "unsupported_rate": 0.0}),
    }


def archive_chat(messages: list[dict]) -> str | None:
    """把当前对话归档。没有实质内容时返回 None。"""
    if not any(m["role"] == "user" for m in messages):
        return None

    title = next((m["content"] for m in messages if m["role"] == "user"), "")[:30]
    compact = []
    for m in messages:
        if m["role"] == "user":
            compact.append({"role": "user", "content": m["content"]})
        else:
            compact.append({"role": "assistant",
                            "result": _compact_result(m["result"])})

    data = _load()
    cid = uuid.uuid4().hex[:10]
    data.append({
        "id": cid,
        "title": title or "未命名对话",
        "time": time.strftime("%m-%d %H:%M"),
        "messages": compact,
    })
    _save(data)
    return cid


def list_chats() -> list[dict]:
    """历史对话列表，新的在前。"""
    return sorted(_load(), key=lambda c: c.get("time", ""), reverse=True)


def load_chat(cid: str) -> list[dict]:
    """取出一条历史对话的消息（可直接回填 session_state["messages"]）。"""
    for c in _load():
        if c["id"] == cid:
            return c["messages"]
    return []


def delete_chat(cid: str) -> None:
    _save([c for c in _load() if c["id"] != cid])
