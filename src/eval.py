"""评测脚本：产出报告里最重的那个 25 分所要的数据。

评测集格式（eval/questions.jsonl，每行一个 JSON）：
    {"question": "反向传播中链式法则的作用是什么",
     "gold": [{"source": "第3章-神经网络.pdf", "page": 12}],
     "type": "answerable",
     "reference": "可选：参考答案，人工判分时当对照"}

    {"question": "这门课讲了量子计算吗",
     "gold": [],
     "type": "unanswerable"}     ← 用来测拒答能力

三种用法
--------
python -m src.eval --mode retrieve    # 只跑检索指标（快，不用大模型）
python -m src.eval --mode full        # 跑完整流程，导出待人工判分的答案
python -m src.eval --calibrate        # 扫描拒答阈值，推荐一个最优值

★ 对比实验就是靠 --mode retrieve 跑出来的：四组配置各跑一遍，
  把 Recall@5 填进报告，这就是"检索策略改进有效"的硬证据。

⚠️ 注意：拒答判断依赖重排分数，所以拒答相关指标只对开启了重排的配置有意义。
   纯向量 / 纯 BM25 的分数不在同一量纲上，跨配置比拒答率是错的。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import config
from src import retrieve

# 消融实验配置：从最朴素的基线开始，逐层叠加我们的设计。
# 报告里的核心证据就是这张表——每一层各自带来多少提升，一目了然。
# 这样呈现比"我们的方案 vs 基线"有力得多：它证明提升来自哪一层设计，而不是碰巧。
CONFIGS: dict[str, dict] = {
    "A_纯向量":       {"use_bm25": False, "use_rerank": False,
                      "use_mmr": False, "use_expansion": False},
    "B_纯BM25":       {"use_vector": False, "use_rerank": False,
                      "use_mmr": False, "use_expansion": False},
    "C_混合RRF":      {"use_rerank": False, "use_mmr": False, "use_expansion": False},
    "D_混合+重排":     {"use_mmr": False, "use_expansion": False},
    "E_+去冗余MMR":    {"use_expansion": False},
    "F_完整方案":      {},   # ← 混合 + 重排 + MMR + 查询扩展
}

KS = (1, 3, 5)
RESULT_DIR = config.EVAL_DIR / "results"


# ---------------------------------------------------------------- 评测集
def load_questions(path: Path | None = None) -> list[dict]:
    path = path or config.EVAL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"评测集不存在：{path}\n"
            f"请把 eval/questions.example.jsonl 复制为 questions.jsonl 并填写题目。"
        )
    items = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            # 跳过空行和两种注释行：# 说明文字、// 注释
            # （⚠️ 曾经只跳过 //，结果评测集文件开头的 # 说明把校准脚本搞崩了）
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if not line.startswith("{"):
                raise ValueError(f"第 {line_no} 行不是合法的题目行（应以 {{ 开头）：{line[:40]}")
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSON：{exc}") from exc

    n_unans = sum(1 for i in items if i.get("type") == "unanswerable")
    print(f"[eval] 载入 {len(items)} 道题（可回答 {len(items) - n_unans} 道，无答案 {n_unans} 道）")
    return items


def _answerable(items: list[dict]) -> list[dict]:
    return [i for i in items if i.get("type") != "unanswerable" and i.get("gold")]


def _gold_set(item: dict) -> set[tuple[str, int]]:
    return {(g["source"], int(g["page"])) for g in item["gold"]}


def _hit(hits: list[dict], gold: set[tuple[str, int]], k: int) -> bool:
    return any((h["source"], int(h["page"])) in gold for h in hits[:k])


# ---------------------------------------------------------------- 指标
def eval_config(items: list[dict], kwargs: dict, show_progress: bool = True) -> dict:
    """对一组检索配置计算全部指标。

    关键优化：每道题**只检索一次**（取 top_k = max(KS)），
    Recall@1/3/5 和 MRR 都从这一次结果里派生。
    否则每题要检索 4 遍，120 道题的评测会从几分钟变成一个多小时。
    """
    pool = _answerable(items)
    if not pool:
        return {}

    top_k = max(KS)
    hit_counts = {k: 0 for k in KS}
    rr_sum = 0.0

    iterator = pool
    if show_progress and len(pool) > 20:
        try:
            from tqdm import tqdm
            iterator = tqdm(pool, desc="    检索中", ncols=70, leave=False)
        except ImportError:
            pass

    for item in iterator:
        hits = retrieve.hybrid_search(item["question"], top_k=top_k, **kwargs)
        gold = _gold_set(item)
        for k in KS:
            if _hit(hits, gold, k):
                hit_counts[k] += 1
        for rank, h in enumerate(hits, start=1):
            if (h["source"], int(h["page"])) in gold:
                rr_sum += 1.0 / rank
                break

    n = len(pool)
    row = {f"Recall@{k}": round(hit_counts[k] / n, 3) for k in KS}
    row["MRR"] = round(rr_sum / n, 3)
    return row


def category_breakdown(items: list[dict],
                       baseline: dict,
                       ours: dict,
                       labels: tuple[str, ...] = ("基线", "本作品")) -> list[dict]:
    """按题型拆解 Recall@5，回答"提升到底来自哪类问题"。

    为什么这一步很值得做：总体 Recall 提升 3 个百分点，
    听起来像是"碰巧赢了"；但如果拆开看是"条件题 +18 个百分点、定义题 0"，
    就说明改进精确命中了词面不一致这个真实痛点——这是设计有效的证据，
    而不只是一个更高的数字。

    评测集里给题目加 "category" 字段即可启用（定义题/条件题/对比题/跨页题/术语题）。
    没写 category 的题目归入"未分类"。
    """
    pool = _answerable(items)
    cats: dict[str, list[dict]] = {}
    for item in pool:
        cats.setdefault(item.get("category") or "未分类", []).append(item)

    if len(cats) <= 1 and "未分类" in cats:
        return []   # 没有分类信息，跳过

    rows = []
    for cat, group in cats.items():
        row = {"题型": cat, "题数": len(group)}
        for label, kwargs in ((labels[0], baseline), (labels[1], ours)):
            hit = 0
            for item in group:
                hits = retrieve.hybrid_search(item["question"], top_k=max(KS), **kwargs)
                if _hit(hits, _gold_set(item), max(KS)):
                    hit += 1
            row[label] = round(hit / len(group), 3)
        row["差值"] = round(row[labels[1]] - row[labels[0]], 3)
        rows.append(row)

    rows.sort(key=lambda r: -abs(r["差值"]))
    return rows


# ---------------------------------------------------------------- 阈值校准
def calibrate_threshold(items: list[dict]) -> list[dict]:
    """扫描拒答阈值，找出能同时兼顾"该答的答对"和"该拒的拒掉"的值。

    做法：把可回答题的 top_score 分布 和 无答案题的 top_score 分布分别算出来，
    看哪个阈值能把两者分开。
    """
    pool = _answerable(items)
    unans = [i for i in items if i.get("type") == "unanswerable"]

    if not pool or not unans:
        print("[calibrate] 评测集里需要同时有可回答题和无答案题，才能校准阈值。")
        return []

    ans_scores = [retrieve.top_score(retrieve.hybrid_search(i["question"])) for i in pool]
    unans_scores = [retrieve.top_score(retrieve.hybrid_search(i["question"])) for i in unans]

    print("\n[calibrate] 分数分布")
    print(f"  可回答题  top_score: 最低 {min(ans_scores):.4f}  最高 {max(ans_scores):.4f}")
    print(f"  无答案题  top_score: 最低 {min(unans_scores):.4f}  最高 {max(unans_scores):.4f}")

    rows = []
    for step in range(1, 20):          # 阈值从 0.05 扫到 0.95
        th = step * 0.05
        tpr = sum(1 for s in ans_scores if s >= th) / len(ans_scores)   # 该答的没被误拒
        tnr = sum(1 for s in unans_scores if s < th) / len(unans_scores)  # 该拒的拒掉了
        f1 = 2 * tpr * tnr / (tpr + tnr) if (tpr + tnr) > 0 else 0.0
        rows.append({"阈值": round(th, 2), "可回答题未被误拒": round(tpr, 3),
                     "无答案题正确拒答": round(tnr, 3), "综合": round(f1, 3)})

    best = max(rows, key=lambda r: r["综合"])
    print(f"\n{'阈值':>6} {'可回答题未被误拒':>16} {'无答案题正确拒答':>16} {'综合':>8}")
    for r in rows:
        mark = "  ← 推荐" if r is best else ""
        print(f"{r['阈值']:>6.2f} {r['可回答题未被误拒']:>16.3f} "
              f"{r['无答案题正确拒答']:>16.3f} {r['综合']:>8.3f}{mark}")

    # 更稳的推荐方式：取"无答案题上限"和"可回答题下限"这段空隙的中点。
    # 直接取扫描表里分数最高的那个阈值往往贴着边界（余量小，换一批题就失效），
    # 取中点则两边留出的安全距离最大。
    lo, hi = max(unans_scores), min(ans_scores)
    if lo < hi:
        suggested = round((lo + hi) / 2, 2)
        print(f"\n[calibrate] 分数空隙：无答案题最高 {lo:.4f}，可回答题最低 {hi:.4f}")
        print(f"            空隙宽度 {(hi - lo):.4f}，中点 {suggested}")
        print(f"[calibrate] 推荐阈值 {suggested}（离两侧边界都最远，最稳）")
    else:
        suggested = best["阈值"]
        print(f"\n[calibrate] ⚠️ 可回答题与无答案题的分数区间出现重叠"
              f"（无答案题最高 {lo:.4f} > 可回答题最低 {hi:.4f}）")
        print(f"            说明仅靠分数无法完全区分，建议：")
        print(f"            1) 补充「答案级拒答检测」（src/answer.py 已实现，会自动兜底）")
        print(f"            2) 检查评测集里是否有标注错的 gold 页码")
        print(f"            3) 此时取综合最优值 {suggested}")

    print(f"\n请把 config.py 里的 MIN_RERANK_SCORE 改为 {suggested}（当前值 {config.MIN_RERANK_SCORE}）")
    return rows


# ---------------------------------------------------------------- 主流程
def run(mode: str, only: list[str] | None = None,
        skip_ablation: bool = False) -> None:
    items = load_questions()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    configs = CONFIGS
    if only:
        configs = {k: v for k, v in CONFIGS.items()
                   if any(k.startswith(p) for p in only)}
        if not configs:
            print(f"[eval] --configs 没有匹配到任何配置，可用：{list(CONFIGS)}")
            return
        print(f"[eval] 只运行 {len(configs)} 组配置：{list(configs)}")

    print(f"\n{'配置':<16} " + " ".join(f"{f'Recall@{k}':>10}" for k in KS) + f" {'MRR':>8}")
    print("-" * (18 + 11 * len(KS) + 9))

    table = []
    out_csv = RESULT_DIR / "retrieval_compare.csv"
    if skip_ablation and out_csv.exists():
        # 上次已跑完消融：直接复用已有对比表，省掉最重的重排计算
        with open(out_csv, encoding="utf-8-sig") as f:
            table = [{"配置": r["配置"],
                      **{f"Recall@{k}": float(r[f"Recall@{k}"]) for k in KS},
                      "MRR": float(r["MRR"])}
                     for r in csv.DictReader(f)]
        print(f"[eval] --skip-ablation：复用已有对比表 {out_csv}")
    else:
        for name, kwargs in configs.items():
            row = eval_config(items, kwargs)
            record = {"配置": name, **row}
            table.append(record)
            print(f"{name:<16} " + " ".join(f"{row[f'Recall@{k}']:>10.3f}" for k in KS)
                  + f" {row['MRR']:>8.3f}")
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)
        print(f"\n[eval] 对比结果已保存：{out_csv}")

    # 查询扩展要调用大模型改写问题，而大模型生成带有随机性，
    # 所以带 E/F 两组的结果在不同次运行之间会有小幅波动。
    # 报告里引用这两组的数字时，建议注明"多次运行取平均"或说明存在波动。
    if any(t["配置"].startswith(("E_", "F_")) for t in table):
        print("[eval] 说明：E/F 两组启用了查询扩展（依赖大模型改写），"
              "多次运行结果会有小幅波动，建议重复 2–3 次取平均后再写入报告。")

    # --- 逐层增量：说明提升来自哪一层设计，而不是碰巧 ---
    print("\n[eval] 逐层增量（相对上一行配置的 Recall@5 变化）")
    print(f"{'配置':<16} {'Recall@5':>10} {'增量':>10}")
    print("-" * 40)
    prev = None
    for row in table:
        delta = "" if prev is None else f"{(row['Recall@5'] - prev) * 100:+.1f} pt"
        print(f"{row['配置']:<16} {row['Recall@5']:>10.3f} {delta:>10}")
        prev = row["Recall@5"]

    # --- 结论：最强基线 vs 完整方案 ---
    baselines = [t for t in table if t["配置"] in ("A_纯向量", "B_纯BM25")]
    final = next((t for t in table if t["配置"].startswith("F_")), None)
    if baselines and final:
        best_base = max(baselines, key=lambda t: t["Recall@5"])
        gain = (final["Recall@5"] - best_base["Recall@5"]) * 100
        print(f"\n[eval] 最强基线（{best_base['配置']}）Recall@5 = {best_base['Recall@5']:.3f}")
        print(f"       完整方案（{final['配置']}）Recall@5 = {final['Recall@5']:.3f}"
              f"　→ {'提升' if gain >= 0 else '下降'} {abs(gain):.1f} 个百分点")
        if gain <= 0:
            print("[eval] ⚠️ 完整方案没有跑赢最强基线。先查这三处：")
            print("       1) 评测集的 gold 页码是否标注正确（最常见）")
            print("       2) 题目是否过于简单——全是「XX 的定义」这类题，任何方法都能命中")
            print("          应补充条件题、对比题、跨页题、术语编号题")
            print("       3) 课件规模是否太小——块数少于 50 时取前 5 条就覆盖大半语料，")
            print("          天花板效应会把所有方法的差异抹平")
            print("       若查完确实是负结果，照实写进报告并分析原因，不要改数据。")

        # --- 按题型拆解：提升来自哪类问题 ---
        base_name, final_name = best_base["配置"], final["配置"]
        breakdown = (None if skip_ablation else
                     category_breakdown(items, configs[base_name], configs[final_name],
                                        labels=(base_name, final_name)))
        if breakdown:
            print(f"\n[eval] 按题型拆解 Recall@5（{base_name} → {final_name}）")
            print(f"{'题型':<10} {'题数':>6} {base_name:>13} {final_name:>13} {'差值':>9}")
            print("-" * 58)
            for r in breakdown:
                print(f"{r['题型']:<10} {r['题数']:>6} {r[base_name]:>13.3f} "
                      f"{r[final_name]:>13.3f} {r['差值']:>+9.3f}")
            print("提升集中在哪些题型，就是这套设计真正解决的问题——"
                  "报告里引用这一节，比只给一个总分有说服力得多。")

    # --- 满流程：导出答案供人工判分，并统计抗幻觉指标 ---
    if mode == "full":
        from src import answer

        out_jsonl = RESULT_DIR / "answers_to_grade.jsonl"
        verify_rows = []          # 核验抽查表：给人工检查核验器判断对不对用
        claims_total = 0
        unsupported_total = 0
        rev_before_claims = rev_before_unsup = 0
        rev_after_unsup = 0
        n_revised = 0

        # 断点续跑：只复用带「修订前/后统计」字段的新版记录；
        # 旧版记录和出错记录一律重跑，保证统计口径一致。
        done: dict[str, dict] = {}
        if out_jsonl.exists():
            with open(out_jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "修订前无依据" in rec and "answer" in rec:
                        done[rec["question"]] = rec
            if done:
                print(f"[eval] 断点续跑：已有 {len(done)} 题结果，跳过不重跑")

        def _tally(rec: dict) -> None:
            nonlocal claims_total, unsupported_total
            nonlocal rev_before_claims, rev_before_unsup, rev_after_unsup, n_revised
            claims_total += rec.get("核验结论数", 0)
            unsupported_total += rec.get("其中无依据", 0)
            for c in rec.get("checks", []):
                e = c["evidence"][0] if c["evidence"] else {}
                verify_rows.append({
                    "问题": rec["question"],
                    "结论": c["claim"],
                    "系统判断": "有依据" if c["supported"] else "无依据",
                    "相关性": c["best_score"],
                    "依据原文（前150字）": (e.get("text", "") or "")[:150],
                    "依据出处": (f"{e.get('source', '')} 第{e.get('page', '')}页" if e else ""),
                    "人工判定（对/错）": "",
                })
            if rec.get("触发自动修订"):
                n_revised += 1
                rev_before_claims += rec.get("修订前结论数", 0)
                rev_before_unsup += rec.get("修订前无依据", 0)
                rev_after_unsup += rec.get("修订后无依据", 0)

        if done:
            # 按题目顺序重写文件，丢掉旧版/出错记录，再以追加模式续写
            tmp = out_jsonl.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for item in items:
                    rec = done.get(item["question"])
                    if rec:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.replace(out_jsonl)

        with open(out_jsonl, "a" if done else "w", encoding="utf-8") as f:
            for item in items:
                if item["question"] in done:
                    _tally(done[item["question"]])
                    continue
                try:
                    result = answer.answer_question(item["question"])
                    cs = result.get("check_summary") or {}
                    claims_total += cs.get("total", 0)
                    unsupported_total += cs.get("unsupported", 0)
                    for c in result.get("checks", []):
                        e = c["evidence"][0] if c["evidence"] else {}
                        verify_rows.append({
                            "问题": item["question"],
                            "结论": c["claim"],
                            "系统判断": "有依据" if c["supported"] else "无依据",
                            "相关性": c["best_score"],
                            "依据原文（前150字）": (e.get("text", "") or "")[:150],
                            "依据出处": (f"{e.get('source', '')} 第{e.get('page', '')}页" if e else ""),
                            "人工判定（对/错）": "",
                        })
                    rev = result.get("revision")
                    if rev:
                        n_revised += 1
                        rev_before_claims += rev["before"].get("total", 0)
                        rev_before_unsup += rev["before"].get("unsupported", 0)
                        rev_after_unsup += rev["after"].get("unsupported", 0)
                    record = {
                        "question": item["question"],
                        "reference": item.get("reference", ""),
                        "category": item.get("category", ""),
                        "answer": result["answer"],
                        "refused": result["refused"],
                        "top_score": round(result["top_score"], 4),
                        "expected_type": item.get("type", "answerable"),
                        "citations": result["citations"],
                        "核验结论数": cs.get("total", 0),
                        "其中无依据": cs.get("unsupported", 0),
                        "触发自动修订": bool(rev),
                        "修订前结论数": rev["before"].get("total", 0) if rev else 0,
                        "修订前无依据": rev["before"].get("unsupported", 0) if rev else 0,
                        "修订后结论数": rev["after"].get("total", 0) if rev else 0,
                        "修订后无依据": rev["after"].get("unsupported", 0) if rev else 0,
                        "checks": result.get("checks", []),
                        "人工评分": "",      # ← 三个人分别填 对/部分对/错
                        "核验抽查": "",      # ← 抽查 checks 里的 supported 判断是否正确，填 对/错
                    }
                except Exception as exc:
                    record = {"question": item["question"], "error": str(exc)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()   # 每题落盘，进程崩溃不丢已完成的题
        print(f"[eval] 待判分答案已导出：{out_jsonl}")
        print("      三个人各自打开这个文件：①「人工评分」栏填 对 / 部分对 / 错，统计答案准确率；"
              "②「核验抽查」栏抽查 supported 判断是否正确，统计核验器自身的准确率。")

        if verify_rows:
            out_csv = RESULT_DIR / "核验抽查表.csv"
            with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(verify_rows[0].keys()))
                writer.writeheader()
                writer.writerows(verify_rows)
            print(f"[eval] 核验抽查表已导出：{out_csv}")
            print("      用 Excel 直接打开：每行是系统的一个判断，人工在最后一列填 对/错，"
                  "统计「对」的比例就是核验器准确率。")

        # --- 抗幻觉指标：逐句核验 + 自动修订的闭环效果 ---
        if claims_total:
            rate = unsupported_total / claims_total
            print(f"\n[eval] 抗幻觉核验：共核验 {claims_total} 条结论，"
                  f"其中 {unsupported_total} 条课件未直接支持"
                  f"（无依据率 {rate:.1%}）")
            if n_revised:
                print(f"[eval] 自动修订闭环：{n_revised} 条回答触发了修订，"
                      f"无依据结论 {rev_before_unsup} → {rev_after_unsup} 条")
                if rev_before_unsup > 0:
                    fixed = rev_before_unsup - rev_after_unsup
                    print(f"       闭环修复了 {fixed} 条"
                          f"（修复率 {fixed / rev_before_unsup:.0%}）——"
                          f"这是「生成→核验→修订」闭环有效性的直接证据")
            print("      把这些数字写进报告的「成果展示与效果验证」——"
                  "它是「答案可不可信」的量化证据，而不是主观声称。")
        else:
            print("\n[eval] 未产生核验数据（可能所有问题都被拒答了）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="评测")
    ap.add_argument("--mode", choices=["retrieve", "full"], default="retrieve")
    ap.add_argument("--calibrate", action="store_true", help="扫描拒答阈值")
    ap.add_argument("--configs", default=None,
                    help="只跑部分配置，逗号分隔的配置名前缀，"
                         "例如 --configs A,D,F（迭代调参时可省时间）")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="复用已有的 retrieval_compare.csv，直接进入满流程阶段")
    args = ap.parse_args()

    if args.calibrate:
        calibrate_threshold(load_questions())
    else:
        only = [s.strip() for s in args.configs.split(",")] if args.configs else None
        run(args.mode, only=only, skip_ablation=args.skip_ablation)
