"""Streamlit 界面。

启动：streamlit run app.py

四个页签对应答辩时要演示的四件事：
  1. 问答 —— 答案带页码出处、不知道会拒答
  2. 学习分析 —— 薄弱知识点清单，能导出
  3. 检索调试 —— 各路召回分数全透明，证明"这个答案是怎么被找出来的"
  4. 课件管理 —— 界面里直接上传 PPT/Word/PDF、管理文件、重建索引
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

import config
from src import answer as answer_mod
from src import chats as chats_mod
from src import index as index_mod
from src import parse as parse_mod
from src import profile as profile_mod
from src import retrieve

st.set_page_config(page_title="课问 · 课程知识问答与复习助手", page_icon="📘", layout="wide")

ALLOWED_SUFFIXES = {".pdf", ".pptx", ".docx"}

# 把上传组件里 Streamlit 自带的英文文字换成中文（纯样式，不影响功能）。
# 选择器来自 1.64 版实际 DOM；升级 Streamlit 后若失效，只是退回英文显示，无功能影响。
st.markdown(
    """
    <style>
    /* "Upload" 按钮 → "浏览文件"（按钮内所有元素字号清零，再用 ::after 写中文） */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * { font-size: 0 !important; }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "浏览文件"; font-size: 0.95rem; white-space: nowrap;
    }
    /* 隐藏 "Drag and drop / 200MB per file" 英文说明，改用下方中文说明 */
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
    /* 屏蔽 Streamlit 右上角的英文推广弹窗（"Help agents write better apps" 等，
       testid 来自 1.64 版实际 DOM） */
    [data-testid="stSkillsNudge"],
    [data-testid="stSkillsNudgeAnchor"],
    [data-testid="stToast"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- 资源加载
def index_ready() -> tuple[bool, str]:
    """检查索引是否已建好。

    刻意不缓存：重建索引后状态必须立刻刷新。
    这里每次读一遍 chunks.jsonl（几千行，毫秒级），
    比缓存后还要记得清缓存简单得多。
    """
    try:
        chunks = index_mod.load_chunks()
        return True, f"已索引 {len(chunks)} 个知识块"
    except FileNotFoundError as exc:
        return False, str(exc)


@st.cache_data(show_spinner=False)
def document_list() -> list[str]:
    data = index_mod.load_chunks()
    return sorted({c["source"] for c in data})


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[bool, str]:
    """在页面打开时就把两个模型加载好并常驻进程。

    ⚠️ 踩过的坑（务必保留这个设计）：
    如果只把模型实例放在模块级全局变量里，一旦 Streamlit 的文件监视器
    重载了模块，全局变量就被清空，于是每次交互都要重新加载模型
    （bge-m3 + bge-reranker 在 CPU 上各要十几秒），界面会一直卡在
    "检索课件并生成答案……"转不出来。
    用 st.cache_resource 才能保证整个进程只加载一次。

    另一条同样重要的经验：演示或录视频前先打开页面预热一次，
    否则评委看到的第一问会等很久。启动命令也建议加上
    --server.fileWatcherType none 关掉文件监视。
    """
    try:
        index_mod.get_embedder()
        retrieve.get_reranker()
        return True, "模型已就绪"
    except Exception as exc:
        return False, str(exc)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("flash", None)


def show_flash() -> None:
    """跨 rerun 显示一次性提示（按钮回调后用 st.rerun 刷新页面，提示会丢）。"""
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if st.session_state.get("flash_err"):
        st.error(st.session_state.pop("flash_err"))


# ---------------------------------------------------------------- 重建索引（后台子进程 + 前端轮询）
# ⚠️ 不能在界面进程里同步建索引：向量化上百个块要一两分钟，
# 会把页面整个阻塞住，浏览器等不到响应就白屏。
# 现在的做法：点按钮 → 启动子进程跑 python -m src.index → 界面每 2 秒轮询一次进度，
# 期间其他页签照常可用，完成后自动刷新数据。

REBUILD_LOG = config.ROOT / "rebuild.log"


def start_rebuild() -> None:
    """在后台子进程里启动索引重建。立即返回，不阻塞界面。"""
    log = open(REBUILD_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.index"],
        cwd=str(config.ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    st.session_state["rebuild_proc"] = proc


@st.fragment(run_every=2)
def rebuild_watcher() -> None:
    """轮询后台重建进度。空闲时（没有重建任务）什么都不显示。"""
    proc = st.session_state.get("rebuild_proc")
    if proc is None:
        return

    if proc.poll() is None:
        # 还在跑：显示实时进度（读子进程的输出尾巴）
        tail = ""
        try:
            lines = [l for l in REBUILD_LOG.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            tail = lines[-1][:120] if lines else ""
        except OSError:
            pass
        st.status(f"⏳ 正在后台重建索引…… {tail}", state="running", expanded=False)
        return

    # 跑完了：清缓存、提示结果
    st.session_state["rebuild_proc"] = None
    if proc.returncode == 0:
        retrieve.clear_caches()
        index_mod.get_collection.cache_clear()
        st.cache_data.clear()
        st.session_state["flash"] = "索引重建完成，可以开始提问了"
    else:
        tail = ""
        try:
            tail = REBUILD_LOG.read_text(encoding="utf-8", errors="ignore")[-300:]
        except OSError:
            pass
        st.session_state["flash_err"] = f"索引重建失败：{tail}"

    st.rerun(scope="app")


# ---------------------------------------------------------------- 侧边栏
def sidebar() -> None:
    with st.sidebar:
        st.title("📘 课问")
        st.caption("AIC 2026 · 算法主题赛（AI+开源）")

        ready, msg = index_ready()
        if ready:
            st.success(msg)
        else:
            st.error("索引尚未建立")
            st.code("python -m src.index", language="bash")
            st.caption("建索引前请先把课件放进 data/raw/")

        with st.expander("文档列表", expanded=ready):
            if ready:
                for name in document_list():
                    st.write("· " + name)
            else:
                st.write("（无）")

        st.divider()
        st.caption("当前参数")
        st.caption(f"分块大小：{config.CHUNK_MAX_CHARS} 字　重叠：{config.CHUNK_OVERLAP} 字")
        st.caption(f"召回：向量 {config.TOP_K_VECTOR} + BM25 {config.TOP_K_BM25} → 取前 {config.TOP_K_FINAL}")
        st.caption(f"拒答阈值：{config.MIN_RERANK_SCORE}")
        st.caption(f"大模型：{config.LLM_MODEL}")
        st.caption(f"小到大检索：{'开' if config.USE_PARENT_EXPANSION else '关'}"
                   f"　MMR 去冗余：{'开' if config.USE_MMR else '关'}")

        with st.expander("检索策略开关（演示对比用）", expanded=False):
            st.caption("现场关掉某一层，就能看出它对结果的影响——这正是消融实验做的事")
            use_vector = st.checkbox("向量召回", value=True)
            use_bm25 = st.checkbox("BM25 召回", value=True)
            use_rerank = st.checkbox("交叉编码器重排", value=True)
            use_mmr = st.checkbox("MMR 去冗余", value=config.USE_MMR)
            use_expansion = st.checkbox("查询扩展（多花 1–3 秒）",
                                        value=config.USE_QUERY_EXPANSION)
            use_check = st.checkbox("答案逐句核验（抗幻觉，多花 1–2 秒）",
                                    value=config.USE_SELF_CHECK,
                                    help="把答案拆成一条条结论，逐条回课件原文里找依据；"
                                         "找不到依据的结论会被标记出来")
            use_revision = st.checkbox("无依据结论自动修订（闭环）",
                                       value=config.USE_SELF_REVISION,
                                       help="核验发现无依据结论时，把它们连同课件依据"
                                            "喂回模型只改这些结论，改完复验、变好才采纳")
        st.session_state["retrieval_kwargs"] = {
            "use_vector": use_vector,
            "use_bm25": use_bm25,
            "use_rerank": use_rerank,
            "use_mmr": use_mmr,
            "use_expansion": use_expansion,
        }
        st.session_state["use_check"] = use_check
        st.session_state["use_revision"] = use_revision

        st.divider()
        st.markdown("**历史对话**")
        saved = chats_mod.list_chats()
        if not saved:
            st.caption("点「➕ 新建对话」后，当前对话会自动存到这里")
        for c in saved[:8]:
            col_open, col_del = st.columns([5, 1])
            if col_open.button(f"📄 {c['title'][:20]}", key=f"open_{c['id']}",
                               use_container_width=True):
                st.session_state["messages"] = chats_mod.load_chat(c["id"])
                st.session_state["last_result"] = None
                st.rerun()
            if col_del.button("🗑", key=f"delchat_{c['id']}"):
                chats_mod.delete_chat(c["id"])
                st.rerun()

        st.divider()
        if st.button("重建索引", use_container_width=True):
            start_rebuild()
            st.rerun()


# ---------------------------------------------------------------- 页签 4：课件管理
def tab_files() -> None:
    st.subheader("课件管理")
    st.caption("上传 PPT / Word / PDF 课件并建立索引，就能对它提问。"
               "课件只保存在本机 data/raw/ 目录，**不会上传到任何服务器**。")

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    # ---------- 上传 ----------
    uploads = st.file_uploader(
        "选择或拖入课件文件（可多选，支持 PDF / PPT / Word）",
        type=["pdf", "pptx", "docx"],
        accept_multiple_files=True,
    )
    st.caption("单个文件不超过 200MB；同名文件会覆盖课件库里的旧版本。")
    if uploads:
        # 显著提示"有文件在等待入库"——旧版两步式流程里，
        # 用户选完文件没点保存就去提问，结果全是拒答，很难想到原因
        st.info("📥 已选择 " + "、".join(Path(u.name).name for u in uploads)
                + "，点击下面按钮保存并建立索引")
        if st.button("💾 保存并重建索引", type="primary"):
            saved, skipped = [], []
            for up in uploads:
                # 只取文件名部分，防意外路径
                name = Path(up.name).name
                if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                    skipped.append(name)
                    continue
                (config.DATA_RAW / name).write_bytes(up.getvalue())
                saved.append(name)

            if skipped:
                st.warning("以下文件格式不支持，已跳过：" + "、".join(skipped))
            if saved:
                start_rebuild()          # 后台重建，界面不卡
                st.rerun()
            else:
                st.error("没有文件被保存，请检查文件格式。")

    st.divider()

    # ---------- 已有课件列表 ----------
    st.markdown("**课件库里的文件**")
    files = sorted(p for p in config.DATA_RAW.iterdir()
                   if p.is_file()) if config.DATA_RAW.exists() else []
    if not files:
        st.info("还没有课件。上传一个，或把文件直接放进 data/raw/ 目录。")
    else:
        for f in files:
            col_name, col_size, col_del = st.columns([6, 2, 1])
            col_name.write("· " + f.name)
            col_size.caption(f"{f.stat().st_size / 1024:,.0f} KB")
            if col_del.button("删除", key=f"del_{f.name}"):
                st.session_state["pending_delete"] = f.name
                st.rerun()

    # 删除确认：先标记，再二次确认。误触一次不会真删——课件删错了要重新上传很麻烦。
    pending = st.session_state.get("pending_delete")
    if pending:
        target = config.DATA_RAW / pending
        if target.exists():
            st.warning(f"⚠️ 确定要删除「{pending}」吗？删除后需要重建索引才会真正生效。")
            c_yes, c_no, _ = st.columns([1.2, 1, 4])
            if c_yes.button("确认删除", key="confirm_del", type="primary"):
                target.unlink()
                st.session_state.pop("pending_delete", None)
                st.rerun()
            if c_no.button("取消", key="cancel_del"):
                st.session_state.pop("pending_delete", None)
                st.rerun()
        else:
            st.session_state.pop("pending_delete", None)
    else:
        if files:
            st.caption("删除前会要求二次确认；删除后需要重建索引才会生效。")

    st.divider()

    # ---------- 重建 ----------
    st.markdown("**建立 / 更新索引**")
    st.caption("换了课件、删除了文件之后，都要点一次重建，界面的回答才会跟着变。"
               "重建在后台进行，期间可以继续使用其他页签。")
    if st.button("🔁 重建索引", type="primary", use_container_width=True):
        start_rebuild()
        st.rerun()


# ---------------------------------------------------------------- 页签 1：问答
def render_answer(result: dict) -> None:
    st.markdown(result["answer"])

    if result["refused"]:
        st.warning(f"已拒答（最高相关性 {result['top_score']:.3f}，"
                   f"低于阈值 {config.MIN_RERANK_SCORE}）")
    else:
        st.caption(f"最高相关性 {result['top_score']:.3f}")

    # --- 逐句核验结果（★ 核心创新：核验-修订闭环）---
    summary = result.get("check_summary") or {}
    checks = result.get("checks") or []
    revision = result.get("revision")
    if summary.get("total"):
        if revision:
            b, a = revision["before"], revision["after"]
            if a["unsupported"] == 0:
                st.success(f"✅ 逐句核验：初稿 {b['total']} 条结论中有 "
                           f"{b['unsupported']} 条课件未直接支持，**已自动修订**，"
                           f"复验后全部在课件中找到依据。")
            else:
                st.info(f"🔧 逐句核验：初稿 {b['unsupported']} 条结论无依据，"
                        f"已自动修订为 {a['unsupported']} 条；"
                        f"仍有无依据的结论已在下文标出，请谨慎采信。")
        elif summary["unsupported"]:
            st.warning(f"⚠️ 逐句核验：{summary['total']} 条结论中有 "
                       f"**{summary['unsupported']} 条课件未直接支持**，"
                       f"已在下文标出，请谨慎采信。")
        else:
            st.success(f"✅ 逐句核验：{summary['total']} 条结论全部在课件中找到依据。")
        with st.expander("查看逐句核验明细（每条结论对应课件哪句话）"):
            if revision:
                st.caption("以下是修订后答案的核验结果")
            for i, c in enumerate(checks, start=1):
                mark = "✅" if c["supported"] else "⚠️"
                st.markdown(f"{mark} **{i}. {c['claim']}**")
                if c["evidence"]:
                    for e in c["evidence"]:
                        st.caption(f"　依据（相关性 {e['score']:.3f}）："
                                   f"{e['source']} 第{e['page']}页 —— “{e['text'][:120]}”")
                else:
                    st.caption("　未在课件中找到支持这句话的原文")

    if result["citations"]:
        st.markdown("**依据出处**（点开任意一条，可定位到课件该处的原文）")
        for c in result["citations"]:
            idx = c["index"]
            label = f"[{idx}] {c['source']}　第 {c['page']} 页"
            if c.get("heading"):
                label += f"　·　{c['heading']}"
            hit = (result["hits"][idx - 1]
                   if 0 < idx <= len(result["hits"]) else None)
            with st.expander(label):
                if hit:
                    st.caption(f"课件中该位置附近的原文（相关性 {hit['score']:.3f}）")
                    st.markdown(hit["text"][:800])
                else:
                    st.caption("（历史对话，原文片段未保留）")

    with st.expander("查看参考答案的原文片段", expanded=False):
        for hit in result["hits"]:
            loc = f"{hit['source']} 第{hit['page']}页"
            st.markdown(f"**#{hit['rank']}　相关性 {hit['score']:.3f}　{loc}**")
            st.text(hit["text"][:800])


def tab_chat() -> None:
    head = st.columns([4, 1])
    with head[0]:
        st.subheader("向课程提问")
        st.caption("答案全部来自你的课件，并标注出处页码；课件里没有的内容会直接说不知道。")
    with head[1]:
        if st.button("➕ 新建对话", use_container_width=True):
            # 有内容就先归档到历史，再开新对话
            if chats_mod.archive_chat(st.session_state["messages"]):
                pass
            st.session_state["messages"] = []
            st.session_state["last_result"] = None
            st.rerun()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["result"])
            else:
                st.markdown(msg["content"])

    question = st.chat_input("例如：反向传播里链式法则起什么作用？")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索课件并生成答案……"):
            try:
                result = answer_mod.answer_question(
                    question,
                    retrieval_kwargs=st.session_state.get("retrieval_kwargs"),
                    self_check=st.session_state.get("use_check", config.USE_SELF_CHECK),
                    self_revision=st.session_state.get("use_revision", config.USE_SELF_REVISION),
                )
            except answer_mod.LLMUnavailable as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                # 重建索引期间向量库可能短暂不可用（子进程正在重写数据）
                st.warning("检索服务暂时不可用——索引可能正在后台重建，几秒后再试。")
                st.caption(str(exc)[:300])
                st.stop()
        render_answer(result)

    st.session_state["messages"].append({"role": "assistant", "result": result})
    # 记录到学习画像，供「学习分析」页签使用
    profile_mod.record(question, result["hits"], refused=result["refused"],
                       check_summary=result.get("check_summary"))
    st.session_state["last_result"] = result


# ---------------------------------------------------------------- 页签 2：学习分析
def tab_profile() -> None:
    st.subheader("我的薄弱知识点")
    st.caption("你反复问的，就是你没掌握的。提问越多，这份分析越准。")

    s = profile_mod.summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("累计提问", s["total"])
    col2.metric("课件外问题", s["refused"])
    col3.metric("识别出的薄弱点", len(s["weak"]))

    # 抗幻觉指标：这是本作品自检机制的运行结果，也是报告里的可验证数据
    if s.get("claims"):
        st.divider()
        st.markdown("**答案可信度核验**")
        st.caption("每条回答都会被拆成若干结论，逐条回课件原文里核对依据。")
        c1, c2, c3 = st.columns(3)
        c1.metric("累计核验结论", s["claims"])
        c2.metric("课件未直接支持", s["unsupported"])
        c3.metric("无依据率", f"{s['unsupported_rate']:.1%}")

    st.divider()

    if not s["weak"]:
        st.info("还没有足够的提问记录。先去「向课程提问」页签问几个问题吧。")
    else:
        # ⚠️ 表格必须拼成完整字符串一次性输出：逐行 st.markdown 会把每行
        # 渲染成独立段落，表格语法不生效，页面上就是一竖排竖线文本
        rows = ["| 知识点 | 类型 | 提问次数 | 最近一次 |",
                "|---|---|---|---|"]
        for item in s["weak"]:
            kind = "章节" if item["type"] == "heading" else "关键词"
            rows.append(f"| {item['label']} | {kind} | {item['count']} | "
                        f"{item['last_time'][:10]} |")
        st.markdown("\n".join(rows))

    st.divider()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.download_button(
            "下载复习清单（Markdown）",
            data=profile_mod.export_markdown(),
            file_name="薄弱知识点复习清单.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_b:
        if st.button("清空记录", use_container_width=True):
            profile_mod.reset()
            st.rerun()

    with st.expander("复习清单预览"):
        st.markdown(profile_mod.export_markdown())


# ---------------------------------------------------------------- 页签 3：检索调试
def tab_debug() -> None:
    st.subheader("检索过程全透明")
    st.caption("同一个问题，各层各自找到什么、融合和重排后排名怎么变——这就是本作品的自研核心所在。")

    query = st.text_input("输入一个问题", value="反向传播中链式法则的作用")
    if not query:
        return

    kwargs = st.session_state.get("retrieval_kwargs", {})

    # --- 查询扩展：让人看到"问题被改写成了什么" ---
    if kwargs.get("use_expansion"):
        with st.spinner("正在用大模型改写问题……"):
            variants = retrieve.expand_query(query)
        if variants:
            st.markdown("**查询扩展：大模型把问题改写成了这些检索式**")
            for v in variants:
                st.caption("· " + v)
            st.caption("学生用口语问、课件写的是术语，扩展检索式就是为了补上这个词面差距。")
        else:
            st.caption("查询扩展未生效（模型调用失败，已回退到原始问题）")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**向量召回 Top 5**")
        try:
            for rank, (cid, score) in enumerate(retrieve.vector_search(query, k=5), start=1):
                st.caption(f"{rank}. 相似度 {score:.3f}　{cid}")
        except Exception as exc:
            st.warning(f"向量召回失败：{exc}")
    with col2:
        st.markdown("**BM25 召回 Top 5**")
        try:
            for rank, (cid, score) in enumerate(retrieve.bm25_search(query, k=5), start=1):
                st.caption(f"{rank}. BM25 {score:.2f}　{cid}")
        except Exception as exc:
            st.warning(f"BM25 召回失败：{exc}")

    st.divider()
    st.markdown("**融合 + 重排 + 去冗余后的最终结果**")
    st.caption("RRF 只看排名不看分数量纲，所以两路分数差多少都不会互相压过。")
    try:
        hits = retrieve.hybrid_search(query, **kwargs)
    except Exception as exc:
        st.error(f"检索失败：{exc}")
        return

    for hit in hits:
        loc = f"{hit['source']} 第{hit['page']}页"
        header = f"#{hit['rank']}　重排分数 {hit['score']:.3f}　{loc}"
        if hit.get("heading"):
            header += f"　·　{hit['heading']}"
        with st.expander(header):
            st.caption(f"命中的片段（{len(hit['text'])} 字）——检索用这一小块，保证精准")
            st.text(hit["text"][:1200])
            extra = len(hit.get("context", "")) - len(hit["text"])
            if extra > 0:
                st.caption(f"↑ 送给大模型时会扩成整章（{len(hit['context'])} 字，+{extra} 字上下文），"
                           f"保证结论和它的前提条件待在一起")


# ---------------------------------------------------------------- 页签 5：项目速览
ABOUT_MD = """
### 这个作品是什么

一个跑在自己电脑上的**课程 AI 助教**：把课件喂进去，学生提问，
它回答并**标注出处页码**；课件里没有的它**不瞎编**；还会根据提问记录生成**薄弱点复习清单**。

参加 **AIC 2026 算法主题赛（AI+开源）**，截止 **10 月 15 日 20:00**。

### 已经能跑的（不是计划，是现在）

- ✅ 完整作品：上传课件 → 建索引 → 提问回答，整条链路已打通
- ✅ 全中文网页界面：提问 / 学习分析 / 检索调试 / 课件管理
- ✅ 界面里直接传 PPT / Word / PDF，点"重建索引"即可
- ✅ 拒答能力：课件里没有的问题会明说，不编造
- ✅ 评测脚本：一条命令产出报告需要的对比数据
- ✅ 运行环境全部装好调通，**断网也能演示**

### 我们还要干的活

| # | 任务 | 为什么重要 |
|---|---|---|
| 1 | **换真实课件**（一门课，100–300 页）「课件管理」页签可传 | 现在是演示数据，必须换成真的 |
| 2 | **出评测题**：100 道可回答 + 20 道课件里没有的，标注答案页码 | "效果验证"占 25 分，全靠它 |
| 3 | 跑实验（eval 脚本两条命令） | 产出报告核心数据表 |
| 4 | 找 20–30 个同学试用 + 问卷 | "用户反馈"是评分明确要的证据 |
| 5 | 给开源项目提 1–2 个 PR | "开放成果"占 15 分 |
| 6 | 缴费、录视频、写报告 | 见文件夹里的详细文档 |

### 关键时间

**10/1–10/2 出评测题 → 10/4 实验数据 → 10/7 问卷收完 → 10/10 PR 提交 →
10/11 功能冻结 → 10/14 材料完成 → 10/15 上午提交（别卡死线）**

### 三条纪律（每个人都要记住）

1. **课件不上传任何网盘/代码托管平台**——受版权保护，只在本机用
2. **所有材料不出现学校名称、校徽、指导教师姓名**——规则原文，违者扣分甚至取消资格
3. **10/11 之后不改代码**——改了要重跑实验、重录视频，来不及

### 想看细节？

文件夹里有完整文档：`开发指导.md`（每天干什么）、`环境说明与交接.md`（环境与命令）、
`报告提纲与素材索引.md`（报告怎么写）、`队友须知.md`（本页的完整版）、
`AIC2026_双赛道行动方案.md`（报名与规则）。
"""


def tab_about() -> None:
    st.markdown(ABOUT_MD)


# ---------------------------------------------------------------- 主入口
def main() -> None:
    init_state()
    show_flash()
    rebuild_watcher()      # 后台重建任务的进度轮询（空闲时什么都不显示）

    ready, _ = index_ready()

    # 页面一打开就预热模型，避免评委看到的第一个问题等太久
    if ready:
        with st.spinner("正在加载模型（首次打开页面需要十几秒，之后会常驻内存）……"):
            models_ok, models_msg = load_models()
        if not models_ok:
            st.error(f"模型加载失败：{models_msg}")
            st.stop()

    sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["💬 向课程提问", "📊 学习分析", "🔍 检索调试", "📁 课件管理", "📖 项目速览"]
    )

    with tab1:
        if not ready:
            st.warning("还没有索引。请到「📁 课件管理」页签上传课件并点「重建索引」。")
        else:
            tab_chat()
    with tab2:
        tab_profile()
    with tab3:
        if not ready:
            st.warning("还没有索引，无法做检索调试。")
        else:
            tab_debug()
    with tab4:
        tab_files()
    with tab5:
        tab_about()


if __name__ == "__main__":
    main()
