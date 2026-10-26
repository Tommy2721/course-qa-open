"""课问 · 课程知识问答与复习助手

模块划分：
    parse     文档解析（PDF/PPTX/DOCX → 带页码文本）
    chunk     分块（自研点 1：章节感知分块）
    index     建索引（向量库 + BM25）
    retrieve  混合检索（自研点 2：RRF 融合 + 重排）
    answer    生成答案（自研点 3：引用溯源 + 拒答）
    profile   学习分析（自研点 4：薄弱点识别）
    eval      评测（Recall@K / 拒答阈值校准）
"""
