# 🧠 左脑 — 本地记忆推理引擎

> 你说人话，它自动搞定。一个为 AI Agent 设计的本地记忆 + 推理 + 关联引擎。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Lines](https://img.shields.io/badge/代码-6900行-orange.svg)]()

---

## 这是什么？

**左脑（Left Brain）** 是一个纯 Python 的本地知识引擎，可以嵌入任何 AI Agent，让它拥有**持久化记忆、语义推理、知识关联**能力。

它最初是为 WorkBuddy AI 助手设计的 Skill 插件（v3.12），现已开源为独立模块。你可以把它接入 Claude、ChatGPT、或你自己写的 AI 应用。

### 一句话

> 你告诉它一件事，它记住；你问它相关问题，它自动联想；你聊到相关话题，它主动把背景知识注入对话。

---

## 核心能力

| 模块 | 功能 | 一句话 |
|------|------|--------|
| 🧠 **MemoryEngine** | 知识图谱引擎 | 增删改查、图扩散搜索、智能纠错、多义词消歧 |
| 🔍 **IntentClassifier** | 意图分类 | 自动识别8种意图（学习/查询/分析/总结等） |
| 🏷️ **GenreClassifier** | 体裁分类 | 8种体裁 + 领域标签 + 骨架提取 |
| 📊 **DataAnalyzer** | 数据分析 | 自动提取数字、趋势、对比、统计 |
| 📝 **Summarizer** | 文章总结 | 提炼核心结论 + 关键论据 + 数据支撑 |
| 🕸️ **EntanglementBridge** | 纠缠场关联 | 词间量子关联分析，发现隐藏关系 |
| 💡 **DeepReason** | 深度推理 | 因果链提取、知识推论 |
| 📋 **ContextStack** | 上下文记忆 | 三层上下文（短程/中程/长程）+ SimHash语义匹配 |
| 🔗 **RecommendRanker** | 智能推荐 | 基于知识图谱 + 纠缠场 + 用户画像 |
| 👤 **UserProfile** | 用户画像 | 自动学习用户偏好和习惯 |

---

## 快速开始

### 环境要求

- Python 3.8+
- 可选：`pycryptodome`（用于备份加密功能）

### 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/zuonao.git
cd zuonao
```

零依赖即可使用核心功能。如果需要备份加密：

```bash
pip install pycryptodome
```

### 5 行代码上手

```python
import sys
sys.path.insert(0, "你的目录/zuonao")
from engine import get_engine

# 获取引擎实例（单例，自动初始化）
engine = get_engine()

# 学习一条知识
engine.add_node("公司年会定在12月25号，地点国际会议中心3楼", category="日程")

# 查询知识
idx = engine.find_node("年会")
if idx is not None:
    print(engine.nodes[idx]["text"])

# 图扩散搜索——从"年会"出发，找2跳内的所有关联
results = engine.graph_search("年会", max_hops=2)
for r in results:
    print(f"[{r['hop']}跳] {r['text']}")
```

### 高级用例

```python
from engine import (
    get_engine,
    IntentClassifier,    # 意图分类
    DataAnalyzer,        # 数据分析
    Summarizer,          # 文章总结
    ContextStack,        # 上下文管理
)

engine = get_engine()

# 1. 学习多条知识并自动建立关联
engine.add_node("Python是最流行的AI编程语言", category="技术")
engine.add_node("TensorFlow是Google开发的深度学习框架", category="技术")
engine.add_node("PyTorch是Meta开发的深度学习框架", category="技术")
# 引擎会自动检测并建立关联！

# 2. 意图分类
intent = IntentClassifier.classify("帮我分析一下这份销售数据")
print(intent)  # → "analyze"

# 3. 数据分析
data_text = "第一季度营收100万，第二季度150万，增长50%"
result = DataAnalyzer.analyze(data_text)
print(result["numbers"])     # → [100, 150, 50]
print(result["trend"])       # → "上升"

# 4. 文章总结
article = "今天上午召开了产品发布会...(长文本)"
summary = Summarizer.summarize(article)
print(summary["core_conclusion"])  # → 核心结论
```

---

## 项目结构

```
zuonao/
├── engine.py          核心引擎（5252行）
│   ├── MemoryEngine       知识图谱引擎
│   ├── IntentClassifier   意图分类器
│   ├── GenreClassifier    体裁分类器
│   ├── DataAnalyzer       数据分析器
│   ├── Summarizer         文章总结器
│   ├── DeepReason         深度推理引擎
│   ├── EntanglementBridge  纠缠场引擎
│   ├── ContextStack       三层上下文记忆
│   ├── ContextMemoryLayer 上下文记忆层
│   ├── RecommendRanker    智能推荐
│   ├── UserProfile        用户画像
│   ├── TFIDFSearch        语义搜索
│   ├── SimpleChineseTokenizer  轻量中文分词
│   └── seed_knowledge()   种子知识预置
│
├── token_db.py        Token 监测与统计（1620行）
│   ├── TokenPricer        20+模型精准计价
│   ├── PreciseTokenizer   精准Token估算
│   ├── SQLite 数据层      计数器/快照/热力图
│   └── 成就系统           18个成就
│
├── _keys.py           密钥模块（本地部署版）
├── crypto_config.py   密钥桥接
├── data/              运行时数据目录（自动创建）
│   ├── memory_duck_data.json   知识图谱数据
│   ├── left_brain.db           统计数据
│   └── entanglement_data.json  纠缠场数据
│
└── README.md          本文件
```

---

## API 参考

### MemoryEngine — 知识图谱引擎

```python
engine = get_engine()

# 添加知识（自动分类 + 体裁检测 + 骨架提取）
idx = engine.add_node(
    text="需要记忆的内容",
    category="技术",        # 可选，不填则自动分类
    genre="技术文档",       # 可选，不填则自动检测
    domain="AI",            # 可选
)

# 精确查询
idx = engine.find_node("关键词")

# 模糊查询（TF-IDF语义匹配）
results = engine.fuzzy_find("大概的关键词", top_k=3)

# 图扩散搜索——发现隐藏关联
results = engine.graph_search("起点", max_hops=2)

# 修改知识
engine.update_node(idx, "新的内容")

# 删除知识
engine.delete_node(idx)

# 智能纠错
corrected = engine.fix_typo("派森")  # → "Python"

# 多义词消歧
meaning = engine.disambiguate("苹果")  # → 根据上下文判断是手机还是水果

# 自动分类
category = engine.auto_category("机器学习框架PyTorch")  # → "技术"
```

### IntentClassifier — 意图分类

```python
from engine import IntentClassifier

# 分类结果: learn / query / analyze / summarize / correct / recommend / search / trace
intent = IntentClassifier.classify("帮我记住明天下午3点开会")
print(intent)  # → "learn"

intent = IntentClassifier.classify("年会是什么时候")
print(intent)  # → "query"

intent = IntentClassifier.classify("分析一下这份销售数据")
print(intent)  # → "analyze"
```

### DataAnalyzer — 数据分析

```python
from engine import DataAnalyzer

result = DataAnalyzer.analyze("""
    1月销售额100万，2月120万，3月90万。
    竞争对手同期分别为80万、100万、110万。
""")

print(result["numbers"])       # → [100, 120, 90, 80, 100, 110]
print(result["trend"])         # → "波动"
print(result["comparison"])    # → 包含对比信息
print(result["stats"])         # → 统计指标（均值、最大、最小等）
```

### Summarizer — 文章总结

```python
from engine import Summarizer

result = Summarizer.summarize("长篇文本...")

print(result["core_conclusion"])  # 核心结论
print(result["key_arguments"])    # 关键论据
print(result["data_points"])      # 数据支撑
```

### ContextStack — 上下文记忆

```python
from engine import ContextStack

ctx = ContextStack()

# 添加对话轮次
ctx.add_turn("用户问了关于Python的问题", metadata={"topic": "编程"})
ctx.add_turn("AI回答了Python的基础语法", metadata={"topic": "编程"})

# 语义检索相关上下文
relevant = ctx.search("Python的面向对象特性")

# 获取摘要
summary = ctx.get_summary()
```

### EntanglementBridge — 纠缠场

```python
from engine import get_entanglement

ent = get_entanglement()

# 分析两个概念的纠缠关系
relations = ent.entangle("机器学习", "深度学习")
print(relations)  # → [("子领域", 0.9), ("包含", 0.85)]

# 查找与某概念最纠缠的其他概念
neighbors = ent.get_top_related("Python", top_k=5)
```

### DeepReason — 深度推理

```python
from engine import DeepReason

# 提取因果链
causal = DeepReason.extract_causal_chain("因为下雨，所以活动延期到下周")
print(causal)  # → [{"cause": "下雨", "effect": "活动延期"}]

# 生成推理指令
prompt = DeepReason.generate_prompt("分析产品A和产品B的优劣")
```

### Token 统计（token_db.py）

```python
from token_db import TokenPricer, PreciseTokenizer, get_realtime_usage

# 精准Token估算
tokens = PreciseTokenizer.estimate("这是一段中文文本", model="gpt-4o")
print(f"估算token数: {tokens}")

# 费用计算
cost = TokenPricer.calc_cost("gpt-4o", input_tokens=1000, output_tokens=500)
print(f"费用: ¥{cost}")

# 查看使用统计
usage = get_realtime_usage()
print(usage["today"]["consumed"])   # 今日消耗
print(usage["total"]["saved_cost"]) # 累计节省费用
```

---

## 数据存储

所有数据存储在 `data/` 目录下：

| 文件 | 格式 | 内容 |
|------|------|------|
| `memory_duck_data.json` | JSON | 知识图谱（节点 + 边 + 上下文 + 工作区） |
| `left_brain.db` | SQLite (WAL) | Token统计、成就、操作日志 |
| `entanglement_data.json` | JSON | 纠缠场数据 |

数据完全本地存储，不需要网络，不需要数据库服务。

---

## 工作区隔离

引擎支持多工作区（Workspace），不同项目的知识互不干扰：

```python
engine = get_engine()

# 自动检测当前工作区（从 os.getcwd() 提取）
# 或手动设置
engine._set_workspace("/path/to/project")

# 当前工作区的知识
nodes = engine.nodes

# 查看所有工作区分布
print(engine._workspace_distribution())
# → {"global": 5, "project_A": 10, "project_B": 3}

# 跨工作区搜索（global 工作区在所有区可见）
results = engine.global_search("关键词")
```

---

## 接入你的 AI Agent

### 接入 Claude / ChatGPT

最简单的方式——把引擎注入到 System Prompt：

```python
from engine import get_engine, ContextStack

engine = get_engine()
ctx = ContextStack()

def build_system_prompt(user_input: str) -> str:
    """每次对话前调用，注入相关知识到 System Prompt"""
    # 搜索相关知识
    relevant = engine.fuzzy_find(user_input, top_k=5)
    context_summary = ctx.get_summary()

    prompt = "你是左脑助手。以下是你的记忆：\n\n"
    prompt += "### 最近上下文\n" + context_summary + "\n\n"
    prompt += "### 相关知识\n"
    for r in relevant:
        prompt += f"- {r['text']}\n"
    prompt += "\n请基于以上记忆回答用户问题。"
    return prompt
```

### 接入 WorkBuddy

引擎已内置 `workbuddy_main()` 统一入口，支持完整的命令分发：

```python
from engine import workbuddy_main

# 自动意图识别 + 分发
result = workbuddy_main("perceive", "帮我记住：服务器IP是192.168.1.1")
print(result)  # → {"status": "ok", "action": "学习", ...}

# 支持的命令
workbuddy_main("learn", "要学习的内容")
workbuddy_main("query", "查询关键词")
workbuddy_main("analyze", "要分析的数据")
workbuddy_main("summarize", "要总结的文章")
workbuddy_main("search", "搜索词")
workbuddy_main("session", "")          # 会话初始化
workbuddy_main("stats", "")            # 引擎统计
workbuddy_main("dashboard", "")        # 仪表盘数据
```

### 接入你自己的 AI 应用

```python
from engine import MemoryEngine, IntentClassifier, DataAnalyzer, Summarizer

class YourAIAgent:
    def __init__(self):
        self.memory = MemoryEngine()
    
    def handle_message(self, user_input: str) -> str:
        # 1. 意图识别
        intent = IntentClassifier.classify(user_input)
        
        # 2. 根据意图处理
        if intent == "learn":
            self.memory.add_node(user_input)
            return f"已记住：{user_input[:50]}..."
        
        elif intent == "query":
            idx = self.memory.find_node(user_input)
            if idx:
                return self.memory.nodes[idx]["text"]
            return "我还没学过这个。"
        
        elif intent == "analyze":
            return DataAnalyzer.analyze(user_input)
        
        # ... 更多意图处理
```

---

## 常见问题

**Q: 需要联网吗？**
A: 不需要。所有功能完全本地运行。

**Q: 数据存在哪里？**
A: `data/` 目录下，JSON + SQLite 双格式。你可以直接复制整个目录迁移。

**Q: 支持多语言吗？**
A: 核心针对中文优化（含轻量中文分词器），也支持英文和其他语言。

**Q: 能处理多大数据量？**
A: SQLite 数据库支持百万级记录。知识图谱在内存中运行，建议控制在 5 万节点以内。

**Q: 和 LangChain / LlamaIndex 有什么区别？**
A: LangChain 是 LLM 编排框架，左脑是本地记忆引擎——没有 embedding、没有向量数据库、零 token 消耗。更轻量，更适合做 AI Agent 的持久化记忆层。

**Q: 可以用于商业项目吗？**
A: 可以。MIT 开源协议。

---

## 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v3.12 | 2026-06 | 全局架构注入、自检系统升级至10项 |
| v3.9 | 2026-05 | 统一分类管线、SimHash语义匹配、体裁感知检索 |
| v3.8 | 2026-04 | KAR融合模块、因果链可视化、三层上下文记忆 |
| v3.6 | 2026-03 | Workspace隔离、自动更新、完整时间戳体系 |
| v3.0 | 2026-02 | RSA+AES嵌套加密、预刷新逻辑、降级策略 |

---

## 许可证

MIT License

---

## 作者

左脑项目始于 WorkBuddy AI 助手的本地记忆插件，现开源为独立引擎模块。

有问题或建议？欢迎提 Issue 或 PR！
