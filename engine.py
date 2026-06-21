# -*- coding: utf-8 -*-
"""
左脑引擎 v3.9 — 记忆 + 推理 + 关联 + 感知增强 + KAR融合 五位一体
=====================================
组成部分:
  1. 🧠 语义记忆引擎（记忆鸭核心）— 知识存储/图扩散检索/纠错/消歧
  2. 🔍 深度推理引擎（输出熊猫核心）— 数据分析/文章总结/骨架匹配/因果推论
  3. 🕸️ 纠缠场引擎 — 词间量子关联分析
  4. 📋 上下文记忆引擎 — 三层上下文（短程/中程/长程）+ SimHash语义匹配
  5. ⚡ 感知增强系统 — 自动感知/记忆注入/会话感知/智能推荐
  6. 🎯 统一分类管线 — IntentClassifier+GenreClassifier合并/体裁感知检索/体裁感知衰减

v3.9 升级（统一管线+深度融合）:
  - P0-1: IntentClassifier与GenreClassifier合并为统一管线，classify_v2()一次调用返回所有维度
  - P0-2: 三层上下文检索升级：SQL精确匹配→SimHash语义匹配+TF-IDF关键词重叠兜底
  - P1-3: 骨架反馈闭环：find_node()新增genre_aware参数，体裁骨架反哺检索策略
  - P1-4: 体裁感知衰减：dialogue 3%/月→definition 0.5%/月，差异化衰减
  - P1-5: 追溯增强：因果链可视化(前端渲染A→B→C)+时间线重建
  - P2-6: GenreDetector置信度阈值(<0.3回退)+体裁冲突检测(前两名分差<0.15降权)

授权模式：加密狗1 — 一机一密 · XOR流加密 · HMAC-SHA256签名 · 硬件指纹绑定
"""

import hashlib, json, os, re, hmac, subprocess, sys, math, random, time, sqlite3, uuid
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ===== SQLite 数据层（增量存储 + 日级快照 + 时段热力图）=====
try:
    from token_db import (
        init_db, counter_incr, counter_get, counter_get_all,
        today_incr, today_get_all, today_reset,
        save_daily_snapshot, get_daily_snapshots,
        hourly_incr, get_hourly_data, get_hourly_range,
        scene_incr as db_scene_incr, scene_get_all,
        feature_incr, feature_get_all,
        log_operation, save_window_pos, load_window_pos,
        check_achievements, get_achievements,
        export_csv, export_csv_to_file, migrate_from_json,
        MonitorBridge, SimpleTokenizer, PreciseTokenizer,
        TokenPricer, record_platform_usage, get_realtime_usage,
        get_dashboard_data, ensure_db,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


# ====================== v3.4: 异常日志 ======================

import logging as _logging
import traceback as _traceback

_log_dir = os.path.join(str(SKILL_DIR), "data") if 'SKILL_DIR' in dir() else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, "error.log")

_logger = _logging.getLogger("zuonao_engine")
_logger.setLevel(_logging.WARNING)
_handler = _logging.FileHandler(_log_path, encoding="utf-8", delay=True)
_handler.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger.addHandler(_handler)

def log_error(module: str, msg: str, exc: Exception = None):
    """记录异常日志"""
    detail = f"{module}: {msg}"
    if exc:
        detail += f"\n{_traceback.format_exc()}"
    _logger.error(detail)

def log_warn(module: str, msg: str):
    _logger.warning(f"{module}: {msg}")

def log_info(module: str, msg: str):
    _logger.info(f"{module}: {msg}")


# ====================== 本地部署版 — 无 DRM 授权 ======================
from crypto_config import SECRET_KEY

def _get_machine_fingerprint():
    """获取本机硬件指纹（用于数据标识，非授权绑定）"""
    try:
        import uuid as _uuid
        return str(_uuid.getnode())
    except Exception:
        return "local-deployment"

def _verify_license():
    """本地部署版：始终通过验证"""
    return True
_SKILL_DIR = Path(__file__).resolve().parent
# 兼容两种目录结构：纯净母包(SKill_DIR/data/) 和 skills安装(SKill_DIR=scripts/, data在上级)
if (_SKILL_DIR / "data").exists():
    DATA_DIR = _SKILL_DIR / "data"
elif (_SKILL_DIR.parent / "data").exists():
    DATA_DIR = _SKILL_DIR.parent / "data"
else:
    DATA_DIR = _SKILL_DIR / "data"
DATA_FILE = DATA_DIR / "memory_duck_data.json"
ENTANGLE_FILE = DATA_DIR / "entanglement_data.json"
DB_PATH = DATA_DIR / "left_brain.db"
os.makedirs(str(DATA_DIR), exist_ok=True)


# ===================================================================
# 第零部分：三大增强功能的基础类
# ===================================================================


# ====================== v3.4: 轻量中文分词器（零依赖） ======================

class SimpleChineseTokenizer:
    """轻量中文分词器 — 基于字符bigram + 关键词提取
    
    不依赖 jieba 等外部库，纯 Python 实现。
    分词策略：
    1. 中文：2-gram 字符对 + 单字（覆盖短词）
    2. 英文/数字：正则提取
    3. 停用词过滤
    """
    
    STOP_WORDS = set("的了是在我他有这不也人们来到时为之就能对以和与及或其但而".split())
    
    @classmethod
    def tokenize(cls, text: str) -> list:
        """将文本分词，返回 token 列表"""
        tokens = []
        
        # 提取英文单词和数字
        for m in re.finditer(r'[a-zA-Z]{2,}|\d+', text):
            tokens.append(m.group().lower())
        
        # 中文 bigram 分词
        chinese_chars = re.findall(r'[一-鿿]', text)
        # 2-gram
        for i in range(len(chinese_chars) - 1):
            bigram = chinese_chars[i] + chinese_chars[i+1]
            if bigram not in cls.STOP_WORDS:
                tokens.append(bigram)
        # 3-gram (for longer phrases)
        for i in range(len(chinese_chars) - 2):
            trigram = chinese_chars[i] + chinese_chars[i+1] + chinese_chars[i+2]
            tokens.append(trigram)
        # single chars (for short queries)
        for c in chinese_chars:
            if c not in cls.STOP_WORDS:
                tokens.append(c)
        
        return tokens


class TFIDFSearch:
    """TF-IDF 向量空间模型 — 轻量语义搜索引擎
    
    为知识图谱中的所有节点构建 TF-IDF 向量，
    查询时计算余弦相似度，返回语义相似的节点。
    """
    
    def __init__(self):
        self._doc_vectors = {}   # node_idx -> {token: tfidf_weight}
        self._idf = {}           # token -> idf
        self._dirty = True       # 是否需要重建索引
        self._doc_count = 0
    
    def mark_dirty(self):
        self._dirty = True
    
    def _build_index(self, engine):
        """构建/重建 TF-IDF 索引"""
        if not self._dirty:
            return
        
        N = len(engine.nodes)
        if N == 0:
            self._dirty = False
            return
        
        # 1. 计算文档频率 (DF)
        df = {}
        all_docs = {}
        for idx, node in enumerate(engine.nodes):
            text = node.get("text", "")
            tokens = SimpleChineseTokenizer.tokenize(text)
            unique_tokens = set(tokens)
            all_docs[idx] = tokens
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1
        
        # 2. 计算 IDF
        self._idf = {}
        for t, count in df.items():
            self._idf[t] = max(0.1, math.log((N + 1) / (count + 1)) + 1)
        
        # 3. 计算 TF-IDF 向量
        self._doc_vectors = {}
        for idx, tokens in all_docs.items():
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            # 归一化 TF
            max_tf = max(tf.values()) if tf else 1
            vec = {}
            for t, freq in tf.items():
                vec[t] = (freq / max_tf) * self._idf.get(t, 0.1)
            self._doc_vectors[idx] = vec
        
        self._doc_count = N
        self._dirty = False
    
    def search(self, engine, query: str, top_k: int = 10) -> list:
        """语义搜索 — 返回 [(node_idx, score), ...]，按余弦相似度排序"""
        self._build_index(engine)
        
        query_tokens = SimpleChineseTokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        # 构建查询向量
        query_vec = {}
        for t in query_tokens:
            query_vec[t] = query_vec.get(t, 0) + 1
        max_qf = max(query_vec.values()) if query_vec else 1
        for t in query_vec:
            query_vec[t] = (query_vec[t] / max_qf) * self._idf.get(t, 0.1)
        
        # 计算余弦相似度
        results = []
        query_norm = math.sqrt(sum(v*v for v in query_vec.values())) or 1
        
        for idx, doc_vec in self._doc_vectors.items():
            dot = 0.0
            doc_norm = math.sqrt(sum(v*v for v in doc_vec.values())) or 1
            for t, qv in query_vec.items():
                if t in doc_vec:
                    dot += qv * doc_vec[t]
            if dot > 0:
                score = dot / (query_norm * doc_norm)
                # 权重衰减：最近访问的节点加分
                node = engine.nodes[idx]
                access_count = node.get("access_count", 0)
                last_access = node.get("last_accessed", 0)
                time_boost = 1.0
                if last_access > 0:
                    days_since = (time.time() - last_access) / 86400
                    time_boost = 1.0 + max(0, 2.0 - days_since * 0.5)
                final_score = score * min(2.0, 1.0 + access_count * 0.1) * time_boost
                results.append((idx, final_score))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

# 全局语义搜索引擎实例
_tfidf_searcher = TFIDFSearch()

class IntentClassifier:
    """意图分类器 — 自动感知用户输入的意图类型（零embedding，纯关键词+模式匹配）"""

    # 意图类型常量
    LEARN = "learn"
    QUERY = "query"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    CHAT = "chat"
    COMMAND = "command"

    # 分类规则
    RULES = {
        LEARN: {
            "patterns": [r"记住", r"记下来", r"学会", r"知识", r"这个很重要", r"备注", r"保存", r"存下来"],
            "scene_boost": ["学习", "通用"],
            "length_bonus": 50,   # 超过此长度倾向学习
        },
        QUERY: {
            "patterns": [r"是什么", r"什么是", r"怎么", r"如何", r"为什么", r"？", r"\?", r"哪", r"哪个", r"多少"],
            "length_penalty": 50,  # 低于此长度倾向查询
        },
        ANALYZE: {
            "patterns": [r"分析", r"对比", r"趋势", r"占比", r"统计", r"数据", r"比较", r"差异"],
            "scene_boost": ["分析"],
        },
        SUMMARIZE: {
            "patterns": [r"总结", r"概括", r"提炼", r"要点", r"核心", r"归纳", r"梳理"],
        },
        CHAT: {
            "patterns": [r"你好", r"哈哈", r"谢谢", r"是的", r"早安", r"晚安", r"没事", r"嗯"],
            "default": True,
        },
        COMMAND: {
            "patterns": [r"^/"],
            "priority": 100,
        },
    }

    @classmethod
    def classify(cls, text: str, scene: str = "") -> str:
        """
        分类流程：
        1. 命令检测：以/开头 → COMMAND
        2. 模式匹配：对每个意图计算匹配置信度
        3. 场景加分：auto_category 结果与意图场景一致时加分
        4. 长度修正：文本长度对学习/查询意图的倾向性
        5. 返回最高置信度的意图
        """
        if not text or not text.strip():
            return cls.CHAT

        # 1. 命令检测
        if text.strip().startswith("/"):
            return cls.COMMAND

        # 2. 模式匹配 + 场景加分 + 长度修正
        scores = {}
        text_len = len(text)

        for intent, rule in cls.RULES.items():
            if intent == cls.COMMAND:
                continue  # 已在步骤1处理

            score = 0.0

            # 模式匹配
            for pattern in rule.get("patterns", []):
                matches = re.findall(pattern, text)
                score += len(matches) * 1.0

            # 场景加分
            if scene and scene in rule.get("scene_boost", []):
                score += 0.5

            # 长度修正
            if intent == cls.LEARN and text_len > rule.get("length_bonus", 50):
                score += 0.3 * min(1.0, (text_len - 50) / 100)
            if intent == cls.QUERY and text_len < rule.get("length_penalty", 50):
                score += 0.2

            scores[intent] = score

        # 3. 兜底：无任何匹配 → CHAT
        if not scores or max(scores.values()) == 0:
            return cls.CHAT

        # 4. 返回最高分意图
        best = max(scores, key=scores.get)
        return best


class ContextStack:
    """对话上下文栈 — 维护最近N轮对话的关键信息"""

    MAX_SIZE = 30  # v3.4: 10→30，支持长会话
    DECAY_FACTOR = 0.7
    MIN_WEIGHT = 0.1

    def __init__(self):
        self.stack: List[Dict] = []  # [{"keywords": [], "scene": "", "ts": float, "weight": 1.0}]
        # v3.4: 对话目标跟踪
        self.current_goal = ""       # 当前对话目标
        self.goal_stack: List[str] = []  # 目标栈（支持嵌套目标）
        self.goal_turn_count = 0     # 当前目标持续的轮数

    def push(self, text: str, scene: str = ""):
        """压入一轮对话的上下文"""
        keywords = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        entry = {
            "keywords": keywords[:8],
            "scene": scene,
            "ts": time.time(),
            "weight": 1.0,
        }
        self.stack.append(entry)

        # 衰减旧条目
        for item in self.stack[:-1]:
            item["weight"] *= self.DECAY_FACTOR

        # 淘汰低权重
        self.stack = [item for item in self.stack if item["weight"] >= self.MIN_WEIGHT]

        # 截断
        if len(self.stack) > self.MAX_SIZE:
            self.stack = self.stack[-self.MAX_SIZE:]

    def get_context_keywords(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """获取当前上下文的关键词（带权重）"""
        kw_weight = {}
        for item in self.stack:
            for kw in item["keywords"]:
                kw_weight[kw] = kw_weight.get(kw, 0) + item["weight"]

        sorted_kw = sorted(kw_weight.items(), key=lambda x: -x[1])
        return sorted_kw[:top_k]

    def get_current_scene(self) -> str:
        """获取当前主要场景"""
        if not self.stack:
            return "通用"
        return self.stack[-1].get("scene", "通用")

    def clear(self):
        """清空上下文（新话题时调用）"""
        self.stack = []

    def to_dict(self) -> Dict:
        """序列化"""
        return {"stack": self.stack}

    def from_dict(self, data: Dict):
        """反序列化"""
        self.stack = data.get("stack", [])


class UserProfile:
    """用户画像 — 基于使用数据构建兴趣模型"""

    def __init__(self, engine: 'MemoryEngine'):
        self.engine = engine
        # 兴趣维度（从 scene_counts 继承）
        self.interest_vector: Dict[str, float] = {}
        # 深度偏好（偏记忆型/推理型/均衡型）
        self.depth_type = "balanced"
        # 活跃时段
        self.active_hours: Dict[str, int] = {}
        # 已推荐历史（去重）
        self.recommended_history: List[List] = []  # [[text_hash, ts], ...]
        self.MAX_HISTORY = 50

    def build(self):
        """构建画像"""
        # 1. 领域偏好：从 scene_counts 归一化
        total = sum(self.engine.scene_counts.values()) or 1
        self.interest_vector = {
            k: v / total for k, v in self.engine.scene_counts.items()
        }

        # 2. 深度偏好：基于 learn/query/analyze 的比例
        lc = self.engine.learn_count
        qc = self.engine.query_count
        ac = self.engine.analyze_count
        learn_ratio = lc / max(lc + qc + ac, 1)
        if learn_ratio > 0.5:
            self.depth_type = "memory_heavy"
        elif learn_ratio < 0.2:
            self.depth_type = "analysis_heavy"
        else:
            self.depth_type = "balanced"

        # 3. 活跃时段：当前小时计数
        hour = str(datetime.now().hour)
        self.active_hours[hour] = self.active_hours.get(hour, 0) + 1

    def match_score(self, category: str) -> float:
        """计算内容与用户画像的匹配度"""
        return self.interest_vector.get(category, 0.1)

    def is_recently_recommended(self, text: str) -> bool:
        """检查是否近期推荐过（3天内）"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        now = time.time()
        # 清理3天前的历史
        self.recommended_history = [
            [h, t] for h, t in self.recommended_history if now - t < 259200
        ]
        return any(h == text_hash for h, _ in self.recommended_history)

    def mark_recommended(self, text: str):
        """标记已推荐"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        self.recommended_history.append([text_hash, time.time()])
        if len(self.recommended_history) > self.MAX_HISTORY:
            self.recommended_history = self.recommended_history[-self.MAX_HISTORY:]

    def reset_recommendations(self):
        """重置推荐历史"""
        self.recommended_history = []

    def to_dict(self) -> Dict:
        return {
            "interest_vector": self.interest_vector,
            "depth_type": self.depth_type,
            "active_hours": self.active_hours,
            "recommended_history": self.recommended_history,
        }

    def from_dict(self, data: Dict):
        self.interest_vector = data.get("interest_vector", {})
        self.depth_type = data.get("depth_type", "balanced")
        self.active_hours = data.get("active_hours", {})
        self.recommended_history = data.get("recommended_history", [])


# ===================================================================
# 第一部分：语义记忆引擎（记忆鸭核心）
# ===================================================================

class MemoryEngine:
    """语义记忆引擎 — 知识存储、图扩散检索、纠错、消歧"""

    def __init__(self):
        self.nodes: List[Dict] = []
        self.hash_index: Dict[int, int] = {}
        self.bitmap_size = 1024 * 8
        self.bitmap = bytearray(self.bitmap_size // 8)
        
        # ===== 多路召回索引（v3.1 查询命中率修复）=====
        self.simhash_index: Dict[int, int] = {}   # SimHash → 节点索引
        self._kw_index: Dict[str, List[int]] = {} # 关键词 → [节点索引列表]
        # ================================================

        # ===== 自动联动模式 =====
        self.auto_mode = True     # 自动学习模式开关（默认开启）
        self.last_auto_text = ""  # 去重防重复学习
        # =========================

        # ===== 监测计数器（持久化）=====
        self.learn_count = 0      # 学习次数
        self.query_count = 0      # 查询次数
        self.search_count = 0     # 图扩散搜索次数
        self.correct_count = 0    # 纠错次数
        self.analyze_count = 0    # 分析次数
        self.summarize_count = 0  # 总结次数
        self.entangle_count = 0   # 纠缠分析次数
        self.token_savings = 0    # 估算节省的Token数
        self.first_use_at = None  # 首次使用时间

        # ===== 对话特征统计（不存原文）=====
        self.short_q_count = 0    # 短句提问 (<20字)
        self.long_q_count = 0     # 长文本提问 (>100字)
        self.medium_q_count = 0   # 中等长度
        self.scene_counts = {}    # 场景计数 {场景名: 次数}
        # =================================

        # ===== 今日数据（每日重置）=====
        self._today = datetime.now().strftime("%Y-%m-%d")
        self.today_learn = 0
        self.today_query = 0
        self.today_search = 0
        self.today_correct = 0
        self.today_analyze = 0
        self.today_summarize = 0
        self.today_entangle = 0
        self.today_tokens = 0
        # =================================

        # 纠错字典
        self._user_typo_dict = {}  # v3.4: 用户纠错词典（自动学习）
        self.typo_dict = {
            "学西": "学习", "字习": "学习", "高心": "高兴",
            "派森": "Python", "派声": "Python",
            "内寸": "内存", "内纯": "内存",
            "数剧库": "数据库", "数倨": "数据",
            "编成": "编程", "边程": "编程",
            "代玛": "代码", "带码": "代码",
            "算发": "算法", "随法": "算法",
            "循还": "循环", "寻环": "循环",
            "路游器": "路由器",
            "安情": "案情", "按情": "案情",
            "派处所": "派出所", "刑贞": "刑侦",
            "反炸": "反诈", "预敬": "预警",
            "研叛": "研判", "穿并": "串并",
            "资今": "资金", "帐号": "账号",
            "冻洁": "冻结",
        }

        # 多义词映射
        self.polysemy_map = {
            "苹果": [
                ("电子产品", ["手机", "电脑", "iphone", "mac", "公司", "平板"]),
                ("食物", ["水果", "吃", "果汁", "甜", "买"]),
            ],
            "内存": [
                ("硬件", ["电脑", "手机", "ram", "8g", "16g", "容量"]),
                ("认知能力", ["回忆", "记住", "忘", "记忆力", "记性"]),
            ],
            "银行": [
                ("金融机构", ["存款", "贷款", "开户", "转账", "网点"]),
                ("地理概念", ["岸边", "河岸", "江边"]),
            ],
        }
        # ===== 上下文增强 =====
        self.context_stack = ContextStack()
        # v3.8: 三层上下文记忆层（KAR融合）
        self._context_memory = ContextMemoryLayer()
        # v3.8: 对话轮次计数器（用于source_turn_index）
        self._turn_index = 0
        # v3.5: 当前工作区
        self._current_workspace = "global"
        # v3.5: 会话级跟踪
        self._session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]  # v3.6.2: 会话唯一ID
        self._session_learned = []      # 本会话学到的知识
        self._session_queried = []       # 本会话查询过的关键词
        self._session_start = time.time()  # 会话开始时间
        self._context_cache = set()      # 已注入的上下文（避免重复）
        # ===== 用户画像 =====
        self.user_profile = UserProfile(self)
        # ======================
        self._load()

    def _hash_64(self, text: str) -> int:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big")

    def _simhash(self, text: str, hash_bits: int = 64) -> int:
        """SimHash — 局部敏感哈希，相似文本映射到相近哈希值
        
        原理：对文本分词后每个词的hash进行加权累加，最终取符号位
        海明距离≤3的SimHash视为相似文本
        """
        # 中文分词：2-4字滑动窗口 + 单字
        words = []
        # 提取中文词组(2-6字)
        words.extend(re.findall(r'[\u4e00-\u9fff]{2,6}', text))
        # 英文单词
        words.extend(re.findall(r'[a-zA-Z]{2,}', text))
        # 数字
        words.extend(re.findall(r'\d{2,}', text))
        # 如果没有有效词，回退到字符级
        if not words:
            words = list(text)
        
        v = [0] * hash_bits
        for word in words:
            wh = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            for i in range(hash_bits):
                bitmask = 1 << i
                if wh & bitmask:
                    v[i] += 1
                else:
                    v[i] -= 1
        
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def _hamming_distance(self, h1: int, h2: int, bits: int = 64) -> int:
        """计算两个SimHash的海明距离"""
        x = h1 ^ h2
        count = 0
        while x and count < bits:
            count += 1
            x &= x - 1
        return count

    def _keyword_index_add(self, text: str, node_idx: int):
        """为节点文本建立关键词倒排索引（含2字子词拆分）"""
        # 提取关键词
        keywords = set()
        keywords.update(re.findall(r'[\u4e00-\u9fff]{2,6}', text))
        keywords.update(re.findall(r'[a-zA-Z]{2,}', text.lower()))
        keywords.update(re.findall(r'\d{2,}', text))
        # v3.1: 对4字以上中文词拆分2字子词（"河图洛书"→"河图"+"图洛"+"洛书"）
        for kw in list(keywords):
            if len(kw) >= 4 and re.match(r'^[\u4e00-\u9fff]+$', kw):
                for i in range(len(kw) - 1):
                    sub = kw[i:i+2]
                    if len(sub) == 2:
                        keywords.add(sub)
        
        for kw in keywords:
            if kw not in self._kw_index:
                self._kw_index[kw] = []
            if node_idx not in self._kw_index[kw]:
                self._kw_index[kw].append(node_idx)

    def _keyword_search(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """关键词倒排索引检索，返回 (节点索引, 匹配度) 列表"""
        keywords = set()
        keywords.update(re.findall(r'[\u4e00-\u9fff]{2,6}', text))
        keywords.update(re.findall(r'[a-zA-Z]{2,}', text.lower()))
        keywords.update(re.findall(r'\d{2,}', text))
        # v3.1: 查询侧也拆分2字子词
        for kw in list(keywords):
            if len(kw) >= 4 and re.match(r'^[\u4e00-\u9fff]+$', kw):
                for i in range(len(kw) - 1):
                    sub = kw[i:i+2]
                    if len(sub) == 2:
                        keywords.add(sub)
        
        counter = {}
        for kw in keywords:
            for idx in self._kw_index.get(kw, []):
                counter[idx] = counter.get(idx, 0) + 1
        
        results = []
        max_hits = max(counter.values()) if counter else 1
        for idx, hits in counter.items():
            score = hits / max_hits
            results.append((idx, score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _bitmap_idx(self, h: int) -> int:
        return h % self.bitmap_size

    def add_node(self, text: str, category: str = "memory", workspace: str = "", source: str = "",
                 genre: str = "", skeleton: str = "", domain: str = "",
                 source_context: str = "", source_turn_index: int = -1,
                 _silent: bool = False) -> int:
        # v3.5: workspace 纳入哈希键，实现项目隔离
        ws = workspace or getattr(self, '_current_workspace', 'global')
        h = self._hash_64(f"{ws}:{text}")
        if h in self.hash_index:
            return self.hash_index[h]
        if not _silent:
            self.learn_count += 1
            if self.first_use_at is None:
                self.first_use_at = datetime.now().isoformat()
            # V3 精准Token计数：按模型分词特性估算
            model = TokenPricer.DEFAULT_MODEL if _DB_AVAILABLE else ""
            if _DB_AVAILABLE:
                actual_tokens = PreciseTokenizer.estimate(text, model)
            else:
                actual_tokens = 200  # 回退到旧估算
            self.token_savings += actual_tokens
            # V3：精准日志 + 平台消耗记录
            if _DB_AVAILABLE:
                log_operation("learn", text, actual_tokens, actual_tokens, category,
                             model=model, platform="workbuddy",
                             input_tokens=actual_tokens, output_tokens=0)
                record_platform_usage("workbuddy", model,
                                     input_tokens=actual_tokens, output_tokens=0,
                                     op_type="learn", saved_tokens=actual_tokens)
                hourly_incr(actual_tokens)
        node = {"text": text, "category": category, "workspace": ws, "strength": 100, "edges": [],
                "source": source or f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · 手动学习",
                "session_id": getattr(self, '_session_id', ''),  # v3.6.2: 写入会话ID
                "learned_at": datetime.now().isoformat(timespec='seconds')}  # v3.6.2: 人类可读学习时间
        # v3.8: KAR融合 — 新增5个增强字段
        if genre:
            node["genre"] = genre
        if skeleton:
            node["skeleton"] = skeleton
        if domain:
            node["domain"] = domain
        if source_context:
            node["source_context"] = source_context
        if source_turn_index >= 0:
            node["source_turn_index"] = source_turn_index
        idx = len(self.nodes)
        self.nodes.append(node)
        self.hash_index[h] = idx
        # v3.1: 同步构建SimHash索引和关键词倒排索引
        sh = self._simhash(text)
        self.simhash_index[sh] = idx
        self._keyword_index_add(text, idx)
        # v3.4: 初始化访问追踪字段
        if "access_count" not in node:
            node["access_count"] = 0
        if "last_accessed" not in node:
            node["last_accessed"] = 0
        if "created_at" not in node:
            node["created_at"] = time.time()
        self._save()
        # v3.5: 同步到 Agent Memory
        if not _silent:
            self._sync_to_agent_memory(text, category, ws)
        # v3.4: 标记语义索引需要重建
        _tfidf_searcher.mark_dirty()
        return idx

    def find_node(self, text: str, workspace: str = None, _debug: bool = False,
                  genre_aware: bool = False) -> Optional[int]:
        """多路召回查询 — 精确哈希 → SimHash模糊 → 关键词倒排 → 子串匹配 → TF-IDF → 体裁感知
        
        v3.6: 全面 workspace 过滤，所有路径都做隔离
        v3.9: genre_aware=True 启用体裁感知检索（第6路）
        """
        ws = workspace if workspace is not None else getattr(self, '_current_workspace', None)
        # 第1路：SHA256精确哈希匹配（带 workspace）
        if ws:
            h = self._hash_64(f"{ws}:{text}")
            idx = self.hash_index.get(h)
            if idx is not None and self.nodes[idx] is not None:
                if _debug: print(f'[find_node] 第1路(带ws)命中: idx={idx}')
                self.query_count += 1
                self._record_query_hit(text, idx, "exact_hash")
                self._touch_node(idx)
                return idx
        # 不带 workspace 的回退（兼容旧数据 — global 节点）
        h = self._hash_64(text)
        idx = self.hash_index.get(h)
        if idx is not None:
            node = self.nodes[idx]
            # v3.6: 仅当该节点是 global 或当前 workspace 时才返回
            if node is not None and (not ws or node.get("workspace", "global") in (ws, "global")):
                if _debug: print(f'[find_node] 第1路(无ws回退)命中: idx={idx} ws={node.get("workspace","global")}')
                self.query_count += 1
                self._record_query_hit(text, idx, "exact_hash")
                return idx
            elif _debug and node is not None:
                print(f'[find_node] 第1路(无ws回退)跳过: idx={idx} node_ws={node.get("workspace","global")} current_ws={ws}')

        # 第2路：SimHash模糊匹配（海明距离≤3视为相似）v3.6: workspace 过滤
        query_sh = self._simhash(text)
        best_idx = None
        best_dist = 999
        for stored_sh, stored_idx in self.simhash_index.items():
            dist = self._hamming_distance(query_sh, stored_sh)
            if dist < best_dist and dist <= 3:  # 海明距离≤3
                # v3.6: workspace 过滤
                node = self.nodes[stored_idx] if stored_idx < len(self.nodes) else None
                if node is not None and ws and node.get("workspace", "global") not in (ws, "global"):
                    if _debug: print(f'[find_node] 第2路SimHash跳过: idx={stored_idx} dist={dist} node_ws={node.get("workspace","global")}')
                    continue
                best_dist = dist
                best_idx = stored_idx
        if best_idx is not None:
            if _debug: print(f'[find_node] 第2路SimHash命中: idx={best_idx} dist={best_dist}')
            self.query_count += 1
            self._record_query_hit(text, best_idx, "simhash")
            return best_idx

        # 第3路：关键词倒排索引检索 v3.6: workspace 过滤
        kw_results = self._keyword_search(text, top_k=5)
        for kw_idx, kw_score in kw_results:
            if kw_score >= 0.5:  # 匹配度≥50%
                node = self.nodes[kw_idx] if kw_idx < len(self.nodes) else None
                if node is not None and ws and node.get("workspace", "global") not in (ws, "global"):
                    if _debug: print(f'[find_node] 第3路关键词跳过: idx={kw_idx} score={kw_score} node_ws={node.get("workspace","global")}')
                    continue  # v3.6: 跳过其他 workspace 的节点
                if _debug: print(f'[find_node] 第3路关键词命中: idx={kw_idx} score={kw_score}')
                self.query_count += 1
                self._record_query_hit(text, kw_idx, "keyword")
                return kw_idx

        # 第4路：子串包含匹配（v3.5: workspace 过滤）
        text_lower = text.lower()
        for i, node in enumerate(self.nodes):
            if node is None: continue
            # workspace 过滤
            if ws and node.get("workspace", "global") not in (ws, "global"):
                continue
            node_text_lower = node["text"].lower()
            if len(text_lower) >= 2 and (text_lower in node_text_lower or node_text_lower in text_lower):
                self.query_count += 1
                self._record_query_hit(text, i, "substring")
                self._touch_node(i)
                return i

        # 第5路：TF-IDF 语义搜索（v3.4 新增）v3.6: workspace 过滤
        semantic_results = _tfidf_searcher.search(self, text, top_k=3)
        for sem_idx, sem_score in semantic_results:
            if sem_score >= 0.3:  # 相似度 ≥ 0.3
                node = self.nodes[sem_idx] if sem_idx < len(self.nodes) else None
                if node is not None and ws and node.get("workspace", "global") not in (ws, "global"):
                    if _debug: print(f'[find_node] 第5路TF-IDF跳过: idx={sem_idx} score={sem_score} node_ws={node.get("workspace","global")}')
                    continue
                if _debug: print(f'[find_node] 第5路TF-IDF命中: idx={sem_idx} score={sem_score}')
                self.query_count += 1
                self._record_query_hit(text, sem_idx, "semantic")
                self._touch_node(sem_idx)
                return sem_idx

        # 第6路：体裁感知检索（v3.9 P1-3）
        # 利用体裁骨架信息指导检索策略，提高特定体裁的召回精度
        if genre_aware:
            result = self._find_node_genre_aware(text, workspace, _debug)
            if result is not None:
                self.query_count += 1
                self._record_query_hit(text, result, "genre_aware")
                self._touch_node(result)
                return result

        return None

    def _find_node_genre_aware(self, text: str, workspace: str = None, _debug: bool = False) -> Optional[int]:
        """v3.9: 体裁感知检索 — 利用骨架信息指导检索策略
        
        process类：优先匹配含步骤关键词的节点
        definition类：优先匹配「X是Y」模式的节点
        argument类：优先匹配含因果链关键词的节点
        data_summary类：优先匹配含数字的节点
        """
        ws = workspace if workspace is not None else getattr(self, '_current_workspace', None)
        genre, _, _, _, _ = GenreClassifier.classify_v2(text)
        skeleton = GenreClassifier.extract_skeleton(text, genre)
        candidates = []

        if genre == 'process':
            # 流程类：用骨架中的步骤关键词搜索
            steps = skeleton.get('steps', [])[:3]
            for step in steps:
                kw_results = self._keyword_search(step, top_k=3)
                candidates.extend(kw_results)

        elif genre == 'definition':
            # 定义类：搜索「X是Y」模式的节点
            def_pat = r'(?:是指|指的是|就是|意为|即|为|是)'
            for i, node in enumerate(self.nodes):
                if node is None: continue
                if ws and node.get("workspace", "global") not in (ws, "global"):
                    continue
                if re.search(def_pat, node["text"]):
                    # 进一步检查是否有领域重叠
                    node_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', node["text"]))
                    query_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
                    overlap = len(node_words & query_words)
                    if overlap > 0:
                        candidates.append((i, overlap))

        elif genre == 'argument':
            # 论证类：搜索含因果链关键词的节点
            causal_kw = ['因为', '所以', '因此', '导致', '由于', '如果']
            for i, node in enumerate(self.nodes):
                if node is None: continue
                if ws and node.get("workspace", "global") not in (ws, "global"):
                    continue
                node_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', node["text"]))
                query_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
                overlap = len(node_words & query_words)
                has_causal = any(kw in node["text"] for kw in causal_kw)
                if overlap > 0 or has_causal:
                    score = overlap + (2 if has_causal else 0)
                    candidates.append((i, score))

        elif genre == 'data_summary':
            # 数据类：优先匹配含数字的节点
            for i, node in enumerate(self.nodes):
                if node is None: continue
                if ws and node.get("workspace", "global") not in (ws, "global"):
                    continue
                if node.get("genre") == "data_summary" or re.search(r'\d+[万亿%]?', node["text"]):
                    node_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', node["text"]))
                    query_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
                    overlap = len(node_words & query_words)
                    if overlap > 0:
                        candidates.append((i, overlap + 1))

        # 排序取最佳候选
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            best_idx = candidates[0][0]
            if _debug:
                print(f'[find_node] 第6路体裁感知命中: idx={best_idx} genre={genre}')
            return best_idx

        return None

    def _record_query_hit(self, text: str, idx: int, method: str):
        """记录查询命中（Token节省统计）"""
        model = TokenPricer.DEFAULT_MODEL if _DB_AVAILABLE else ""
        if _DB_AVAILABLE:
            input_t = PreciseTokenizer.estimate(text, model)
            output_t = int(input_t * 2)
            saved = input_t + output_t
        else:
            input_t = 200
            output_t = 400
            saved = 800
        self.token_savings += saved
        if _DB_AVAILABLE:
            log_operation("query", text, input_t, saved, "",
                         model=model, platform="workbuddy",
                         input_tokens=input_t, output_tokens=0)
            record_platform_usage("workbuddy", model,
                                 input_tokens=input_t, output_tokens=0,
                                 op_type="query", saved_tokens=saved)
            hourly_incr(saved)

    def fuzzy_find(self, keyword: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """模糊搜索 — 关键词子串/分词匹配，返回 (节点索引, 匹配度) 列表"""
        results = []
        kw_len = len(keyword)
        # 将关键词拆分为2-4字子词（容错匹配）
        sub_words = [keyword[i:i+w] for i in range(len(keyword)) for w in range(2, 5) if i+w <= len(keyword)]
        # 去重并按长度排序（优先长词）
        sub_words = sorted(set(sub_words), key=lambda x: -len(x))

        for i, node in enumerate(self.nodes):
            score = 0.0
            # 精确子串匹配（最高分）
            if keyword in node["text"]:
                score = kw_len / max(len(node["text"]), 1) + len(node["edges"]) * 0.05
            else:
                # 分词匹配：统计匹配的子词数量
                matched = sum(1 for sw in sub_words if sw in node["text"])
                if matched > 0:
                    score = matched / max(len(sub_words), 1) * 0.5 + len(node["edges"]) * 0.05
            if score > 0:
                results.append((i, score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def add_edge(self, a: int, b: int, rel: str):
        if a < 0 or a >= len(self.nodes) or b < 0 or b >= len(self.nodes):
            return
        if b not in [e[0] for e in self.nodes[a]["edges"]]:
            self.nodes[a]["edges"].append((b, rel))
        if a not in [e[0] for e in self.nodes[b]["edges"]]:
            self.nodes[b]["edges"].append((a, rel))
        self._save()


    # ====================== v3.5: Agent Memory 双向同步 ======================

    def _sync_to_agent_memory(self, text: str, category: str, workspace: str):
        """将新学知识写入 Agent Memory 的每日日志（去重追加）"""
        try:
            mem_dir = os.environ.get("WORKBUDDY_MEMORY_DIR") or os.path.join(
                os.path.expanduser("~"), ".workbuddy", "memory")
            os.makedirs(mem_dir, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(mem_dir, f"{today}.md")
            entry = f"- [{category}] {text[:100]}\n"
            # 去重：检查今日日志是否已有类似内容
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    existing = f.read()
                if text[:30] in existing:
                    return  # 已存在，跳过
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass  # Agent Memory 不可用时静默跳过
    
    def _sync_memory_md(self):
        """v3.6.7: 每次数据变更时自动更新 WorkBuddy 记忆文件
        
        生成段落式记忆摘要，写入项目专用 _memory.md（WorkBuddy 真正读取的文件）。
        """
        try:
            mem_dir = os.environ.get("WORKBUDDY_MEMORY_DIR") or os.path.join(
                os.path.expanduser("~"), ".workbuddy", "memory")
            os.makedirs(mem_dir, exist_ok=True)
            
            # 找到项目专用的 _memory.md 文件（形如 xxx_memory.md）
            target_file = None
            for f in os.listdir(mem_dir):
                if f.endswith("_memory.md") and f not in ("MEMORY.md",):
                    target_file = os.path.join(mem_dir, f)
                    break
            if not target_file:
                # fallback: 用全局 MEMORY.md
                target_file = os.path.join(mem_dir, "MEMORY.md")
            
            # 收集数据
            valid_nodes = [n for n in self.nodes if n is not None]
            total = len(valid_nodes)
            ws = getattr(self, '_current_workspace', 'global')
            ws_nodes = [n for n in valid_nodes if n.get("workspace", "global") in (ws, "global")]
            
            recent = sorted(ws_nodes, key=lambda n: n.get("created_at", 0), reverse=True)[:8]
            hot = sorted(ws_nodes, key=lambda n: n.get("access_count", 0), reverse=True)[:3]
            
            cats = {}
            for n in ws_nodes:
                c = n.get("category", "未分类")
                cats[c] = cats.get(c, 0) + 1
            top_cats = sorted(cats.items(), key=lambda x: -x[1])[:5]
            
            # 生成段落式记忆
            lines = []
            lines.append(f"你在当前工作区积累了 {len(ws_nodes)} 条知识（全局共 {total} 条）。")
            if top_cats:
                cat_str = "、".join([f"{c}({n}条)" for c, n in top_cats])
                lines.append(f"主要涉及: {cat_str}。")
            if hot:
                hot_str = "、".join([f'「{n["text"][:30]}」' for n in hot])
                lines.append(f"最近常被查询: {hot_str}。")
            if recent:
                lines.append("最近学到:")
                for n in recent[:5]:
                    ts = n.get("learned_at") or ""
                    ts = ts[:10] if ts else ""
                    lines.append(f"- {ts} [{n.get('category','')}] {n['text'][:120]}")
            
            tag_start = "<!-- LEFT_BRAIN_AUTO_START -->"
            tag_end = "<!-- LEFT_BRAIN_AUTO_END -->"
            left_block = f"{tag_start}\n🧠 {datetime.now().strftime('%m-%d %H:%M')}\n\n" + "\n".join(lines) + f"\n{tag_end}"
            
            # 读写目标文件
            if os.path.exists(target_file):
                with open(target_file, "r", encoding="utf-8") as f:
                    existing = f.read()
                if tag_start in existing and tag_end in existing:
                    pre = existing[:existing.index(tag_start)]
                    post = existing[existing.index(tag_end) + len(tag_end):]
                    existing = pre + left_block + post
                else:
                    existing = existing.rstrip() + "\n\n" + left_block
            else:
                existing = left_block
            
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(existing)
            
            # 同时更新全局 MEMORY.md 和 _inject.md
            global_md = os.path.join(mem_dir, "MEMORY.md")
            if os.path.exists(global_md):
                with open(global_md, "r", encoding="utf-8") as f:
                    gm_content = f.read()
                if tag_start in gm_content and tag_end in gm_content:
                    gm_content = gm_content[:gm_content.index(tag_start)] + left_block + gm_content[gm_content.index(tag_end)+len(tag_end):]
                else:
                    gm_content = gm_content.rstrip() + "\n\n" + left_block
                with open(global_md, "w", encoding="utf-8") as f:
                    f.write(gm_content)
            
            try:
                inject_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_inject.md")
                with open(inject_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
            except Exception:
                pass
        except Exception:
            pass



    def _touch_node(self, idx: int):
        """v3.4: 更新节点访问信息（时间衰减+热度提升）"""
        if 0 <= idx < len(self.nodes):
            node = self.nodes[idx]
            node["access_count"] = node.get("access_count", 0) + 1
            node["last_accessed"] = time.time()
            node["last_accessed_at"] = datetime.now().isoformat(timespec='seconds')  # v3.6.2: 可读访问时间
            # 每 10 次访问自动保存
            if node["access_count"] % 10 == 0:
                self._save()

    # ====================== v3.5: 增删改查 ======================

    def update_node(self, keyword: str, new_text: str) -> Dict:
        """修改已存在的知识节点"""
        idx = self.find_node(keyword)
        if idx is None:
            # 尝试模糊查找
            results = self.fuzzy_find(keyword, top_k=3)
            if results:
                candidates = [{"text": self.nodes[r[0]]["text"][:60], "score": round(r[1], 2)} for r in results]
                return {"status": "needs_clarification", "message": "没找到精确匹配，以下可能是你要改的：", "candidates": candidates}
            return {"status": "not_found", "keyword": keyword}
        old_text = self.nodes[idx]["text"]
        self.nodes[idx]["text"] = new_text
        self.nodes[idx]["updated_at"] = time.time()
        self.nodes[idx]["updated_at_iso"] = datetime.now().isoformat(timespec='seconds')  # v3.6.2: 可读更新时间
        self.nodes[idx]["updated_by_session"] = getattr(self, '_session_id', '')  # v3.6.2: 哪次对话更新的
        # 重建索引
        self._keyword_index_add(new_text, idx)
        self._save()
        _tfidf_searcher.mark_dirty()
        return {"status": "ok", "action": "修改", "old": old_text[:60], "new": new_text[:60], "idx": idx}

    def delete_node(self, keyword: str) -> Dict:
        """删除知识节点"""
        idx = self.find_node(keyword)
        if idx is None:
            results = self.fuzzy_find(keyword, top_k=3)
            if results:
                candidates = [{"text": self.nodes[r[0]]["text"][:60], "score": round(r[1], 2)} for r in results]
                return {"status": "needs_clarification", "message": "没找到精确匹配，以下可能是你要删的：", "candidates": candidates}
            return {"status": "not_found", "keyword": keyword}
        text = self.nodes[idx]["text"]
        # 清除其他节点指向此节点的边
        for node in self.nodes:
            node["edges"] = [e for e in node.get("edges", []) if e[0] != idx]
        # 清除 hash 索引
        h = self._hash_64(text)
        if h in self.hash_index:
            del self.hash_index[h]
        # 删除节点
        self.nodes[idx] = None  # 标记删除（保留索引稳定）
        self._save()
        _tfidf_searcher.mark_dirty()
        return {"status": "ok", "action": "删除", "text": text[:60], "idx": idx}

    def list_nodes(self, category: str = "", page: int = 1, page_size: int = 20, workspace: str = None) -> Dict:
        """分页列出知识节点
        v3.6: 支持 workspace 过滤
        """
        ws = workspace or getattr(self, '_current_workspace', None)
        nodes = [n for n in self.nodes if n is not None]
        if category:
            nodes = [n for n in nodes if n.get("category", "") == category]
        # v3.6: workspace 过滤（显示当前 workspace + global）
        if ws:
            nodes = [n for n in nodes if n.get("workspace", "global") in (ws, "global")]
        total = len(nodes)
        total_pages = max(1, (total + page_size - 1) // page_size)
        start = (page - 1) * page_size
        page_nodes = nodes[start:start + page_size]
        items = [{"idx": i, "text": n["text"][:80], "category": n.get("category", ""),
                   "edges": len(n.get("edges", [])), "access": n.get("access_count", 0),
                   "learned_at": n.get("learned_at", ""),  # v3.6.2: 可读学习时间
                   "updated_at_iso": n.get("updated_at_iso", ""),  # v3.6.2: 可读更新时间
                   "session_id": n.get("session_id", "")}  # v3.6.2: 写入会话ID 
                 for i, n in enumerate(self.nodes) if n is not None and n in page_nodes]
        return {"status": "ok", "total": total, "page": page, "total_pages": total_pages, "items": items, "categories": list(set(n.get("category", "") for n in nodes))}

    # ====================== v3.5: WorkBuddy 联动 ======================

    def inject_context(self, text: str, max_items: int = 3) -> Dict:
        """自动注入相关知识到对话上下文 — 不需要用户主动查询
        v3.6: 尊重 workspace 隔离，find_node 自动按当前 workspace 过滤
        """
        keywords = re.findall(r'[一-鿿]{2,6}', text)[:8]
        results = []
        seen = set()
        for kw in keywords:
            if kw in self._context_cache or kw in seen:
                continue
            idx = self.find_node(kw)  # v3.6: find_node 已内置 workspace 过滤
            if idx is not None:
                node = self.nodes[idx]
                # v3.6: 放宽条件 — 孤立节点也可注入（单条知识也有价值）
                related = [self.nodes[e[0]]["text"][:60] for e in node.get("edges", [])[:2]]
                results.append({
                    "trigger": kw,
                    "match": node["text"][:80],
                    "related": related,
                })
                self._context_cache.add(kw)
                seen.add(kw)
            if len(results) >= max_items:
                break
        return {"status": "ok", "injected": len(results), "items": results}

    def session_summary(self) -> Dict:
        """生成当前会话摘要"""
        learned = len(self._session_learned)
        queried = len(self._session_queried)
        duration = int((time.time() - self._session_start) / 60)
        total_nodes = len([n for n in self.nodes if n is not None])
        # 最近学到的
        recent = []
        for item in self._session_learned[-5:]:
            recent.append(item)
        return {
            "status": "ok",
            "duration_minutes": duration,
            "learned_this_session": learned,
            "queried_this_session": queried,
            "total_knowledge": total_nodes,
            "recent_learned": recent,
        }

    def session_start(self) -> Dict:
        """新会话开始时调用 — 回顾上次学到了什么
        v3.6: 优先展示当前 workspace 的知识
        v3.6.2: 每次新会话生成唯一 session_id
        v3.6.3: 输出完整上下文（workspace概览 + 分类摘要 + 高频知识 + 最近知识）
        """
        self._session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]  # v3.6.2: 新会话ID
        self._session_start = time.time()
        self._session_learned = []
        self._session_queried = []
        self._context_cache = set()
        ws = getattr(self, '_current_workspace', None)

        # ---- 1. 收集当前 workspace + global 的所有活跃节点 ----
        all_nodes = []
        for node in self.nodes:
            if node is None:
                continue
            node_ws = node.get("workspace", "global")
            if ws and node_ws not in (ws, "global"):
                continue
            all_nodes.append(node)

        total = len(all_nodes)

        # ---- 2. 分类统计 ----
        cat_stats = {}
        for n in all_nodes:
            cat = n.get("category", "未分类")
            cat_stats[cat] = cat_stats.get(cat, 0) + 1
        # 按数量排序，取 top 5 分类
        top_cats = sorted(cat_stats.items(), key=lambda x: -x[1])[:5]

        # ---- 3. 高频访问知识（access_count 最高的 top 5）----
        high_freq = sorted(all_nodes, key=lambda n: n.get("access_count", 0), reverse=True)[:5]
        high_freq_items = [{"text": n["text"][:80], "access": n.get("access_count", 0),
                            "learned_at": n.get("learned_at", "")} for n in high_freq if n.get("access_count", 0) > 0]

        # ---- 4. 最近学习的知识（按 learned_at/created_at 倒序 top 5）----
        def _sort_key(node):
            lat = node.get("learned_at", "")
            if isinstance(lat, str) and lat:
                return lat  # ISO 格式字符串，可排序
            cat = node.get("created_at", 0)
            if isinstance(cat, (int, float)) and cat:
                return str(cat)  # Unix 时间戳转字符串
            return ""  # 兜底
        recent = sorted(all_nodes, key=_sort_key, reverse=True)[:5]
        recent_items = [{"text": n["text"][:80], "learned_at": n.get("learned_at", ""),
                         "session_id": n.get("session_id", "")} for n in recent]

        # ---- 5. 最近更新的知识（有 updated_at_iso 的 top 3）----
        updated_nodes = [n for n in all_nodes if n.get("updated_at_iso")]
        updated_nodes.sort(key=lambda n: n.get("updated_at_iso", ""), reverse=True)
        recent_updated = [{"text": n["text"][:80], "updated_at": n.get("updated_at_iso", ""),
                           "updated_by": n.get("updated_by_session", "")} for n in updated_nodes[:3]]

        # ---- 6. workspace 分布概览 ----
        ws_counts = {}
        for n in self.nodes:
            if n is None:
                continue
            ws_name = n.get("workspace", "global")
            ws_counts[ws_name] = ws_counts.get(ws_name, 0) + 1

        result = {
            "status": "ok",
            "workspace": ws or "global",
            "session_id": self._session_id,  # v3.6.2
            "total_knowledge": total,
            "total_global": len([n for n in self.nodes if n is not None]),
            "category_summary": [{"category": cat, "count": cnt} for cat, cnt in top_cats],
            "high_freq_knowledge": high_freq_items,
            "recent_knowledge": recent_items,
            "recent_updated": recent_updated,
            "workspace_distribution": ws_counts,
        }

        # v3.6.7: _inject.md + MEMORY.md 在 _save() 中自动更新（每次数据变更触发）
        # session_start 仅重置会话变量，不重复写文件
        try:
            self._sync_memory_md()
        except Exception:
            pass

        return result

    def selfcheck(self) -> Dict:
        """v3.10: 自检系统 — 9项检查 + 6项修复能力

        触发方式：/左脑 自检

        检查项：
        1. 数据库完整性（空/损坏节点 → 清理+重建索引）
        2. 注入一致性（_inject.md vs 实际数据 → 重写聚合版）
        3. 对话历史可达（需AI调用conversation_search）
        4. Memory日志文件（目录/文件可读性）
        5. 纠缠场一致性（断裂关联 → 清理dangling edges）
        6. workspace隔离校验（归属异常检测）
        7. 系统运行状态
        8. Token数据库一致性（JSON vs SQLite → 同步修复）
        9. 备份时效（超过30天告警）
        """
        try:
            from scripts.selfcheck import LeftBrainSelfCheck
        except ImportError:
            # fallback: 直接导入
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "selfcheck",
                str(Path(__file__).resolve().parent / "scripts" / "selfcheck.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            LeftBrainSelfCheck = mod.LeftBrainSelfCheck

        checker = LeftBrainSelfCheck(engine=self)
        return checker.run()

    def learn_from_content(self, text: str, source: str = "unknown") -> Dict:
        """从任意内容中提取知识（搜索结果、文件内容、网页等）
        v3.6: 自动更新已有条目，避免重复学习相似内容
        """
        learned = []
        updated = []
        # v3.6.9: 过滤 MEMORY.md 格式内容，防止自动学习把左脑摘要当知识
        _GARBAGE_KW = [
            'LEFT_BRAIN_AUTO', '## 🧠', '左脑状态快照', '左脑记忆',
            '工作区积累了', '最近常被查询', '知识总量:', '分类: 标讯',
            '- [标讯] 最近', '- [通用] 8万',
        ]
        sents = [s.strip() for s in re.split(r'(?<=[。！？\.\n])\s*', text) if len(s.strip()) > 10]
        for sent in sents[:5]:
            # 过滤：太短、太像标题、纯数字
            if len(sent) < 10 or re.match(r'^[\d\s\.\-\(\)]+$', sent):
                continue
            # v3.6.9: 过滤 MEMORY.md 垃圾内容
            if any(kw in sent for kw in _GARBAGE_KW):
                continue
            # v3.6: 检查是否有高度相似的已有节点
            similar = self._find_similar_node(sent)
            if similar is not None:
                idx, sim_score = similar
                if sim_score >= 0.6:
                    # 自动更新：内容有变化时更新旧节点
                    old_text = self.nodes[idx]["text"]
                    if old_text != sent:
                        self.nodes[idx]["text"] = sent
                        self.nodes[idx]["updated_at"] = time.time()
                        self.nodes[idx]["updated_at_iso"] = datetime.now().isoformat(timespec='seconds')  # v3.6.2
                        self.nodes[idx]["update_source"] = source
                        self.nodes[idx]["updated_by_session"] = getattr(self, '_session_id', '')  # v3.6.2
                        self._keyword_index_add(sent, idx)
                        _tfidf_searcher.mark_dirty()
                        updated.append({"old": old_text[:60], "new": sent[:60], "score": round(sim_score, 2)})
                    continue  # 无论是否更新，都不再新增
            cat = self.auto_category(sent)
            idx = self.add_node(sent, cat)
            learned.append({"text": sent[:80], "category": cat, "source": source})
        result = {"status": "ok", "learned_count": len(learned), "items": learned}
        if updated:
            result["updated_count"] = len(updated)
            result["updated_items"] = updated
        return result

    def _find_similar_node(self, text: str, threshold: float = 0.5) -> Optional[Tuple[int, float]]:
        """v3.6: 查找与给定文本高度相似的已有节点

        综合使用 SimHash + 关键词重叠度判断相似性：
        - SimHash 海明距离 ≤ 2 → 高度相似
        - 核心关键词重叠率 ≥ 60% → 高度相似
        返回 (node_idx, similarity_score) 或 None
        """
        ws = getattr(self, '_current_workspace', None)
        candidates = []

        # 方法1：SimHash 模糊匹配
        query_sh = self._simhash(text)
        for stored_sh, stored_idx in self.simhash_index.items():
            dist = self._hamming_distance(query_sh, stored_sh)
            if dist <= 2:
                node = self.nodes[stored_idx]
                if node is None:
                    continue
                node_ws = node.get("workspace", "global")
                if ws and node_ws not in (ws, "global"):
                    continue
                score = 1.0 - (dist / 8.0)  # 距离0=1.0, 距离2=0.75
                candidates.append((stored_idx, score))

        # 方法2：关键词重叠率
        text_kws = set(re.findall(r'[\u4e00-\u9fff]{2,6}', text))
        text_kws.update(re.findall(r'[a-zA-Z]{2,}', text.lower()))
        if text_kws:
            for idx in self._keyword_search(text, top_k=5):
                node_idx, kw_score = idx
                node = self.nodes[node_idx]
                if node is None:
                    continue
                node_ws = node.get("workspace", "global")
                if ws and node_ws not in (ws, "global"):
                    continue
                # 计算关键词重叠率
                node_kws = set(re.findall(r'[\u4e00-\u9fff]{2,6}', node["text"]))
                node_kws.update(re.findall(r'[a-zA-Z]{2,}', node["text"].lower()))
                if node_kws:
                    overlap = len(text_kws & node_kws) / max(len(text_kws | node_kws), 1)
                    if overlap >= 0.6:
                        candidates.append((node_idx, overlap))

        if not candidates:
            return None
        # 取最高分
        candidates.sort(key=lambda x: -x[1])
        return candidates[0]

    def get_workspace_key(self, path: str = "") -> str:
        """v3.6: 稳定 workspace key — 从路径中提取项目名，而非哈希整个路径

        策略（优先级从高到低）：
        1. 含 .workbuddy 目录 → 取其父目录名作为项目标识
        2. 含 WorkBuddy 日期目录(YYYY-MM-DD-HH-MM-SS) → 取其父目录名
        3. 纯路径回退 → 取最后一级非日期目录名
        同一项目多次会话产生相同的 workspace key。
        """
        if not path:
            path = os.getcwd()
        path = os.path.normpath(path)
        parts = path.replace("\\", "/").split("/")

        # 策略1：找 .workbuddy 目录，取其父目录名
        # WorkBuddy 会话路径形如 .../项目名/.workbuddy/... 或 .../项目名/YYYY-MM-DD...
        for i, p in enumerate(parts):
            if p == ".workbuddy" and i > 0:
                project_name = parts[i - 1]
                # 排除纯日期格式的目录名
                if not re.match(r'^\d{4}-\d{2}-\d{2}', project_name):
                    return f"ws_{project_name}"

        # 策略2：跳过 WorkBuddy 日期目录(YYYY-MM-DD-HH-MM-SS)，取其父目录
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')
        for i, p in enumerate(parts):
            if date_pattern.match(p) and i > 0:
                project_name = parts[i - 1]
                if project_name not in ("WorkBuddy", "workbuddy"):
                    return f"ws_{project_name}"

        # 策略3：取最后一级非日期目录名
        for p in reversed(parts):
            if p and not date_pattern.match(p) and not re.match(r'^\d{4}-\d{2}-\d{2}$', p):
                return f"ws_{p}"

        # 最终回退：路径哈希
        h = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        return f"ws_{h}"

    def suggest_if_relevant(self, text: str, threshold: float = 0.3) -> Dict:
        """主动介入 — 检测到相关话题时自动建议"""
        keywords = re.findall(r'[一-鿿]{2,6}', text)[:6]
        suggestions = []
        for kw in keywords:
            if kw in self._context_cache:
                continue
            # TF-IDF 语义搜索
            results = _tfidf_searcher.search(self, kw, top_k=2)
            for idx, score in results:
                if score >= threshold and idx < len(self.nodes):
                    node = self.nodes[idx]
                    if node is not None:
                        suggestions.append({
                            "trigger": kw,
                            "suggestion": node["text"][:100],
                            "confidence": round(score, 2),
                            "edges": len(node.get("edges", [])),
                        })
                        self._context_cache.add(kw)
        suggestions.sort(key=lambda x: -x["confidence"])
        return {"status": "ok", "suggestions": suggestions[:3]}

    def query_diffusion(self, start_text: str, max_hops: int = 3, workspace: str = None) -> Dict:
        self.search_count += 1
        idx = self.find_node(start_text)
        if idx is None:
            corrected = self.fix_typo(start_text)
            if corrected != start_text:
                idx = self.find_node(corrected)
                if idx is not None:
                    start_text = corrected
        if idx is None:
            return {"status": "not_found", "query": start_text}
        visited: Set[int] = {idx}
        results = []
        queue = [(idx, 0, "root", [idx])]
        while queue:
            cur, hop, relation, path = queue.pop(0)
            if cur != idx:
                # v3.5: 可解释关联 — 附带路径说明
                path_str = " → ".join(self.nodes[p]["text"][:20] for p in path)
                results.append({
                    "text": self.nodes[cur]["text"],
                    "hop": hop,
                    "relation": relation,
                    "path": path_str,
                    "explain": f"第{hop}跳：通过「{self.nodes[path[hop-1]]['text'][:15]}」的「{relation}」关系找到",
                })
            if hop >= max_hops:
                continue
            for nxt, r in self.nodes[cur]["edges"]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, hop + 1, r, path + [nxt]))
        by_hop = {}
        for item in results:
            by_hop.setdefault(item["hop"], []).append(item)
        return {
            "status": "ok", "query": start_text,
            "center_node": self.nodes[idx]["text"],
            "total": len(results),
            "by_hop": {str(h): len(v) for h, v in by_hop.items()},
            "details": results,
        }

    def fix_typo(self, text: str) -> str:
        """v3.4: 智能纠错 — 静态字典 + 可学习用户词典 + 编辑距离模糊匹配"""
        corrected = text
        # 第1层：精确匹配（静态字典 + 自动学习的用户词典）
        combined = {}
        combined.update(self.typo_dict)
        combined.update(getattr(self, '_user_typo_dict', {}))
        for wrong, right in combined.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)
        # 第2层：编辑距离模糊匹配
        words = re.findall(r'[\u4e00-\u9fff]{2,8}', corrected)
        for word in words:
            if word in combined:
                continue
            candidates = []
            for node in self.nodes:
                node_words = re.findall(r'[\u4e00-\u9fff]{2,8}', node.get("text", ""))
                for nw in node_words:
                    dist = self._edit_distance(word, nw)
                    if 0 < dist <= 2:
                        candidates.append((nw, dist, node.get("access_count", 0)))
            if candidates:
                candidates.sort(key=lambda x: (-x[2], x[1]))
                best = candidates[0][0]
                self._user_typo_dict[word] = best
                corrected = corrected.replace(word, best)
        return corrected

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein 编辑距离"""
        if len(s1) < len(s2):
            return MemoryEngine._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(prev[j] if c1 == c2 else 1 + min(prev[j], prev[j+1], curr[-1]))
            prev = curr
        return prev[-1]

    def disambiguate(self, word: str, context: str = "") -> str:
        ctx = context.lower()
        if word in self.polysemy_map:
            for label, keywords in self.polysemy_map[word]:
                if any(k in ctx for k in keywords):
                    return label
        return "通用"

    def detect_relation(self, a: str, b: str) -> str:
        if not a or not b: return "未知"
        if a == b: return "同义"
        if a in b or b in a: return "同义"
        if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]: return "派生"
        if a[0] == b[0]: return "相关"
        return "扩散"

    def auto_category(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["项目", "部署", "方案", "架构", "系统", "招标", "投标", "采购", "评分"]): return "标讯"
        if any(k in t for k in ["案件", "研判", "反诈", "预警", "串并", "线索", "刑侦", "涉网"]): return "案件"
        if any(k in t for k in ["python", "代码", "api", "编程", "算法", "函数", "bug", "debug"]): return "代码"
        if any(k in t for k in ["学习", "教程", "课程", "考试", "知道", "什么", "如何", "怎么"]): return "学习"
        if any(k in t for k in ["合同", "公文", "报告", "方案", "通知", "函", "请示", "纪要"]): return "公文"
        if any(k in t for k in ["数据", "趋势", "对比", "统计", "增长", "下降", "占比", "分析"]): return "分析"
        if any(k in t for k in ["你好", "哈哈", "谢谢", "是的", "早安", "晚安", "没事"]): return "闲聊"
        return "通用"

    def stats(self) -> Dict:
        edge_count = sum(len(n["edges"]) for n in self.nodes) // 2
        return {
            "version": "3.1", "nodes": len(self.nodes),
            "edges": edge_count, "typo_rules": len(self.typo_dict),
            "polysemy_words": len(self.polysemy_map),
            "learn_count": self.learn_count,
            "query_count": self.query_count,
            "search_count": self.search_count,
            "correct_count": self.correct_count,
            "analyze_count": self.analyze_count,
            "summarize_count": self.summarize_count,
            "entangle_count": self.entangle_count,
            "token_savings": self.token_savings,
        }

    def dashboard(self) -> Dict:
        """监测仪表盘 — 展示实时节省的 Token、时间等核心指标"""
        edge_count = sum(len(n["edges"]) for n in self.nodes) // 2
        # 操作统计
        total_mem_ops = self.learn_count + self.query_count + self.search_count + self.correct_count
        total_ops = total_mem_ops + self.analyze_count + self.summarize_count + self.entangle_count

        # ==== 时间相关 ====
        hours_saved_s = self.token_savings * 5.0 / 800  # 每 800 token ≈ 5s LLM 推理
        hours_saved = hours_saved_s / 3600
        # 首次使用到现在
        days_active = 0
        seconds_active = 0
        if self.first_use_at:
            try:
                delta = datetime.now() - datetime.fromisoformat(self.first_use_at)
                days_active = max(1, delta.days)
                seconds_active = delta.total_seconds()
            except:
                days_active = max(1, days_active)
        # 增速：日均节省时间
        daily_save_hours = hours_saved / max(1, days_active)
        # 等效生成速度 (token/s)：节省的 token ÷ 节省的时间（按LLM推算）
        equiv_speed = self.token_savings / max(1, hours_saved_s) if hours_saved_s > 0 else 0

        # ==== 金额 V3（精准计价）====
        if _DB_AVAILABLE:
            money_saved = TokenPricer.calc_saving(TokenPricer.DEFAULT_MODEL, self.token_savings, "mixed")
        else:
            TOKEN_PRICE_PER_K = 0.01  # 回退旧价
            money_saved = self.token_savings / 1000 * TOKEN_PRICE_PER_K

        # ==== CO₂ ====
        co2_saved_g = total_mem_ops * 0.042

        # ==== 功能发挥（各功能占比）====
        func_counts = {
            "学习": self.learn_count,
            "查询": self.query_count,
            "搜索": self.search_count,
            "纠错": self.correct_count,
            "分析": self.analyze_count,
            "总结": self.summarize_count,
            "纠缠": self.entangle_count,
        }
        func_active = {k: v for k, v in func_counts.items() if v > 0}
        func_total = sum(func_counts.values())
        func_share = {}
        if func_total > 0:
            for k, v in func_counts.items():
                pct = round(v / func_total * 100, 1)
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                func_share[k] = {"count": v, "pct": pct, "bar": bar}

        # ==== 上下文窗口使用率 ====
        max_nodes = self.bitmap_size  # bitmap_size bits = 最大节点数
        ctx_usage_pct = round(len(self.nodes) / max(1, max_nodes) * 100, 2)
        ctx_bar = "█" * int(ctx_usage_pct / 2) + "░" * (50 - int(ctx_usage_pct / 2))

        return {
            "version": "3.1",
            "summary": {
                "total_operations": total_ops,
                "memory_operations": total_mem_ops,
                "knowledge_nodes": len(self.nodes),
                "knowledge_edges": edge_count,
                "days_active": days_active,
                "seconds_active": round(seconds_active),
                "first_used": self.first_use_at or "首次使用",
            },
            "savings": {
                "tokens_saved": self.token_savings,
                "tokens_saved_display": self._fmt_number(self.token_savings),
                "time_saved_hours": round(hours_saved, 2),
                "time_saved_display": self._fmt_time(hours_saved),
                "money_saved": round(money_saved, 2),
                "money_saved_display": f"¥{money_saved:.2f}",
                "daily_save_hours": round(daily_save_hours, 2),
                "daily_save_display": self._fmt_time(daily_save_hours) + "/天",
                "equiv_speed": round(equiv_speed),
                "equiv_speed_display": f"{equiv_speed:,.0f} token/s",
                "co2_saved_g": round(co2_saved_g, 2),
                "co2_saved_kg": round(co2_saved_g / 1000, 4),
            },
            "usage": {
                "total_actions": func_total,
                "active_functions": len(func_active),
                "detail": func_counts,
                "share": func_share,
                # 对话特征
                "short_questions": self.short_q_count,
                "medium_questions": self.medium_q_count,
                "long_questions": self.long_q_count,
                "scenes": self.scene_counts,
                # 今日数据
                "today": {
                    "learn": self.today_learn,
                    "query": self.today_query,
                    "search": self.today_search,
                    "correct": self.today_correct,
                    "analyze": self.today_analyze,
                    "summarize": self.today_summarize,
                    "entangle": self.today_entangle,
                    "tokens": self.today_tokens,
                },
            },
            "engine": {
                "typo_rules": len(self.typo_dict),
                "polysemy_words": len(self.polysemy_map),
                "graph_density": round(edge_count / max(1, len(self.nodes)), 2),
                "context_usage_pct": ctx_usage_pct,
                "context_usage_bar": ctx_bar,
                "max_nodes": max_nodes,
                "used_nodes": len(self.nodes),
            },
        }

    @staticmethod
    def _fmt_number(n: int) -> str:
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000: return f"{n/1_000:.1f}K"
        return str(n)

    @staticmethod
    def _fmt_time(hours: float) -> str:
        if hours >= 24:
            days = hours / 24
            return f"{days:.1f}天" if days < 30 else f"{days/30:.1f}个月"
        elif hours >= 1:
            return f"{hours:.1f}小时"
        elif hours * 60 >= 1:
            return f"{hours*60:.0f}分钟"
        else:
            return f"{hours*3600:.0f}秒"

    def dashboard_text(self) -> str:
        """返回格式化的仪表盘文本，适合 WorkBuddy 展示"""
        d = self.dashboard()
        s = d["summary"]
        sv = d["savings"]
        u = d["usage"]
        e = d["engine"]
        lines = []
        lines.append("📊 左脑 · ToKen 监测助手")
        lines.append("=" * 40)
        lines.append(f"⏱ 已运行 {s['days_active']} 天 | 总操作 {s['total_operations']} 次")
        lines.append(f"🧠 知识图谱: {s['knowledge_nodes']} 节点 / {s['knowledge_edges']} 关联")
        lines.append("")
        lines.append("💰 节省统计")
        lines.append("-" * 30)
        lines.append(f"  Token:   {sv['tokens_saved_display']}")
        lines.append(f"  💵 金额:  {sv['money_saved_display']}")
        lines.append(f"  ⏱ 时间:  {sv['time_saved_display']} (日均{sv['daily_save_display']})")
        lines.append(f"  ⚡ 等效:  {sv['equiv_speed_display']}")
        lines.append(f"  🌱 CO₂:   {sv['co2_saved_g']}g")
        lines.append("")
        lines.append("📈 功能发挥 ({}/{} 项活跃)".format(u['active_functions'], u['total_actions'] > 0 and '7' or '0'))
        lines.append("-" * 30)
        for name, info in sorted(u['share'].items(), key=lambda x: -x[1]['count']):
            if info['count'] > 0:
                lines.append(f"  {name:<4s} {info['bar']} {info['count']:>4d}次 ({info['pct']}%)")
        lines.append("")
        lines.append("🎯 上下文窗口")
        lines.append("-" * 30)
        lines.append(f"  使用率: {e['context_usage_bar']} {e['context_usage_pct']}%")
        lines.append(f"  ({e['used_nodes']}/{e['max_nodes']} 节点)")
        lines.append("")
        lines.append("⚙️ 引擎状态")
        lines.append("-" * 30)
        lines.append(f"  纠错规则: {e['typo_rules']}条 | 多义词: {e['polysemy_words']}组")
        lines.append(f"  图谱密度: {e['graph_density']}")
        return "\n".join(lines)

    def usage_analysis(self) -> Dict:
        """使用习惯分析 — 总结用户的使用模式"""
        d = self.dashboard()
        u = d["usage"]
        s = d["summary"]
        sv = d["savings"]
        e = d["engine"]

        # 1. 使用模式分类：偏记忆型 / 偏推理型 / 均衡型
        mem_ops = u["detail"]["学习"] + u["detail"]["查询"] + u["detail"]["搜索"] + u["detail"]["纠错"]
        reasoning_ops = u["detail"]["分析"] + u["detail"]["总结"] + u["detail"]["纠缠"]
        total = mem_ops + reasoning_ops
        if total == 0:
            pattern = "未开始"
            pattern_desc = "还没开始使用左脑，快试试吧！"
        elif mem_ops > reasoning_ops * 1.5:
            pattern = "记忆达人 🧠"
            pattern_desc = "你更倾向于使用记忆功能（学习/查询/搜索），适合快速存储和检索知识"
        elif reasoning_ops > mem_ops * 1.5:
            pattern = "推理高手 🔍"
            pattern_desc = "你更倾向于使用推理功能（分析/总结/纠缠），善于深度挖掘信息"
        else:
            pattern = "全面均衡 ⚖️"
            pattern_desc = "记忆和推理功能使用均衡，左右脑协同最佳状态"

        # 2. 活跃度评估
        days = s["days_active"]
        daily_ops = round(total / max(1, days), 1)
        if daily_ops == 0:
            activity = "未活跃"
        elif daily_ops < 3:
            activity = "轻度使用 🌱"
        elif daily_ops < 10:
            activity = "中度使用 🌿"
        elif daily_ops < 30:
            activity = "活跃用户 🔥"
        else:
            activity = "重度用户 💪"

        # 3. 知识图谱健康度
        nodes = s["knowledge_nodes"]
        edges = s["knowledge_edges"]
        if nodes == 0:
            graph_health = "待建设"
            graph_tip = "还没有存储知识，试试 /左脑 learn 开始学习"
        elif edges == 0:
            graph_health = "孤岛型"
            graph_tip = f"已有 {nodes} 个知识点但没有关联，试试 /左脑 relate 建立连接"
        elif edges / nodes > 2:
            graph_health = "高密度网 🕸️"
            graph_tip = f"知识关联丰富（{edges}条边/{nodes}节点），图扩散搜索效果最佳"
        elif edges / nodes > 0.5:
            graph_health = "生长中 🌱"
            graph_tip = f"知识网络在成长（{edges}条边/{nodes}节点），继续建立更多关联"
        else:
            graph_health = "稀疏型"
            graph_tip = f"关联较少（{edges}/{nodes}），关联越多搜索越准"

        # 4. 最常用功能排行
        sorted_funcs = sorted(u["detail"].items(), key=lambda x: -x[1])
        top_func = sorted_funcs[0][0] if sorted_funcs and sorted_funcs[0][1] > 0 else "无"
        top_count = sorted_funcs[0][1] if sorted_funcs else 0

        return {
            "pattern": pattern,
            "pattern_desc": pattern_desc,
            "activity": activity,
            "daily_operations": daily_ops,
            "graph_health": graph_health,
            "graph_tip": graph_tip,
            "top_function": top_func,
            "top_function_count": top_count,
            "total_operations": total,
            "days_active": days,
            "mem_ratio": round(mem_ops / max(1, total) * 100, 1),
            "reasoning_ratio": round(reasoning_ops / max(1, total) * 100, 1),
        }

    def _analysis_text(self) -> str:
        """格式化使用习惯分析文本，适合 WorkBuddy 展示"""
        a = self.usage_analysis()
        lines = []
        lines.append("🧠 使用习惯分析")
        lines.append("=" * 30)
        lines.append(f"  类型: {a['pattern']}")
        lines.append(f"  活跃度: {a['activity']} ({a['daily_operations']}次/天)")
        lines.append(f"  记忆型操作: {a['mem_ratio']}% | 推理型操作: {a['reasoning_ratio']}%")
        lines.append(f"  最常用功能: {a['top_function']} ({a['top_function_count']}次)")
        lines.append(f"  知识图谱: {a['graph_health']}")
        lines.append(f"  {a['graph_tip']}")
        lines.append("")
        lines.append("💡 试试 /左脑 tips 看优化建议")
        return "\n".join(lines)

    def optimization_tips(self) -> List[str]:
        """根据使用情况生成优化建议"""
        a = self.usage_analysis()
        d = self.dashboard()
        u = d["usage"]
        s = d["summary"]
        tips = []

        n = s["knowledge_nodes"]
        e = s["knowledge_edges"]
        lc = u["detail"]["学习"]
        qc = u["detail"]["查询"]
        sc = u["detail"]["搜索"]
        cc = u["detail"]["纠错"]
        ac = u["detail"]["分析"]
        suc = u["detail"]["总结"]
        ec = u["detail"]["纠缠"]

        # 基于知识图谱的优化
        if n == 0:
            tips.append("💡 开始学习第一条知识：/左脑 learn <你想记住的内容>")
        elif e == 0 and n > 3:
            tips.append("🔗 你的知识之间还没有关联：试试 /左脑 relate A,B 建立连接")
        elif e > 0 and e / n < 0.5:
            tips.append("🕸️ 关联偏少，多用 /左脑 search 图扩散搜索自动建立关联")

        # 基于功能使用的优化
        if lc > 0 and sc == 0:
            tips.append("🔍 学了很多但没搜过？图扩散搜索能挖出知识间的隐藏关联")
        if ac > 3 and suc == 0:
            tips.append("📄 做了分析但没总结过？试试 /左脑 summarize 提炼核心")
        if qc > 3 and sc == 0:
            tips.append("🕸️ 单点查询用得多，试试 /左脑 search 图扩散一次找到所有关联")
        if cc > 3 and lc == 0:
            tips.append("📝 纠错用得多但没学过新知识？试试 /左脑 learn 存下来")
        if u["total_actions"] > 10 and not self.auto_mode:
            tips.append("⚡ 操作已经很熟练了，开启 /左脑 auto on 自动学习更省心")

        # 基于效率的优化
        if n > 10 and d["savings"]["tokens_saved"] < 5000:
            tips.append("📊 知识库不小了，多用 /左脑 query 查询就能省大量ToKen！")
        if ec > 0 and sc == 0:
            tips.append("🕸️ 纠缠场分析看到了关联词，再用 /左脑 search 图扩散能挖更深")

        # 如果没有特别建议
        if not tips:
            tips.append("✅ 当前使用状态很好，继续保持！试试 /左脑 dashboard 看详细数据")

        return tips[:5]  # 最多5条

    def reset(self):
        self.nodes = []
        self.hash_index = {}
        self.simhash_index = {}
        self._kw_index = {}
        self.bitmap = bytearray(self.bitmap_size // 8)
        if DATA_FILE.exists(): DATA_FILE.unlink()
        return {"status": "ok"}

    def _save(self):
        data = {
            "version": "3.10", "bitmap_size": self.bitmap_size,
            "hash_index": {str(k): v for k, v in self.hash_index.items()},
            "nodes": self.nodes, "updated_at": datetime.now().isoformat(),
            # 监测数据
            "learn_count": self.learn_count,
            "query_count": self.query_count,
            "search_count": self.search_count,
            "correct_count": self.correct_count,
            "analyze_count": self.analyze_count,
            "summarize_count": self.summarize_count,
            "entangle_count": self.entangle_count,
            "token_savings": self.token_savings,
            "first_use_at": self.first_use_at,
            "auto_mode": self.auto_mode,
            # 对话特征
            "short_q_count": self.short_q_count,
            "long_q_count": self.long_q_count,
            "medium_q_count": self.medium_q_count,
            "scene_counts": self.scene_counts,
            # 今日数据
            "_today": self._today,
            "today_learn": self.today_learn,
            "today_query": self.today_query,
            "today_search": self.today_search,
            "today_correct": self.today_correct,
            "today_analyze": self.today_analyze,
            "today_summarize": self.today_summarize,
            "today_entangle": self.today_entangle,
            "today_tokens": self.today_tokens,
            # 上下文增强
            "context_stack": self.context_stack.to_dict(),
            # 用户画像
            "user_profile": self.user_profile.to_dict(),
        }
        with open(str(DATA_FILE), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # ===== SQLite 增量双写 =====
        if _DB_AVAILABLE:
            try:
                self._sync_to_sqlite()
            except Exception:
                pass  # SQLite写入失败不影响主流程

        # v3.6.7: 每次数据变更时自动更新 MEMORY.md
        try:
            self._sync_memory_md()
        except Exception:
            pass

    def _sync_to_sqlite(self):
        """将关键计数器同步写入SQLite（增量）"""
        counter_map = {
            "learn_count": self.learn_count,
            "query_count": self.query_count,
            "search_count": self.search_count,
            "correct_count": self.correct_count,
            "analyze_count": self.analyze_count,
            "summarize_count": self.summarize_count,
            "entangle_count": self.entangle_count,
            "token_savings": self.token_savings,
        }
        if self.first_use_at:
            counter_map["first_use_at_str"] = 0  # 标记位，实际值用下面方式存

        for key, value in counter_map.items():
            old = counter_get(key)
            if value != old:
                # 全量覆盖（因为引擎内存中的值是全量的）
                conn = sqlite3.connect(str(DB_PATH), timeout=10)
                try:
                    conn.execute("""
                        INSERT INTO counters (key, value, updated_at)
                        VALUES (?, ?, datetime('now','localtime'))
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now','localtime')
                    """, (key, value))
                    conn.commit()
                finally:
                    conn.close()

        # 同步今日计数器
        today_map = {
            "today_learn": self.today_learn,
            "today_query": self.today_query,
            "today_search": self.today_search,
            "today_correct": self.today_correct,
            "today_analyze": self.today_analyze,
            "today_summarize": self.today_summarize,
            "today_entangle": self.today_entangle,
            "today_tokens": self.today_tokens,
        }
        today_str = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        try:
            for key, value in today_map.items():
                conn.execute("""
                    INSERT INTO today_counters (key, value, date)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, date = excluded.date
                """, (key, value, today_str))
            conn.commit()
        finally:
            conn.close()

        # 保存日级快照
        save_daily_snapshot(
            nodes_count=len(self.nodes),
            edges_count=sum(len(n.get("edges", [])) for n in self.nodes) // 2
        )

    def _apply_decay(self):
        """v3.9: 体裁感知记忆衰减 — 长期未访问的知识按体裁差异化降权"""
        now = time.time()
        cold_days = 30  # 30天未访问开始衰减
        max_strength = 200
        min_strength = 20

        # v3.9: 体裁差异化衰减率（每月衰减比例，值越大衰减越快）
        GENRE_DECAY_RATES = {
            'dialogue':      0.03,   # 对话类：3%/月，闲聊3个月衰减到~40%
            'essay':         0.03,   # 随笔类：3%/月
            'poem':          0.04,   # 诗歌类：4%/月
            'process':       0.02,   # 流程类：2%/月，步骤有长期价值
            'argument':      0.015,  # 论证类：1.5%/月
            'paper':         0.01,   # 论文类：1%/月
            'data_summary':  0.01,   # 数据类：1%/月，统计数据有长期参考价值
            'definition':    0.005,  # 定义类：0.5%/月，概念定义几乎不过时
            'knowledge':     0.01,   # 兜底：1%/月
        }
        DEFAULT_DECAY = 0.95  # 旧节点无genre标签时的衰减因子

        for node in self.nodes:
            if node is None: continue
            last = node.get("last_accessed", node.get("created_at", now))
            days_since = (now - last) / 86400
            if days_since > cold_days:
                months = max(0, (days_since - cold_days) / 30)
                current = node.get("strength", 100)

                # v3.9: 根据体裁选择衰减率
                genre = node.get("genre", "")
                decay_rate = 1.0 - GENRE_DECAY_RATES.get(genre, 0.05)
                # 旧节点无genre标签时用默认衰减因子
                if not genre:
                    decay_rate = DEFAULT_DECAY

                # 访问频率加成
                access = node.get("access_count", 0)
                boost = min(access * 2, 50)  # 高频访问加分
                new_strength = max(min_strength, int(current * (decay_rate ** months)) + boost)
                new_strength = min(max_strength, new_strength)
                node["strength"] = new_strength

    def _load(self):
        if not DATA_FILE.exists(): return
        try:
            with open(str(DATA_FILE), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # 数据文件损坏：备份后重置，避免状态不一致
            bak = DATA_FILE.with_suffix(".json.bak")
            try:
                DATA_FILE.rename(bak)
            except Exception:
                pass
            print(f"⚠️ 左脑数据文件损坏已备份至 {bak.name}，将使用空数据启动：{e}")
            return
        try:
            self.nodes = data.get("nodes", [])
            self.hash_index = {int(k): v for k, v in data.get("hash_index", {}).items()}
            # v3.1: 重建SimHash索引和关键词倒排索引
            self.simhash_index = {}
            self._kw_index = {}
            for i, node in enumerate(self.nodes):
                try:
                    # v3.8: 为旧节点补全新字段（兼容旧数据）
                    if node is not None:
                        node.setdefault("genre", "")
                        node.setdefault("skeleton", "")
                        node.setdefault("domain", "")
                        node.setdefault("source_context", "")
                        node.setdefault("source_turn_index", -1)
                    sh = self._simhash(node["text"])
                    self.simhash_index[sh] = i
                    self._keyword_index_add(node["text"], i)
                except Exception:
                    pass  # 损坏节点跳过
            # 恢复监测数据
            self.learn_count = data.get("learn_count", 0)
            self.query_count = data.get("query_count", 0)
            self.search_count = data.get("search_count", 0)
            self.correct_count = data.get("correct_count", 0)
            self.analyze_count = data.get("analyze_count", 0)
            self.summarize_count = data.get("summarize_count", 0)
            self.entangle_count = data.get("entangle_count", 0)
            self.token_savings = data.get("token_savings", 0)
            self.first_use_at = data.get("first_use_at")
            self.auto_mode = data.get("auto_mode", True)
            # 恢复对话特征
            self.short_q_count = data.get("short_q_count", 0)
            self.long_q_count = data.get("long_q_count", 0)
            self.medium_q_count = data.get("medium_q_count", 0)
            self.scene_counts = data.get("scene_counts", {})
            # 恢复今日数据
            self._today = data.get("_today", "")
            self.today_learn = data.get("today_learn", 0)
            self.today_query = data.get("today_query", 0)
            self.today_search = data.get("today_search", 0)
            self.today_correct = data.get("today_correct", 0)
            self.today_analyze = data.get("today_analyze", 0)
            self.today_summarize = data.get("today_summarize", 0)
            self.today_entangle = data.get("today_entangle", 0)
            self.today_tokens = data.get("today_tokens", 0)
            # 恢复上下文增强
            if "context_stack" in data:
                self.context_stack.from_dict(data["context_stack"])
            # 恢复用户画像
            if "user_profile" in data:
                self.user_profile.from_dict(data["user_profile"])
        except (KeyError, TypeError, ValueError) as e:
            # 字段缺失或类型错误：保留已读取的部分，打印警告
            print(f"⚠️ 左脑数据部分字段异常，已跳过：{e}")

        # v3.5: 加载后执行衰减（每次启动时触发一次）
        try:
            self._apply_decay()
        except Exception:
            pass

    # ===== 自动联动（和 WorkBuddy 无缝配合）=====

    def set_auto_mode(self, on: bool) -> Dict:
        """开启/关闭自动感知模式（增强版：原仅自动学习 → 自动感知+路由）"""
        self.auto_mode = on
        return {"status": "ok", "auto_mode": "已开启 ✅" if on else "已关闭 ❌",
                "tip": "对话内容将自动感知意图并智能路由（学习/查询/分析/推荐）" if on else "停止自动感知"}

    def auto_process(self, text: str) -> Dict:
        """一键自动处理：学习 + 分析 + 纠错 + 推荐关联"""
        results = {"learned": [], "corrected": None, "analyzed": None, "suggested": []}

        # 去重：和上次一样的内容跳过
        text = text.strip()
        if not text or text == self.last_auto_text:
            return results
        self.last_auto_text = text

        # 0. 记录对话特征（不存原文）
        self._check_today_reset()
        txt_len = len(text)
        if txt_len < 20:
            self.short_q_count += 1
        elif txt_len > 100:
            self.long_q_count += 1
        else:
            self.medium_q_count += 1
        scene = self.auto_category(text)
        self.scene_counts[scene] = self.scene_counts.get(scene, 0) + 1

        # 1. 自动纠错
        corrected = self.fix_typo(text)
        if corrected != text:
            results["corrected"] = {"original": text, "corrected": corrected}
            text = corrected  # 用纠错后的文本继续

        # 2. 自动学习（提取关键句）
        sents = [s.strip() for s in re.split(r'(?<=[。！？\n])\s*', text) if len(s.strip()) > 8]
        for sent in sents[:3]:
            if self.find_node(sent) is None:
                cat = self.auto_category(sent)
                idx = self.add_node(sent, cat)
                results["learned"].append({"text": sent, "category": cat})

        # 3. 自动检查有没有相关记忆可以推荐
        keywords = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        for kw in keywords[:5]:
            idx = self.find_node(kw)
            if idx is not None:
                node = self.nodes[idx]
                if node["edges"]:
                    related = [self.nodes[e[0]]["text"] for e in node["edges"][:3]]
                    results["suggested"].append({"keyword": kw, "related": related})

        self.correct_count += 1 if results["corrected"] else 0
        self.learn_count += len(results["learned"])
        if results["learned"]:
            self._record_today("learn", 200 * len(results["learned"]))
        return results

    def _extract_key_sentences(self, text: str) -> List[str]:
        """智能关键句提取 — 比正则切分更精准，按信息价值评分排序"""
        sents = [s.strip() for s in re.split(r'(?<=[。！？\n])\s*', text) if s.strip()]

        scored = []
        for sent in sents:
            if len(sent) < 8:
                continue
            score = 0
            # 包含数据 → 高价值
            if re.search(r'\d+[万亿千百]?[元人次条项个台套份]', sent):
                score += 3
            # 包含定义性表述 → 高价值
            if re.search(r'(是指|定义为|即|也就是|是指的)', sent):
                score += 3
            # 包含因果关系 → 高价值
            if re.search(r'(因为|所以|因此|导致|由于|结果)', sent):
                score += 2
            # 长度适中 → 高价值
            if 15 < len(sent) < 80:
                score += 1
            scored.append((sent, score))

        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:5]]

    def auto_process_v2(self, text: str, intent: str = "") -> Dict:
        """基于意图的自动处理 v2 — v3.9统一管线：一次分类返回所有维度"""
        # v3.9: 统一分类管线（GenreClassifier已合并IntentClassifier）
        scene = self.auto_category(text)
        _genre, _tags, _action_intent, _content_intent, _conf = GenreClassifier.classify_v2(text, scene=scene)
        # 用功能路由意图作为分路依据
        if not intent:
            intent = _action_intent

        results = {"status": "ok", "intent": intent, "learned": [], "corrected": None,
                   "queried": None, "analyzed": None, "summarized": None,
                   "suggested": [], "enhanced": None, "recommended": None}

        # 通用：自动纠错（所有意图都做）
        corrected = self.fix_typo(text)
        if corrected != text:
            results["corrected"] = {"original": text, "corrected": corrected}
            text = corrected

        # 更新上下文栈
        scene = self.auto_category(text)
        self.context_stack.push(text, scene)
        # v3.5: push 后持久化，新对话可恢复上下文
        if self.context_stack.goal_turn_count % 5 == 0:
            self._save()

        # 根据意图分路处理
        # v3.9: 统一管线已提前分类，直接使用_genre/_tags/_content_intent/_conf
        _domain = _tags[0] if _tags else ""
        _skeleton_obj = GenreClassifier.extract_skeleton(text, _genre)
        _skeleton_str = json.dumps(_skeleton_obj, ensure_ascii=False)[:500]
        _source_context = self._extract_source_context(text)
        self._turn_index += 1  # v3.8: 递增对话轮次

        if intent == "learn":
            # 提取关键句，智能学习
            sents = self._extract_key_sentences(text)
            if not sents:
                # fallback: 使用简单切分
                sents = [s.strip() for s in re.split(r'(?<=[。！？\n])\s*', text) if len(s.strip()) > 8][:3]
            for sent in sents[:5]:
                if self.find_node(sent) is None:
                    cat = self.auto_category(sent)
                    # v3.8: 带增强字段写入
                    idx = self.add_node(sent, cat,
                                        source=f"{datetime.now().strftime('%m-%d %H:%M')} · 自动感知",
                                        genre=_genre, skeleton=_skeleton_str,
                                        domain=_domain, source_context=_source_context,
                                        source_turn_index=self._turn_index)
                    results["learned"].append({"text": sent, "category": cat, "genre": _genre})

        elif intent == "query":
            # 自动查询 + 图扩散搜索
            keywords = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
            for kw in keywords[:3]:
                idx = self.find_node(kw)
                if idx is not None:
                    diffusion = self.query_diffusion(kw, max_hops=2)
                    results["queried"] = {"keyword": kw, "diffusion": diffusion}
                    break
            # v3.4: 查询中也可能包含值得记住的新信息
            sents = [s.strip() for s in re.split(r'(?<=[。！？\n])\s*', text) if len(s.strip()) > 15]
            for sent in sents[:2]:
                if self.find_node(sent) is None:
                    cat = self.auto_category(sent)
                    # v3.8: 带增强字段写入
                    self.add_node(sent, cat, genre=_genre, skeleton=_skeleton_str,
                                  domain=_domain, source_context=_source_context,
                                  source_turn_index=self._turn_index)
                    results["learned"].append({"text": sent, "category": cat})

        elif intent == "analyze":
            # 数据分析
            results["analyzed"] = DataAnalyzer.analyze(text)
            self.analyze_count += 1
            self._record_today("analyze")

        elif intent == "summarize":
            # 文章总结
            results["summarized"] = Summarizer.summarize(text)
            self.summarize_count += 1
            self._record_today("summarize")

        elif intent == "chat":
            # v3.4: 全自动记忆 — 聊天中自动提取有价值的信息学习
            # 不再只学1句，而是智能评估每句话的信息密度
            sents = [s.strip() for s in re.split(r'(?<=[。！？\n])\s*', text) if s.strip()]
            learned_count = 0
            for sent in sents:
                if len(sent) < 8 or learned_count >= 5:
                    continue
                # 评估信息价值：包含数字/日期/名称/定义的句子优先学习
                has_fact = bool(re.search(r'\d{2,}|[\u4e00-\u9fff]{3,}是|叫做|称为|指|即', sent))
                min_len = 10 if has_fact else 20  # 有事实的句子降低门槛
                if len(sent) < min_len:
                    continue
                if self.find_node(sent) is None:
                    cat = self.auto_category(sent)
                    # v3.9: 对每句分别做体裁检测（更精细），使用统一管线
                    sent_genre, sent_tags, _, sent_content_intent, sent_conf = GenreClassifier.classify_v2(sent)
                    sent_skeleton = json.dumps(GenreClassifier.extract_skeleton(sent, sent_genre), ensure_ascii=False)[:500]
                    self.add_node(sent, cat,
                                  genre=sent_genre, skeleton=sent_skeleton,
                                  domain=sent_tags[0] if sent_tags else "",
                                  source_context=_source_context,
                                  source_turn_index=self._turn_index)
                    results["learned"].append({"text": sent, "category": cat, "cat": cat, "genre": sent_genre})
                    learned_count += 1
            
            # v3.4: 同轮对话学到的知识自动建边（聚类）
            learned_items = results.get("learned", [])
            if len(learned_items) >= 2:
                for i in range(len(learned_items)):
                    for j in range(i+1, len(learned_items)):
                        try:
                            a_idx = self.find_node(learned_items[i].get("text",""))
                            b_idx = self.find_node(learned_items[j].get("text",""))
                            if a_idx is not None and b_idx is not None and a_idx != b_idx:
                                # 使用场景作为关系名
                                rel = learned_items[i].get("cat", scene) or "相关"
                                self.add_edge(a_idx, b_idx, rel)
                        except Exception:
                            pass
            
            # v3.5: 自动学习的知识加入待确认队列
            self._pending_queue = getattr(self, '_pending_queue', [])
            for item in results.get("learned", []):
                if len(self._pending_queue) < 20:  # 最多积压 20 条
                    self._pending_queue.append({"text": item.get("text",""), "category": item.get("cat", scene)})

        # 记录对话特征
        self._record_conversation_features(text)

        # 🆕 v3.1: 实时Token消耗记录到平台数据库（ToKen监测助手可实时展示）
        self._record_platform_usage(text, intent, results)

        # v3.4: 生成感知摘要，让用户知道左脑做了什么
        parts = []
        if results.get("learned"):
            parts.append(f"学习了{len(results['learned'])}条知识")
        if results.get("corrected") and results["corrected"]["original"] != results["corrected"]["corrected"]:
            parts.append(f"纠错:{results['corrected']['original'][:15]}→{results['corrected']['corrected'][:15]}")
        if results.get("queried"):
            parts.append(f"找到了相关知识")
        if parts:
            results["_perception_summary"] = "🧠 " + " | ".join(parts)

        # v3.8: KAR融合 — 自动写入三层上下文
        try:
            self._write_context_layers(text, intent, results)
        except Exception:
            pass

        return results

    # ===== v3.8: 方案B — 知识节点+对话切片 =====

    def _extract_source_context(self, current_text: str, window_before: int = 2, window_after: int = 0) -> str:
        """截取当前对话轮次前后各window轮的原文，作为source_context

        规则：
          - 默认前后各2轮（before=2, after=0 因为after还没发生）
          - 有因果链扩展到5轮
          - 单条≤2000字，超出截断保留因果链
          - 格式："用户：XXX\nAI：XXX\n..."
        """
        stack = self.context_stack.stack
        if not stack:
            return ""

        # 检测是否有因果链（需要更大窗口）
        has_causal = bool(re.search(r'(因为|所以|因此|导致|由于|从而|如果.*就)', current_text))
        if has_causal:
            window_before = min(5, len(stack))

        # 取最近window_before轮
        start_idx = max(0, len(stack) - window_before)
        context_parts = []
        for i in range(start_idx, len(stack)):
            entry = stack[i]
            keywords = entry.get("keywords", [])
            scene = entry.get("scene", "")
            context_parts.append(f"用户：{' '.join(keywords[:5])} [{scene}]")

        # 当前轮
        context_parts.append(f"用户：{current_text[:200]}")

        result = '\n'.join(context_parts)
        # 截断到2000字
        if len(result) > self._context_memory.MAX_SOURCE_CONTEXT:
            result = result[:self._context_memory.MAX_SOURCE_CONTEXT]

        return result

    def _write_context_layers(self, text: str, intent: str, results: dict):
        """v3.8: 写入三层上下文（短程/中程/长程）"""
        # 短程：保存到session
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        entities = keywords[:10]
        injection_parts = []
        if results.get("learned"):
            for item in results["learned"][:3]:
                injection_parts.append(f"学习了: {item.get('text', '')[:80]}")
        if results.get("queried"):
            injection_parts.append(f"查询到: {results['queried'].get('keyword', '')}")
        injection = '\n'.join(injection_parts)

        self._context_memory.save_session(
            text[:50],
            {
                "entities": entities,
                "injection": injection,
                "intent": intent,
            }
        )

        # 中程+长程：主题合并
        if keywords:
            self._context_memory.merge_topic(
                text[:30],
                {
                    "entities": entities,
                    "injection": injection,
                }
            )

    def trace_source_context(self, keyword: str) -> Dict:
        """v3.9: 按需回溯source_context — 追溯知识来源 + 因果链可视化 + 时间线重建

        触发条件：
        1. 用户追问"为什么"/"当时怎么说的"
        2. /左脑 追溯 <关键词>
        3. 同主题多次查询
        """
        # 1. 先在知识节点中查找（v3.9: 启用体裁感知检索）
        idx = self.find_node(keyword, genre_aware=True)
        node_source_context = ""
        node_genre = ""
        node_skeleton = ""
        node_learned_at = ""
        node_session_id = ""
        node_turn_index = -1

        if idx is not None:
            node = self.nodes[idx]
            node_source_context = node.get("source_context", "")
            node_genre = node.get("genre", "")
            node_skeleton = node.get("skeleton", "")
            node_learned_at = node.get("learned_at", "")
            node_session_id = node.get("session_id", "")
            node_turn_index = node.get("source_turn_index", -1)

        # 2. 查三层上下文（v3.9: 已升级为SimHash语义匹配）
        last_ctx = self._context_memory.get_last_context(keyword)
        inherited = self._context_memory.inherit_context(keyword, last_ctx) if last_ctx else ""
        merged = self._context_memory.get_merged(keyword)

        # 3. v3.9: 因果链可视化 — 从source_context中提取因果链
        causal_links = []
        if node_source_context:
            causal_links = KnowledgeInferrer.extract_causal_links(node_source_context)
        # 也从相关节点文本中提取
        if not causal_links and idx is not None:
            causal_links = KnowledgeInferrer.extract_causal_links(self.nodes[idx]["text"])

        # 4. v3.9: 时间线重建 — 从上下文栈中恢复对话切片
        timeline = self._rebuild_timeline(keyword)

        # 5. 组合输出
        parts = []
        if node_source_context:
            parts.append(f"【知识来源对话】")
            parts.append(node_source_context)
        if node_genre:
            parts.append(f"\n【体裁】{node_genre}")
        if node_skeleton:
            try:
                sk = json.loads(node_skeleton) if isinstance(node_skeleton, str) else node_skeleton
                parts.append(f"【推理骨架】{json.dumps(sk, ensure_ascii=False)[:300]}")
            except Exception:
                pass
        if causal_links:
            parts.append(f"\n【因果链】")
            for link in causal_links:
                parts.append(f"  {link['cause']} → {link['effect']}")
        if timeline:
            parts.append(f"\n【时间线】")
            for entry in timeline:
                parts.append(f"  轮次{entry['turn']}: {entry['text'][:60]}")
        if inherited:
            parts.append(f"\n{inherited}")
        if merged:
            parts.append(f"\n{merged}")

        return {
            "status": "ok",
            "keyword": keyword,
            "has_source": bool(node_source_context or inherited or merged),
            "trace": '\n'.join(parts) if parts else "未找到追溯信息",
            "genre": node_genre,
            "has_context_layer": bool(last_ctx),
            "causal_links": causal_links,       # v3.9: 前端可渲染 A→B→C
            "timeline": timeline,                # v3.9: 时间线重建
            "learned_at": node_learned_at,
            "session_id": node_session_id,
            "turn_index": node_turn_index,
        }

    def _rebuild_timeline(self, keyword: str) -> list:
        """v3.9: 从上下文栈和知识节点中重建对话时间线"""
        timeline = []
        q_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', keyword))

        # 从上下文栈中提取相关轮次
        for i, entry in enumerate(self.context_stack.stack):
            kw_list = entry.get("keywords", [])
            scene = entry.get("scene", "")
            # 检查关键词重叠
            entry_words = set(kw_list) if isinstance(kw_list, list) else set()
            if q_words & entry_words or any(w in keyword for w in kw_list if isinstance(w, str)):
                timeline.append({
                    "turn": i + 1,
                    "text": ' '.join(kw_list[:5]) if isinstance(kw_list, list) else str(kw_list)[:60],
                    "scene": scene,
                    "source": "context_stack",
                })

        # 从知识节点中补充时间信息
        for node in self.nodes:
            if node is None: continue
            node_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', node["text"]))
            if q_words & node_words:
                timeline.append({
                    "turn": node.get("source_turn_index", -1),
                    "text": node["text"][:60],
                    "scene": node.get("category", ""),
                    "source": "knowledge_node",
                    "learned_at": node.get("learned_at", ""),
                })

        # 按轮次排序
        timeline.sort(key=lambda x: x.get("turn", 999))
        return timeline[:10]

    def _record_conversation_features(self, text: str):
        """记录对话特征（不存原文）"""
        self._check_today_reset()
        txt_len = len(text)
        if txt_len < 20:
            self.short_q_count += 1
            if _DB_AVAILABLE:
                feature_incr("short_q")
        elif txt_len > 100:
            self.long_q_count += 1
            if _DB_AVAILABLE:
                feature_incr("long_q")
        else:
            self.medium_q_count += 1
            if _DB_AVAILABLE:
                feature_incr("medium_q")
        scene = self.auto_category(text)
        self.scene_counts[scene] = self.scene_counts.get(scene, 0) + 1
        if _DB_AVAILABLE:
            db_scene_incr(scene)

    def _record_platform_usage(self, text: str, intent: str, results: Dict):
        """v3.1: 将每次感知操作的Token消耗记录到平台数据库，ToKen监测助手实时可查
        
        估算逻辑：
        - 用户输入：按字符数估算（中文≈1.5 token/字，英文≈0.25 token/word）
        - 左脑节省：查询命中/纠错/图扩散/分析等操作节省的重复提问Token
        - 操作类型映射：intent → op_type
        """
        if not _DB_AVAILABLE:
            return
        
        # 估算输入Token
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_words = len(re.findall(r'[a-zA-Z]+', text))
        input_tokens = int(cn_chars * 1.5 + en_words * 1.3)
        
        # 估算左脑节省的Token
        saved_tokens = 0
        # 查询命中 → 避免重复向大模型提问
        if results.get("queried"):
            saved_tokens += 800  # 一次图扩散查询节省约800 token
        # 纠错 → 避免大模型纠错请求
        if results.get("corrected"):
            saved_tokens += 200
        # 学习 → 知识已存储，后续查询无需重复学习
        learned_count = len(results.get("learned", []))
        if learned_count > 0:
            saved_tokens += learned_count * 150
        # 分析 → 避免大模型做数据分析
        if results.get("analyzed"):
            saved_tokens += 500
        # 总结 → 避免大模型做总结
        if results.get("summarized"):
            saved_tokens += 600
        # 推荐 → 主动推荐关联知识
        recommended = results.get("recommended") or results.get("suggested", [])
        if recommended:
            saved_tokens += len(recommended) * 100
        
        # 更新内部Token节省计数
        self.token_savings += saved_tokens
        
        # 意图 → 操作类型映射
        op_type_map = {
            "learn": "learn",
            "query": "query", 
            "analyze": "analysis",
            "summarize": "summarize",
            "chat": "conversation",
            "command": "command",
        }
        op_type = op_type_map.get(intent, "perceive")
        
        try:
            record_platform_usage(
                platform="workbuddy",
                model="zuonao-perceive",
                input_tokens=input_tokens,
                output_tokens=0,  # 感知模式不直接输出token
                op_type=op_type,
                saved_tokens=saved_tokens,
            )
        except Exception:
            pass  # 不阻塞主流程

    def _check_today_reset(self):
        """检查是否跨天，如果是则重置今日计数"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            self.today_learn = 0
            self.today_query = 0
            self.today_search = 0
            self.today_correct = 0
            self.today_analyze = 0
            self.today_summarize = 0
            self.today_entangle = 0
            self.today_tokens = 0

    def _record_today(self, action: str, tokens: int = 0):
        """记录一次操作到今日计数"""
        self._check_today_reset()
        if action == "learn": self.today_learn += 1
        elif action == "query": self.today_query += 1
        elif action == "search": self.today_search += 1
        elif action == "correct": self.today_correct += 1
        elif action == "analyze": self.today_analyze += 1
        elif action == "summarize": self.today_summarize += 1
        elif action == "entangle": self.today_entangle += 1
        self.today_tokens += tokens
        # SQLite增量写入
        if _DB_AVAILABLE:
            today_incr(f"today_{action}", 1)
            if tokens > 0:
                today_incr("today_tokens", tokens)
            hourly_incr(tokens)

    def suggest(self, text: str, max_items: int = 5) -> Dict:
        """根据输入自动推荐相关知识（精确匹配 + 模糊搜索 fallback）"""
        keywords = list(set(re.findall(r'[\u4e00-\u9fff]{2,6}', text)))
        hits = []
        seen_idx = set()
        for kw in keywords:
            # 精确匹配
            idx = self.find_node(kw)
            if idx is not None and idx not in seen_idx:
                node = self.nodes[idx]
                hits.append({"keyword": kw, "match": node["text"],
                             "category": node["category"],
                             "edges": len(node["edges"])})
                seen_idx.add(idx)
            else:
                # 模糊搜索 fallback
                fuzzy = self.fuzzy_find(kw, top_k=2)
                for fi, score in fuzzy:
                    if fi not in seen_idx:
                        node = self.nodes[fi]
                        hits.append({"keyword": kw, "match": node["text"],
                                     "category": node["category"],
                                     "edges": len(node["edges"]),
                                     "fuzzy": True})
                        seen_idx.add(fi)
        # 按关联边数排序
        hits.sort(key=lambda x: -x["edges"])
        return {
            "status": "ok",
            "keywords_checked": len(keywords),
            "matches_found": len(hits),
            "related": hits[:max_items],
            "results": hits[:max_items],
            "tip": "试试 /左脑 search <关键词> 查看关联网络" if hits else "暂无匹配记忆",
        }

    def enhance(self, text: str = "", context_window: int = 3) -> Dict:
        """上下文增强 — 三路检索+合并+注入，为当前对话补充相关知识"""
        # 1. 更新上下文栈
        if text:
            scene = self.auto_category(text)
            self.context_stack.push(text, scene)
            # v3.5: 每 5 轮持久化上下文
            if self.context_stack.goal_turn_count % 5 == 0:
                self._save()

        # 2. 获取上下文关键词
        ctx_keywords = self.context_stack.get_context_keywords(top_k=5)

        if not ctx_keywords:
            return {"status": "no_context", "injections": [], "injection_text": "暂无上下文信息"}

        # 3. 三路并行检索
        # 路径1: suggest
        suggest_kw = [kw for kw, _ in ctx_keywords[:3]]
        suggest_results = self.suggest(" ".join(suggest_kw))

        # 路径2: 图扩散（精确+模糊）
        diffusion_results = {"results": []}
        for kw, weight in ctx_keywords[:2]:
            if weight > 0.3:
                diff = self.query_diffusion(kw, max_hops=2)
                if diff.get("details"):
                    for item in diff["details"][:5]:
                        fi = self.find_node(item["text"])
                        cat = self.nodes[fi]["category"] if fi is not None else "通用"
                        diffusion_results["results"].append({
                            "text": item["text"],
                            "category": cat,
                            "hops": item.get("hop", 1),
                        })
                else:
                    # 精确匹配失败 → 模糊搜索
                    fuzzy = self.fuzzy_find(kw, top_k=3)
                    for fi, score in fuzzy:
                        node = self.nodes[fi]
                        diffusion_results["results"].append({
                            "text": node["text"],
                            "category": node["category"],
                            "hops": 1,
                            "fuzzy": True,
                        })

        # 路径3: 纠缠场
        entangle_results = {"injection": "", "top_pairs": []}
        eb = get_entanglement()
        if eb:
            for kw, _ in ctx_keywords[:2]:
                inj = eb.generate_injection(kw)
                if inj.get("network"):
                    for word, strength in inj["network"].items():
                        entangle_results["top_pairs"].append({
                            "word_a": kw,
                            "word_b": word,
                            "strength": strength,
                        })

        # 4. 合并
        merged = ContextMerger.merge(suggest_results, diffusion_results,
                                      entangle_results, self.context_stack)

        # 5. 格式化注入文本
        injection_text = ""
        for item in merged["injections"]:
            injection_text += f"[{item['source']}] {item['text']} (相关度{item['score']:.1f})\n"

        return {
            "status": "ok",
            "injections": merged["injections"],
            "injection_text": injection_text.strip(),
            "context_keywords": [(kw, f"{w:.2f}") for kw, w in ctx_keywords],
            "total_candidates": merged.get("total_candidates", 0),
        }

    def recommend(self, context_text: str = "", mode: str = "auto") -> Dict:
        """智能推荐 — 三路推荐源 + 排序 + 过滤"""
        # 1. 构建用户画像
        self.user_profile.build()

        # 2. 确定推荐种子
        if context_text:
            seed_keywords = re.findall(r'[\u4e00-\u9fff]{2,6}', context_text)[:5]
        else:
            # 无上下文时，从高频节点取种子
            top_nodes = sorted(range(len(self.nodes)),
                               key=lambda i: len(self.nodes[i]["edges"]), reverse=True)[:3]
            seed_keywords = [self.nodes[i]["text"][:6] for i in top_nodes if i < len(self.nodes)]

        if not seed_keywords:
            return {"status": "no_data", "recommendations": [],
                    "message": "知识库为空，先学习一些知识吧"}

        candidates = []

        # 来源1: 图扩散推荐（精确+模糊）
        for kw in seed_keywords[:2]:
            idx = self.find_node(kw)
            if idx is not None:
                diffusion = self.query_diffusion(kw, max_hops=2)
                for item in diffusion.get("details", [])[:5]:
                    fi = self.find_node(item["text"])
                    cat = self.nodes[fi]["category"] if fi is not None else "通用"
                    candidates.append({
                        "text": item["text"],
                        "category": cat,
                        "diffusion_score": 1.0 / max(item.get("hop", 1), 1),
                        "entangle_score": 0,
                        "created_ts": time.time(),
                    })
            else:
                # 精确匹配失败 → 模糊搜索
                fuzzy = self.fuzzy_find(kw, top_k=3)
                for fi, score in fuzzy:
                    node = self.nodes[fi]
                    # 从模糊匹配节点出发做图扩散
                    diffusion = self.query_diffusion(node["text"], max_hops=1)
                    if diffusion.get("details"):
                        for item in diffusion["details"][:3]:
                            candidates.append({
                                "text": item["text"],
                                "category": self.nodes[item.get("path_idx", fi)]["category"] if item.get("path_idx", fi) < len(self.nodes) else "通用",
                                "diffusion_score": 1.0 / max(item.get("hop", 1), 1),
                                "entangle_score": 0,
                                "created_ts": time.time(),
                            })
                    else:
                        candidates.append({
                            "text": node["text"],
                            "category": node["category"],
                            "diffusion_score": score * 0.5,
                            "entangle_score": 0,
                            "created_ts": time.time(),
                            "fuzzy": True,
                        })

        # 来源2: 纠缠场推荐
        eb = get_entanglement()
        if eb:
            for kw in seed_keywords[:2]:
                entangled = eb.get_entangled(kw)
                for word, strength in entangled[:5]:
                    candidates.append({
                        "text": f"{kw} → {word} (强度{strength:.2f})",
                        "category": "纠缠",
                        "diffusion_score": 0,
                        "entangle_score": strength,
                        "created_ts": time.time(),
                    })

        # 来源3: 冷启动推荐（知识库节点 < 10 时）
        if len(self.nodes) < 10:
            for kw in seed_keywords[:2]:
                idx = self.find_node(kw)
                if idx is None:
                    candidates.append({
                        "text": f"建议学习「{kw}」相关知识（知识库中尚未记录）",
                        "category": "冷启动",
                        "diffusion_score": 0.1,
                        "entangle_score": 0,
                        "created_ts": time.time(),
                    })

        # 3. 排序
        ranked = RecommendRanker.rank(candidates, self.user_profile)

        # 4. 格式化输出
        recommendations = []
        for item in ranked:
            if item.get("diffusion_score", 0) > 0:
                reason = f"关联度{item['diffusion_score']:.0%}"
            elif item.get("entangle_score", 0) > 0:
                reason = f"纠缠强度{item['entangle_score']:.2f}"
            else:
                reason = f"匹配你{item.get('category', '')}领域偏好"
            recommendations.append({
                "text": item["text"],
                "reason": reason,
                "score": item.get("final_score", 0),
            })

        return {
            "status": "ok",
            "recommendations": recommendations,
            "profile_type": self.user_profile.depth_type,
            "top_interests": sorted(self.user_profile.interest_vector.items(),
                                    key=lambda x: -x[1])[:3],
        }


class ContextMerger:
    """上下文合并器 — 三路检索结果的去重、排序、截断"""

    MAX_INJECTIONS = 5

    @staticmethod
    def merge(suggest_results: Dict, diffusion_results: Dict,
              entangle_results: Dict, context_stack: ContextStack) -> Dict:
        """
        三路合并逻辑：
        1. 收集所有候选知识条目
        2. 对每条计算综合评分
        3. 去重（前20字符哈希）
        4. 多样性控制（同领域最多2条）
        5. 截断到 MAX_INJECTIONS 条
        """
        candidates = []

        # 来源1: suggest 结果 (权重 0.3)
        if suggest_results.get("related"):
            for item in suggest_results["related"]:
                candidates.append({
                    "text": item.get("match", item.get("keyword", "")),
                    "source": "suggest",
                    "score": item.get("edges", 1) * 0.3,
                    "category": item.get("category", "通用"),
                })

        # 来源2: 图扩散结果 (权重 0.4)
        if diffusion_results.get("results"):
            for item in diffusion_results["results"][:5]:
                candidates.append({
                    "text": item["text"],
                    "source": "diffusion",
                    "score": 1.0 / max(item.get("hops", 1), 1) * 0.4,
                    "category": item.get("category", "通用"),
                })

        # 来源3: 纠缠场结果 (权重 0.3)
        if entangle_results.get("top_pairs"):
            for item in entangle_results["top_pairs"][:5]:
                candidates.append({
                    "text": f"{item['word_a']} ↔ {item['word_b']}",
                    "source": "entangle",
                    "score": item.get("strength", 0) * 0.3,
                    "category": "纠缠",
                })

        # 排序
        candidates.sort(key=lambda x: -x["score"])

        # 去重
        seen = set()
        unique = []
        for c in candidates:
            key = c["text"][:20]
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # 多样性控制
        category_count = {}
        final = []
        for c in unique:
            cat = c["category"]
            category_count[cat] = category_count.get(cat, 0) + 1
            if category_count[cat] <= 2:
                final.append(c)

        return {
            "injections": final[:ContextMerger.MAX_INJECTIONS],
            "total_candidates": len(candidates),
            "after_dedup": len(unique),
        }


class RecommendRanker:
    """推荐排序器 — 多源推荐结果的评分、去重、多样性控制"""

    # 评分权重
    W_DIFFUSION = 0.4
    W_ENTANGLE = 0.3
    W_PROFILE = 0.2
    W_FRESHNESS = 0.1

    MAX_RECOMMENDATIONS = 5
    MAX_SAME_CATEGORY = 2

    @staticmethod
    def rank(candidates: List[Dict], profile: UserProfile) -> List[Dict]:
        """
        综合排序：
        1. 计算每条候选的综合评分
        2. 已推荐过滤
        3. 去重
        4. 多样性控制
        5. 截断
        """
        scored = []
        now = time.time()

        for c in candidates:
            # 已推荐过滤
            if profile.is_recently_recommended(c.get("text", "")):
                continue

            # 综合评分
            score = (
                RecommendRanker.W_DIFFUSION * c.get("diffusion_score", 0) +
                RecommendRanker.W_ENTANGLE * c.get("entangle_score", 0) +
                RecommendRanker.W_PROFILE * profile.match_score(c.get("category", "通用")) +
                RecommendRanker.W_FRESHNESS * min(1.0, (now - c.get("created_ts", now)) / 86400)
            )

            c["final_score"] = round(score, 3)
            scored.append(c)

        # 排序
        scored.sort(key=lambda x: -x["final_score"])

        # 多样性控制
        category_count = {}
        final = []
        for c in scored:
            cat = c.get("category", "通用")
            category_count[cat] = category_count.get(cat, 0) + 1
            if category_count[cat] <= RecommendRanker.MAX_SAME_CATEGORY:
                final.append(c)
                profile.mark_recommended(c.get("text", ""))

        return final[:RecommendRanker.MAX_RECOMMENDATIONS]


# ===================================================================
# 第二部分：深度推理引擎（数据分析/文章总结/因果推论/推理指令）v3.8
# ===================================================================

def _clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def _split_sentences(text):
    sents = re.split(r'(?<=[。！？\n])\s*', text)
    return [s.strip() for s in sents if len(s.strip()) > 5]


class DataAnalyzer:
    """数据分析引擎 v3.8 — 数字提取、趋势分析、对比提取、统计信息
    增强版：融合KAR deep_reason.py的DataAnalyzer，更精准的数字提取和趋势分析
    """

    NUM_PAT = re.compile(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|[\d]+\.[\d]+%?|[零一二三四五六七八九十百千万亿]+)')
    UNITS_PATTERN = r'[年月中天日小时分秒件个人次元种项个条只%倍百分点](?:元|块|万|亿)?'
    TREND_UP = r'(增长|上升|提升|增加|上涨|提高|扩大|增量)'
    TREND_DOWN = r'(下降|降低|减少|下跌|下滑|缩减|减量|降低)'
    TREND_STABLE = r'(持平|不变|稳|保持|稳定)'
    COMPARE = r'(高于|低于|大于|小于|超过|不及|多于|少于|优于|劣于)'

    @classmethod
    def analyze(cls, text):
        text = _clean_text(text)
        numbers = cls._extract_numbers(text)
        trends = cls._extract_trends(text)
        comparisons = cls._extract_comparisons(text)
        stats = cls._extract_stats(text)

        parts = []
        if numbers:
            parts.append(f"【数据点】发现 {len(numbers)} 个数值")
            for n in numbers[:5]:
                parts.append(f"  · {n['value']} ({n['context'][:40]})")
        if trends:
            parts.append(f"\n【趋势】发现 {len(trends)} 个变化")
            for t in trends[:3]:
                arrow = '↑' if t['direction'] == '上升' else ('↓' if t['direction'] == '下降' else '→')
                parts.append(f"  {arrow} {t['subject']} {t['direction']}({t['change']})" if t['change'] else f"  {arrow} {t['subject']} {t['direction']}")
        if comparisons:
            parts.append(f"\n【对比】发现 {len(comparisons)} 个对比")
            for c in comparisons[:3]:
                parts.append(f"  · {c['subject']} {c['relation']} {c['target']}")
        if stats:
            parts.append(f"\n【统计】发现 {len(stats)} 个统计信息")
            for s in stats[:3]:
                parts.append(f"  · [{s['type']}] {', '.join(s['values'][:3])}")

        return {
            "has_data": bool(numbers or trends or comparisons or stats),
            "numbers_count": len(numbers), "trends_count": len(trends),
            "comparisons_count": len(comparisons), "stats_count": len(stats),
            "summary": '\n'.join(parts) if parts else "未发现数据",
        }

    @classmethod
    def _extract_numbers(cls, text):
        results = []
        # 匹配数字，然后尝试捕获后面的中文单位（1-3个汉字）
        UNIT_SUFFIXES = r'(?:[年月中天日小时分秒件个人次元种项个条只%倍百分点万亿](?:元|块|万|亿)?)?'
        for sent in _split_sentences(text):
            for m in re.finditer(r'(\d+[\d,\.]*)(\s*' + UNIT_SUFFIXES + r')', sent):
                num_val = m.group(1)
                unit = m.group(2).strip() if m.group(2) else ""
                # 额外检查：数字后直接跟 "万元" "亿元" 等复合单位
                if not unit:
                    m2 = re.match(r'(\d+[\d,\.]*)(万?元|亿?元|万元|亿元|万人次|人次)', sent[m.start():m.start()+20])
                    if m2 and m2.group(1) == num_val:
                        unit = m2.group(2)
                display = f"{num_val}{unit}" if unit else num_val
                results.append({'value': display, 'raw_number': num_val, 'unit': unit, 'context': sent[:80]})
        seen = set()
        unique = []
        for r in results:
            key = f"{r['value']}|{r['context'][:30]}"
            if key not in seen: seen.add(key); unique.append(r)
        return unique[:20]

    @classmethod
    def _extract_trends(cls, text):
        trends = []
        for sent in _split_sentences(text):
            direction = None
            if re.search(cls.TREND_UP, sent): direction = '上升'
            elif re.search(cls.TREND_DOWN, sent): direction = '下降'
            elif re.search(cls.TREND_STABLE, sent): direction = '持平'
            if direction:
                subject = ''
                for pat in [r'(.)(?:的)?(?:增长|下降|提升|降低)', r'(.)(?:呈|保持)(?:现出)?']:
                    m = re.search(pat, sent)
                    if m: subject = m.group(1)[-5:]
                change = ''
                m = re.search(r'(\d+[\d,\.]*%?)', sent)
                if m: change = m.group(1)
                trends.append({'direction': direction, 'subject': subject, 'change': change})
        return trends[:10]

    @classmethod
    def _extract_comparisons(cls, text):
        comparisons = []
        for sent in _split_sentences(text):
            m = re.search(r'([\u4e00-\u9fff\w]{2,20})(' + cls.COMPARE.replace('(', '').replace(')', '') + r')([\u4e00-\u9fff\w\d.%]+)', sent)
            if m: comparisons.append({'subject': m.group(1), 'relation': m.group(2), 'target': m.group(3)[:30]})
        return comparisons[:10]

    @classmethod
    def _extract_stats(cls, text):
        stats = []
        for sent in _split_sentences(text):
            stat_type = ''
            if re.search(r'(平均|总计|合计|累计|总和)', sent): stat_type = '总计'
            elif re.search(r'(占比|比例|率|百分比)', sent): stat_type = '占比'
            elif re.search(r'(最高|最低|最大|最小|极值)', sent): stat_type = '极值'
            elif re.search(r'(分布|排序|排名|排行)', sent): stat_type = '分布'
            else: continue
            nums = re.findall(r'\d+[\d,\.]*%?', sent)
            stats.append({'type': stat_type, 'values': nums[:5]})
        return stats[:10]


class Summarizer:
    """文章总结引擎 v3.8 — 增强版：融合KAR章节提取+关键点分类
    新增：章节结构提取、关键点按类型分组（结论/核心/论据/数据/例子）
    """

    @classmethod
    def summarize(cls, text, max_sections=5):
        text = _clean_text(text)
        char_count = len(text)
        word_count = len(re.findall(r'[\u4e00-\u9fff]+', text))
        sentences = _split_sentences(text)

        # v3.8: 提取章节结构
        sections = cls._extract_sections(text)

        # v3.8: 增强关键点提取（来自KAR Summarizer）
        key_points = cls._extract_key_points(text)

        # 按类型分组
        by_type = defaultdict(list)
        for p in key_points:
            by_type[p['type']].append(p['text'])

        parts = []
        parts.append(f"【文章基本信息】总字数: {char_count} | 中文词: {word_count} | 句子数: {len(sentences)}")
        if sections:
            parts.append(f"\n【章节结构】({len(sections)}段)")
            for sec in sections[:max_sections]:
                preview = sec['content'][:60].replace('\n', ' ')
                if preview:
                    parts.append(f"  · {sec['title']}: {preview}...")
        if by_type.get('结论'):
            parts.append("\n【核心结论】")
            for c in by_type['结论'][:3]: parts.append(f"  → {c}")
        if by_type.get('核心'):
            parts.append("\n【核心论点】")
            for c in by_type['核心'][:3]: parts.append(f"  · {c}")
        if by_type.get('论据'):
            parts.append("\n【论据】")
            for a in by_type['论据'][:5]: parts.append(f"  · {a}")
        if by_type.get('数据'):
            parts.append("\n【数据支撑】")
            for d in by_type['数据'][:3]: parts.append(f"  · {d}")

        return {
            "char_count": char_count, "word_count": word_count,
            "sentences": len(sentences),
            "sections": sections[:max_sections],
            "key_points": key_points,
            "by_type": {k: v[:5] for k, v in by_type.items()},
            "summary": '\n'.join(parts),
        }

    @classmethod
    def _extract_sections(cls, text):
        """v3.8: 提取文章章节结构"""
        sections = []
        lines = text.split('\n')
        current_section = '开头'
        current_content = []
        for line in lines:
            if re.match(r'^#{1,3}\s', line):
                if current_content:
                    sections.append({'title': current_section,
                                     'content': '\n'.join(current_content).strip()})
                current_section = re.sub(r'^#+\s*', '', line)[:30]
                current_content = []
            else:
                current_content.append(line)
        if current_content:
            sections.append({'title': current_section,
                             'content': '\n'.join(current_content).strip()})
        return sections

    @classmethod
    def _extract_key_points(cls, text):
        """v3.8: 从段落中提取核心论点（增强版）"""
        sentences = _split_sentences(text)
        points = []
        for sent in sentences:
            is_key = False
            point_type = '信息'
            if re.search(r'(总之|综上所述|因此|所以|可见|结论是)', sent):
                is_key = True; point_type = '结论'
            elif re.search(r'(重要的是|关键|核心|本质|实质)', sent):
                is_key = True; point_type = '核心'
            elif re.search(r'(第一|第二|第三|首先|其次|最后|一方面|另一方面)', sent):
                is_key = True; point_type = '论据'
            elif re.search(r'(研究表明|调查显示|据统计|数据显示)', sent):
                is_key = True; point_type = '数据'
            elif re.search(r'(例如|比如|如|譬如)', sent):
                is_key = True; point_type = '例子'
            if is_key:
                points.append({'type': point_type, 'text': sent[:150]})
        return points[:15]


class GenreClassifier:
    """体裁分类器 — v3.8 增强版：融合KAR GenreDetector + 骨架提取 + 领域检测

    8种体裁：process/paper/definition/argument/data_summary/dialogue/essay/poem
    新增功能：
      - detect(): 返回 (genre, confidence, scores_dict)
      - extract_skeleton(): 按体裁提取结构化骨架
      - classify(): 向后兼容接口，返回 (genre, tags)
      - classify_v2(): 增强接口，返回 (genre, tags, intent, confidence)
    """

    GENRE_PATTERNS = {
        'paper': [
            r'(摘要|引言|背景|方法|实验|结果|讨论|结论|参考文献)',
            r'(本文|本研究|本论文|该研究|实验结果表明)',
            r'(提出了一种|提出一个|设计了|实现了)',
        ],
        'essay': [
            r'(我记得|有一天|那时候|小时候|第一次|感动|难忘)',
            r'(开始|后来|最后|终于|从此)',
        ],
        'poem': [
            r'韵脚|平仄|押韵|对仗',
            r'[，,][\u4e00-\u9fff]{5}[，,]',
            r'[，,][\u4e00-\u9fff]{7}[，,]',
        ],
        'process': [
            r'(第[一二三四五六七八九十\d]+[步个环节阶段])[:：]?',
            r'(先|然后|接着|再|最后|首先|其次|随后|步骤)',
            r'(流程|步骤|方法|操作|指南|教程)',
        ],
        'definition': [
            r'(是指|指的是|就是|意为|意味着|即|所谓|定义)',
            r'(分为|包括|包含|由.{0,10}组成|分类)',
        ],
        'argument': [
            r'(因为|所以|因此|从而|导致|由于|原因在于)',
            r'(如果.{0,20}就|只有.{0,20}才|不仅.{0,20}而且)',
            r'(但是|然而|不过|虽然|尽管|总之|综上所述)',
        ],
        'data_summary': [
            r'\d{2,}[年月中天日小时分秒件个人次元种项个条只]',
            r'(增长|下降|提升|减少|占比|达到|超过|低于|统计|平均|%)',
        ],
        'dialogue': [
            r'[「『""][^「『""]{2,40}[」』""]',
            r'^(你|我|他|她|它|我们|你们)',
            r'(哈哈|嘻嘻|嗯|好的|是的|对啊|哦|好吧|加油)',
        ],
    }

    DOMAIN_PATTERNS = {
        '医疗': r'(医疗|医院|医生|病人|患者|药品|手术|诊断|治疗|NMPA|注册证)',
        '编程': r'(Python|Java|代码|函数|算法|类|接口|API|数据库|前端|后端)',
        '法律': r'(法律|法规|合同|合规|条款|诉讼|仲裁|法院)',
        '金融': r'(金融|投资|股票|基金|融资|财务|会计|预算)',
        '招标': r'(招标|投标|标书|采购|政府采购|评分)',
        '公安': r'(公安|反诈|预警|研判|案件|线索|刑侦)',
        '教育': r'(教育|学校|课程|教学|考试|学生|教师|培训)',
        '电子': r'(电路|芯片|PCB|FPGA|嵌入式|传感器|单片机)',
        'AI': r'(AI|人工智能|机器|学习|模型|训练|推理|神经网络)',
    }

    # v3.8: 内容分析意图（来自KAR SmartClassifier）
    INTENT_PATTERNS = {
        '流程': r'(如何|怎么|步骤|流程|方法|方式|过程)',
        '定义': r'(什么|是|定义|概念|含义|解释|指)',
        '论证': r'(为什么|原因|理由|因为|所以|因此|论证)',
        '总结': r'(总结|概括|归纳|综述|概述|汇总)',
        '写作': r'(作文|文章|论文|报告|写作|写一篇)',
    }

    # v3.9: 功能路由意图（来自IntentClassifier，合并为统一管线）
    ACTION_PATTERNS = {
        'learn': {
            "patterns": [r"记住", r"记下来", r"学会", r"知识", r"这个很重要", r"备注", r"保存", r"存下来"],
            "scene_boost": ["学习", "通用"],
            "length_bonus": 50,
        },
        'query': {
            "patterns": [r"是什么", r"什么是", r"怎么", r"如何", r"为什么", r"？", r"\?", r"哪", r"哪个", r"多少"],
            "length_penalty": 50,
        },
        'analyze': {
            "patterns": [r"分析", r"对比", r"趋势", r"占比", r"统计", r"数据", r"比较", r"差异"],
            "scene_boost": ["分析"],
        },
        'summarize': {
            "patterns": [r"总结", r"概括", r"提炼", r"要点", r"核心", r"归纳", r"梳理"],
        },
        'chat': {
            "patterns": [r"你好", r"哈哈", r"谢谢", r"是的", r"早安", r"晚安", r"没事", r"嗯"],
            "default": True,
        },
        'command': {
            "patterns": [r"^/"],
            "priority": 100,
        },
    }

    @classmethod
    def detect(cls, text):
        """v3.9: 检测文本的体裁，返回 (primary_genre, confidence, all_scores)
        增强版：置信度阈值 + 兜底策略 + 体裁冲突检测
        """
        if not text or len(text) < 10:
            return ('knowledge', 0, {})

        scores = {}
        for genre, patterns in cls.GENRE_PATTERNS.items():
            score = 0
            for pat in patterns:
                matches = re.findall(pat, text)
                score += len(matches)
            if score > 0:
                scores[genre] = score / len(patterns)

        if not scores:
            # 默认：如果有中文字符且>20字，归为knowledge
            if len(re.findall(r'[\u4e00-\u9fff]+', text)) > 20:
                scores['knowledge'] = 1.0
            else:
                scores['dialogue'] = 0.5

        primary = max(scores, key=scores.get) if scores else 'knowledge'
        confidence = scores.get(primary, 0)

        # v3.9: 置信度不足 → 回退到knowledge
        if confidence < 0.3 and len(scores) <= 1:
            return ('knowledge', 0.5, {'knowledge': 0.5})

        # v3.9: 体裁冲突检测 — 前两名分差太小，标记低置信度
        if len(scores) > 1:
            top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
            if top2[0][1] - top2[1][1] < 0.15:
                # 分差太小，降低置信度但不改变主体裁
                confidence = confidence * 0.7

        return primary, round(confidence, 3), scores

    @classmethod
    def extract_skeleton(cls, text, genre=None):
        """v3.8: 从文本中提取骨架（结构化的关键点）
        来自KAR extract_skeleton()的增强版
        """
        if genre is None:
            genre, _, _ = cls.detect(text)

        if genre == 'process':
            return cls._extract_process(text)
        elif genre == 'paper':
            return cls._extract_paper(text)
        elif genre == 'definition':
            return cls._extract_definition(text)
        elif genre == 'argument':
            return cls._extract_argument(text)
        elif genre == 'data_summary':
            return cls._extract_data(text)
        else:
            return {'type': genre, 'content': text[:200]}

    @classmethod
    def _extract_process(cls, text):
        """提取流程骨架：步骤列表、顺序"""
        steps = []
        patterns = [
            r'(第[一二三四五六七八九十\d]+[步个环节阶段])[:：]?\s*([^。；\n]{2,60})',
            r'(?:先|首先|第一步)[:：]?\s*([^。；\n]{2,60})',
            r'(?:然后|接着|其次|第二步)[:：]?\s*([^。；\n]{2,60})',
            r'(?:再|接下来|随后|第三步)[:：]?\s*([^。；\n]{2,60})',
            r'(?:最后|第四步)[:：]?\s*([^。；\n]{2,60})',
        ]
        for pat in patterns:
            matches = re.findall(pat, text)
            for m in matches:
                clean = m.strip() if isinstance(m, str) else m[-1].strip()
                if len(clean) > 3 and clean not in steps:
                    steps.append(clean)
        if not steps:
            sentences = re.split(r'[。；\n]', text)
            for s in sentences:
                if any(kw in s for kw in ['需要','必须','步骤','先','然后','最后']):
                    s = s.strip()
                    if len(s) > 5 and s not in steps:
                        steps.append(s)
        return {'type': 'process', 'steps': steps, 'step_count': len(steps)}

    @classmethod
    def _extract_paper(cls, text):
        """提取论文骨架：章节、论点"""
        sections = {}
        current = 'preamble'
        for line in text.split('\n'):
            if re.match(r'^#{1,3}\s', line):
                current = re.sub(r'^#+\s*', '', line)[:30]
                sections[current] = []
            else:
                s = line.strip()
                if len(s) > 10 and current not in ['preamble']:
                    sections[current] = s
        return {'type': 'paper', 'sections': list(sections.keys()), 'structure': sections}

    @classmethod
    def _extract_definition(cls, text):
        """提取定义骨架：定义→分类→属性"""
        def_pat = r'([^，。；\n]{2,20})(?:是指|指的是|就是|意为|即|为|是)([^。；\n]{5,80})'
        definitions = re.findall(def_pat, text)
        cat_pat = r'([^，。；\n]{2,20})(?:分为|包括|包含|由)([^。；\n]{5,80})'
        categories = re.findall(cat_pat, text)
        return {'type': 'definition',
                'definitions': [f'{a}{b}' for a, b in definitions],
                'categories': [f'{a}{b}' for a, b in categories]}

    @classmethod
    def _extract_argument(cls, text):
        """提取论证骨架：原因→结论"""
        reasons = re.findall(r'(?:因为|由于)[^，；。\n]{5,60}', text)
        conclusions = re.findall(r'(?:所以|因此|从而)[^，；。\n]{5,60}', text)
        conditions = re.findall(r'(?:如果|只有).{2,30}(?:就|才)', text)
        contrasts = re.findall(r'(?:虽然|尽管|但是|然而)[^，；。\n]{5,60}', text)
        return {'type': 'argument', 'reasons': reasons,
                'conclusions': conclusions, 'conditions': conditions, 'contrasts': contrasts}

    @classmethod
    def _extract_data(cls, text):
        """提取数据骨架：关键数值"""
        nums = re.findall(r'(\d+[\d,\.]*%?)\s*([年月中天日小时分秒件个人次元种项个条只%倍百分点](?:元|块|万|亿)?)?', text)
        data_points = [f'{n}{u}' for n, u in nums[:10] if n]
        trends = re.findall(r'(增长|下降|提升|减少|上升|下跌)[^，；。\n]{0,30}', text)
        return {'type': 'data_summary', 'data_points': data_points[:10], 'trends': trends[:5]}

    @classmethod
    def classify(cls, text):
        """向后兼容接口：返回 (genre, tags)"""
        genre, confidence, _ = cls.detect(text)
        tags = [d for d, p in cls.DOMAIN_PATTERNS.items() if re.search(p, text)]
        return genre, tags or ['general']

    @classmethod
    def classify_v2(cls, text, query="", scene=""):
        """v3.9 统一管线：一次调用返回所有分类维度
        返回 (genre, tags, action_intent, content_intent, confidence)
        
        genre: 体裁（process/paper/definition/argument/data_summary/dialogue/essay/poem/knowledge）
        tags: 领域标签（编程/AI/医疗/金融等）
        action_intent: 功能路由意图（learn/query/analyze/summarize/chat/command）
        content_intent: 内容分析意图（流程/定义/论证/总结/写作/query）
        confidence: 置信度 0-1
        """
        if not text or not text.strip():
            return ('knowledge', ['general'], 'chat', 'unknown', 0.3)

        # v3.9: 功能路由意图不受文本长度限制（"总结要点"只有4字也应识别）
        action_intent = cls._classify_action(text, scene)

        # 体裁检测需要一定长度
        full = query + " " + text if query else text
        if len(text) < 10:
            # 短文本：只返回功能路由意图，体裁和内容意图用默认值
            tags = [d for d, p in cls.DOMAIN_PATTERNS.items() if re.search(p, full)]
            if not tags:
                tags.append('general')
            return ('knowledge', tags, action_intent, 'unknown', 0.3)

        # 1. 体裁检测
        genre, confidence, scores = cls.detect(text)

        # 2. 功能路由意图已在上方提前计算（action_intent）

        # 3. 内容分析意图
        content_intent = 'query'
        for it, pat in cls.INTENT_PATTERNS.items():
            if re.search(pat, full):
                content_intent = it
                break

        # 内容意图影响体裁选择
        if content_intent == '流程':
            scores['process'] = scores.get('process', 0) + 30
        elif content_intent == '定义':
            scores['definition'] = scores.get('definition', 0) + 20
        elif content_intent == '写作':
            scores['essay'] = scores.get('essay', 0) + 20
        elif content_intent == '总结':
            scores['data_summary'] = scores.get('data_summary', 0) + 20

        genre = max(scores, key=scores.get) if scores else 'knowledge'

        # 4. 领域标签
        tags = [d for d, p in cls.DOMAIN_PATTERNS.items() if re.search(p, full)]
        if not tags:
            tags.append('general')

        final_conf = min(1.0, (scores.get(genre, 0) if scores else 0) / 3 + 0.3)
        return genre, tags, action_intent, content_intent, round(final_conf, 2)

    @classmethod
    def _classify_action(cls, text, scene=""):
        """v3.9: 功能路由意图分类（原IntentClassifier逻辑，已合并）
        返回 learn/query/analyze/summarize/chat/command
        """
        if not text or not text.strip():
            return 'chat'

        # 命令检测
        if text.strip().startswith("/"):
            return 'command'

        # 模式匹配 + 场景加分 + 长度修正
        text_len = len(text)
        scores = {}

        for intent, rule in cls.ACTION_PATTERNS.items():
            if intent == 'command':
                continue

            score = 0.0
            for pattern in rule.get("patterns", []):
                matches = re.findall(pattern, text)
                score += len(matches) * 1.0

            if scene and scene in rule.get("scene_boost", []):
                score += 0.5

            if intent == 'learn' and text_len > rule.get("length_bonus", 50):
                score += 0.3 * min(1.0, (text_len - 50) / 100)
            if intent == 'query' and text_len < rule.get("length_penalty", 50):
                score += 0.2

            scores[intent] = score

        if not scores or max(scores.values()) == 0:
            return 'chat'

        return max(scores, key=scores.get)




# ===== v3.8: 知识推论引擎（来自KAR deep_reason.py） =====

class KnowledgeInferrer:
    """知识推论引擎 — 从已有知识进行推论

    能力：
      - 因果链提取：A导致B
      - 事实提取：识别定义/分类/属性
      - 跨域关联：从因果链推导新结论
    """

    @classmethod
    def extract_causal_links(cls, text):
        """提取因果链"""
        links = []
        sentences = _split_sentences(text)
        for sent in sentences:
            m = re.search(
                r'(?:由于|因为)([\u4e00-\u9fff\w]{5,50})(?:[，,]\s*)?(?:所以|因此|从而|导致)([\u4e00-\u9fff\w]{5,60})',
                sent)
            if m:
                links.append({
                    'cause': m.group(1)[:40],
                    'effect': m.group(2)[:40],
                    'context': sent[:100]
                })
        return links[:10]

    @classmethod
    def infer(cls, text, query=""):
        """进行知识推论"""
        text = _clean_text(text)
        causal_links = cls.extract_causal_links(text)

        # 提取可以用作推论前提的事实
        facts = []
        sentences = _split_sentences(text)
        for sent in sentences:
            if re.search(r'(是|属于|分为|包括|位于|具有|包含)', sent):
                facts.append(sent[:100])

        result = {
            'causal_links': causal_links,
            'facts': facts[:10],
            'summary': ''
        }

        if causal_links:
            parts = ["【因果链】"]
            for link in causal_links:
                parts.append(f"  {link['cause']} → {link['effect']}")
            result['summary'] = '\n'.join(parts)
        elif facts:
            parts = ["【已知事实】"]
            for f in facts[:5]:
                parts.append(f"  · {f}")
            result['summary'] = '\n'.join(parts)

        return result


class DeepReason:
    """深度推理指令生成 — 根据查询类型生成结构化推理骨架

    不是让模型自己推理，而是给模型结构化骨架让它填充。
    识别查询的推理需求(data_analysis/summarize/inference/comparison/composition)
    """

    @classmethod
    def classify_query(cls, text):
        """识别查询的推理需求"""
        types = []
        if re.search(r'(数据|数字|统计|占比|指标|增长|下降|趋势|对比)', text):
            types.append('data_analysis')
        if re.search(r'(总结|概括|摘要|归纳|要点)', text):
            types.append('summarize')
        if re.search(r'(推论|推断|推测|预测|结论|所以|因此)', text):
            types.append('inference')
        if re.search(r'(对比|比较|区别|不同|差异)', text):
            types.append('comparison')
        if re.search(r'(论文|作文|报告|文章)', text):
            types.append('composition')
        if not types:
            types.append('general_query')
        return types

    @classmethod
    def generate_instruction(cls, query, analysis_data=None):
        """生成结构化推理指令"""
        types = cls.classify_query(query)

        parts = []
        parts.append("【深度推理指令】")
        parts.append(f"  查询类型: {'/'.join(types)}")
        parts.append(f"  查询内容: {query}")
        parts.append("")

        if 'data_analysis' in types:
            parts.append("【数据分析】")
            parts.append("  请根据以下数据骨架进行分析：")
            parts.append("  - 找出关键数值的变化趋势")
            parts.append("  - 解释数据变化的原因")
            parts.append("  - 总结数据反映的规律")
            if analysis_data and analysis_data.get('summary'):
                parts.append("")
                parts.append(analysis_data['summary'])
            parts.append("")

        if 'summarize' in types:
            parts.append("【文章总结】")
            parts.append("  请按以下结构输出摘要：")
            parts.append("  1. 核心论点（1-2句）")
            parts.append("  2. 关键论据（2-3个要点）")
            parts.append("  3. 结论/启发（1句）")
            parts.append("")

        if 'inference' in types:
            parts.append("【知识推论】")
            parts.append("  基于已有事实进行推断：")
            parts.append("  - 列出已知条件")
            parts.append("  - 逻辑推导过程")
            parts.append("  - 得出结论")
            parts.append("")

        if 'comparison' in types:
            parts.append("【对比分析】")
            parts.append("  输出格式：")
            parts.append("  | 维度 | A | B |")
            parts.append("  | 对比指标1 | ... | ... |")
            parts.append("  | 结论 | ... | ... |")
            parts.append("")

        parts.append("【输出要求】")
        parts.append("  - 语言简洁，避免啰嗦")
        parts.append("  - 按骨架结构输出，不要跳过")
        parts.append("  - 如果有数据要具体，不要笼统")

        return {
            'types': types,
            'instruction': '\n'.join(parts)
        }


# ===== v3.8: 三层上下文记忆引擎（来自KAR context_memory.py） =====

class ContextMemoryLayer:
    """三层上下文记忆层 — 短程对话/中程主题/长程聚合

    v3.8 融合自KAR context_memory.py，适配左脑引擎：
      1. 短程：当前对话话题延续（复用左脑ContextStack + 增强）
      2. 中程：同主题多次查询知识归约（SQLite持久化）
      3. 长程：主题知识聚合块（SQLite持久化）
    """

    MAX_SESSION_AGE = 3600  # 1小时无更新 → 过期
    MAX_SOURCE_CONTEXT = 2000  # 单条source_context最大字符数

    def __init__(self, data_dir=None):
        self._data_dir = data_dir or DATA_DIR
        self._db_path = self._data_dir / 'context_memory.db'
        self._init_db()

    def _conn(self):
        return sqlite3.connect(str(self._db_path), timeout=10)

    def _init_db(self):
        """建表（幂等）"""
        self._data_dir.mkdir(exist_ok=True)
        conn = self._conn()
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                keyword TEXT,
                context TEXT,
                entities TEXT,
                injection TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS topic_merges (
                topic TEXT,
                keyword TEXT,
                context TEXT,
                merged_text TEXT,
                count INTEGER DEFAULT 1,
                last_updated REAL,
                PRIMARY KEY (topic, keyword)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_keyword ON sessions(keyword)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at)")
        conn.commit()
        conn.close()

    # ----- 1. 短程：对话上下文 -----

    def save_session(self, keyword: str, context: dict) -> bool:
        """保存一轮查询的结果到当前会话"""
        now = time.time()
        session_id = str(int(now)) + '_' + uuid.uuid4().hex[:8]
        entities = context.get("entities", [])
        injection = context.get("injection", "")[:2000]
        context_json = json.dumps(context, ensure_ascii=False)[:5000]
        entities_str = ','.join(entities[:20])

        conn = self._conn()
        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (session_id, keyword, context, entities, injection, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, keyword, context_json, entities_str, injection, now, now))
        conn.commit()
        conn.close()
        return True

    def get_last_context(self, keyword: str, max_age: float = None) -> dict:
        """v3.9: 获取最近的相关上下文 — SimHash语义匹配 + TF-IDF关键词重叠兜底"""
        max_age = max_age or self.MAX_SESSION_AGE
        now = time.time()
        cutoff = now - max_age

        conn = self._conn()
        c = conn.cursor()

        # 第1路：精确匹配（最快）
        c.execute("""
            SELECT keyword, context, entities, injection, updated_at
            FROM sessions WHERE keyword = ? AND updated_at > ?
            ORDER BY updated_at DESC LIMIT 3
        """, (keyword, cutoff))
        rows = c.fetchall()

        if not rows:
            # 第2路：SimHash 语义匹配 — 加载所有session，计算海明距离
            c.execute("""
                SELECT keyword, context, entities, injection, updated_at
                FROM sessions WHERE updated_at > ?
                ORDER BY updated_at DESC LIMIT 50
            """, (cutoff,))
            all_rows = c.fetchall()

            if all_rows:
                # 计算 query 的 SimHash
                q_hash = self._simhash(keyword)
                q_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', keyword))

                scored_rows = []
                for row in all_rows:
                    kw = row[0]
                    # SimHash 海明距离
                    kw_hash = self._simhash(kw)
                    hamming = bin(q_hash ^ kw_hash).count('1')
                    sim_score = max(0, 64 - hamming) / 64  # 0-1, 越大越相似

                    # TF-IDF 关键词重叠
                    kw_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', kw))
                    word_overlap = len(q_words & kw_words) / max(1, len(q_words | kw_words))

                    # 综合分：SimHash*0.6 + 关键词重叠*0.4
                    combined = sim_score * 0.6 + word_overlap * 0.4

                    if combined > 0.3:  # 阈值：30%相似度
                        scored_rows.append((row, combined))

                if scored_rows:
                    scored_rows.sort(key=lambda x: -x[1])
                    rows = [r for r, _ in scored_rows[:3]]

        if not rows:
            # 第3路兜底：取最新不同关键词会话
            c.execute("""
                SELECT keyword, context, entities, injection, updated_at
                FROM sessions WHERE keyword != ? AND keyword NOT LIKE ?
                ORDER BY updated_at DESC LIMIT 1
            """, (keyword, f'%{keyword}%'))
            rows = c.fetchall()

        conn.close()

        if not rows:
            return {}

        merged_injections = []
        all_entities = []
        last_time = 0

        for kw, ctx_json, entities_str, injection, ua in rows:
            if injection:
                merged_injections.append(injection)
            if entities_str:
                all_entities.extend(entities_str.split(','))
            last_time = max(last_time, ua)

        # 去重实体
        seen = set()
        unique_entities = []
        for e in all_entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)

        last_injections = '\n'.join(merged_injections[-3:])
        if len(last_injections) > 3000:
            head = last_injections[:1500]
            sections = re.findall(r'【[^】]+】', last_injections)
            last_injections = head + '\n... [截断] ...\n' + ' '.join(sections)

        prev_real_kw = rows[0][0] if rows else keyword

        return {
            "found": True,
            "age": round(now - last_time, 1),
            "previous_keyword": prev_real_kw,
            "entities": unique_entities,
            "injections_summary": last_injections[:2000],
            "injections_count": len(merged_injections)
        }

    @staticmethod
    def _simhash(text: str, hashbits: int = 64) -> int:
        """v3.9: 简易SimHash — 用于ContextMemoryLayer语义匹配"""
        if not text:
            return 0
        v = [0] * hashbits
        words = re.findall(r'[\u4e00-\u9fff]{2,4}|[a-zA-Z]{2,8}|\d+', text)
        if not words:
            words = [text[i:i+2] for i in range(0, len(text)-1, 2)]
        for word in words:
            h = hash(word) & ((1 << hashbits) - 1)
            for i in range(hashbits):
                bitmask = 1 << i
                if h & bitmask:
                    v[i] += 1
                else:
                    v[i] -= 1
        fingerprint = 0
        for i in range(hashbits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def inherit_context(self, new_keyword: str, prev_context: dict) -> str:
        """从上一个上下文继承实体和推理链"""
        if not prev_context.get("found"):
            return ""

        parts = []
        parts.append("【继承上下文：关联上次对话】")
        parts.append(f"  上次话题: {prev_context.get('previous_keyword', '?')}")
        parts.append(f"  距上次: {prev_context.get('age', '?')}秒前")

        entities = prev_context.get('entities', [])
        if entities:
            q_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', new_keyword))
            overlap = [e for e in entities if e in new_keyword or any(kw in e for kw in q_words)]
            if overlap:
                parts.append(f"  涉及实体: {', '.join(overlap[:10])}")
            else:
                parts.append(f"  相关实体: {', '.join(entities[:10])}")

        prev_inject = prev_context.get('injections_summary', '')
        if prev_inject:
            skeleton = ''
            for line in prev_inject.split('\n'):
                if '【' in line:
                    skeleton += line + ' '
            if skeleton:
                parts.append(f"  上次骨架: {skeleton.strip()[:300]}")

        return '\n'.join(parts)

    def check_topic_continuation(self, prev_keyword: str, new_keyword: str) -> dict:
        """判断两个关键词是否在同一话题延续"""
        if not prev_keyword or not new_keyword:
            return {"is_continuation": False}

        prev_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', prev_keyword))
        new_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', new_keyword))
        overlap = prev_words & new_words

        word_contained = False
        for pw in prev_words:
            for nw in new_words:
                if pw in nw or nw in pw:
                    word_contained = True
                    if pw not in overlap:
                        overlap.add(pw)
                    break

        prev_chars = set(prev_keyword)
        new_chars = set(new_keyword)
        char_overlap = len(prev_chars & new_chars)

        is_cont = bool(overlap) or char_overlap >= 3 or word_contained

        return {
            "is_continuation": is_cont,
            "prev_keyword": prev_keyword,
            "new_keyword": new_keyword,
            "overlap_words": list(overlap) if overlap else [],
            "overlap_chars": char_overlap,
        }

    # ----- 2. 中程+长程：知识聚合 -----

    def merge_topic(self, keyword: str, context: dict) -> dict:
        """把新知识合并到该主题的聚合块"""
        topic = self._extract_topic(keyword)

        entities = context.get("entities", [])
        entities_str = ','.join(entities[:15]) if entities else keyword

        new_text = context.get("injection", "")[:1000]
        if not new_text:
            new_text = keyword

        conn = self._conn()
        c = conn.cursor()

        c.execute("""
            SELECT merged_text, count FROM topic_merges
            WHERE topic = ? AND keyword = ?
        """, (topic, keyword))

        row = c.fetchone()
        now = time.time()

        if row:
            existing_text = row[0]
            count = row[1] + 1
            if new_text in existing_text:
                merged = existing_text
            else:
                merged = existing_text + '\n' + new_text
            if len(merged) > 5000:
                merged = merged[-5000:]
            c.execute("""
                UPDATE topic_merges SET merged_text = ?, count = ?, last_updated = ?
                WHERE topic = ? AND keyword = ?
            """, (merged, count, now, topic, keyword))
        else:
            c.execute("""
                INSERT INTO topic_merges (topic, keyword, context, merged_text, count, last_updated)
                VALUES (?, ?, ?, ?, 1, ?)
            """, (topic, keyword, entities_str, new_text, now))

        conn.commit()

        c.execute("""
            SELECT SUM(count), COUNT(*) FROM topic_merges WHERE topic = ?
        """, (topic,))
        total_freq, unique_q = c.fetchone()
        conn.close()

        return {
            "topic": topic,
            "total_queries": total_freq or 1,
            "unique_keywords": unique_q or 1
        }

    def get_merged(self, keyword: str) -> str:
        """获取此主题已聚合的知识"""
        topic = self._extract_topic(keyword)
        q_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', keyword))

        conn = self._conn()
        c = conn.cursor()

        c.execute("""
            SELECT keyword, merged_text, count FROM topic_merges
            WHERE topic = ?
            ORDER BY count DESC LIMIT 5
        """, (topic,))
        rows = c.fetchall()

        if not rows or len(rows) < 2:
            c.execute("""
                SELECT keyword, merged_text, count FROM topic_merges
                ORDER BY last_updated DESC LIMIT 20
            """)
            all_rows = c.fetchall()
            filtered = []
            for kw, txt, cnt in all_rows:
                kw_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', kw))
                if q_words & kw_words:
                    filtered.append((kw, txt, cnt))
            if filtered:
                rows = sorted(filtered, key=lambda x: -x[2])[:5]

        conn.close()

        if not rows:
            return ""

        parts = []
        total = sum(r[2] for r in rows)
        parts.append(f"【历史知识聚合】主题: {topic} (共{total}次查询)")

        for kw, text, cnt in rows[:3]:
            lines = text.split('\n')
            meaningful = [l for l in lines if l.strip() and not l.startswith('{') and not l.startswith('[')]
            if meaningful:
                parts.append(f"\n  [查询{cnt}次: {kw[:30]}]")
                for line in meaningful[:4]:
                    parts.append(f"    {line[:120]}")

        return '\n'.join(parts)

    def _extract_topic(self, keyword: str) -> str:
        """从关键词中提取主题"""
        cn = re.findall(r'[\u4e00-\u9fff]', keyword)
        if len(cn) >= 4:
            return ''.join(cn[:4])
        elif cn:
            return ''.join(cn)
        en = re.findall(r'[a-zA-Z]+', keyword)
        if en:
            return en[0][:4]
        return keyword[:4]

    def stat(self) -> dict:
        """统计上下文层状态"""
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM sessions")
        sessions = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM topic_merges")
        merges = c.fetchone()[0]
        c.execute("SELECT SUM(count), COUNT(*) FROM topic_merges")
        row = c.fetchone()
        conn.close()
        return {
            "sessions": sessions,
            "topic_merges": merges,
            "total_queries": row[0] or 0,
            "unique_topics": row[1] or 0,
        }


# ===================================================================
# 第三部分：纠缠场引擎（词间关联分析）
# ===================================================================

class EntanglementBridge:
    """纠缠场 — 词间量子关联"""

    def __init__(self, vector_dim=8):
        self.vector_dim = vector_dim
        self.word_phases = {}
        self.entanglement_pairs = {}
        self.evolved_queries = 0

    def _hash_phase(self, word):
        if word in self.word_phases: return self.word_phases[word]
        # 基于词的哈希值确定性生成向量（结果可复现）
        seed_bytes = hashlib.sha256(word.encode("utf-8")).digest()
        vec = []
        for i in range(self.vector_dim):
            # 用不同的哈希段生成每个维度
            chunk = hashlib.sha256(f"{word}_dim_{i}".encode("utf-8")).digest()
            val = int.from_bytes(chunk[:4], "big") / (2**32) * 2 - 1  # 映射到 [-1, 1]
            vec.append(val)
        norm = math.sqrt(sum(v*v for v in vec))
        if norm > 0: vec = [v/norm for v in vec]
        self.word_phases[word] = vec
        return vec

    def evolve(self, query, hit_words):
        q_words = list(set(re.findall(r'[\u4e00-\u9fff]{2,4}', query)))
        added = 0
        for qw in q_words[:3]:
            for hw in hit_words[:20]:
                self._entangle_pair(qw, hw, 0.15 / max(1, len(q_words)))
                added += 1
        for i in range(min(len(hit_words), 30)):
            for j in range(i+1, min(len(hit_words), 30)):
                self._entangle_pair(hit_words[i], hit_words[j], 0.08 / (1 + (j-i) * 0.1))
                added += 1
        self.evolved_queries += 1
        self.save()  # 每次演化后自动持久化
        return added

    def _entangle_pair(self, word_a, word_b, strength=0.1):
        if word_a == word_b: return
        va = self._hash_phase(word_a)
        vb = self._hash_phase(word_b)
        dim = min(len(va), len(vb))
        for k in range(dim):
            va[k] = va[k] + vb[k] * strength
        # Re-normalize
        norm = math.sqrt(sum(v*v for v in va[:dim]))
        if norm > 0:
            for k in range(dim): va[k] /= norm
        pair = (min(word_a, word_b), max(word_a, word_b))
        old = self.entanglement_pairs.get(pair, 0.0)
        self.entanglement_pairs[pair] = min(1.0, old + strength * 0.5)

    def get_entangled(self, seed_word, top_k=12):
        if seed_word not in self.word_phases:
            return []
        vi = self.word_phases[seed_word]
        scores = {}
        for pair, hist in self.entanglement_pairs.items():
            if seed_word in pair:
                other = pair[1] if pair[0] == seed_word else pair[0]
                if other not in self.word_phases: continue
                vj = self.word_phases[other]
                inner = sum(vi[k]*vj[k] for k in range(min(len(vi), len(vj))))
                scores[other] = abs(inner) * 0.3 + hist * 0.7
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    def generate_injection(self, query_text, top_k=12):
        seeds = list(set(re.findall(r'[\u4e00-\u9fff]{2,4}', query_text)))[:5]
        if not seeds: return {"injection": "", "network": {}, "seeds": [], "node_count": 0}
        all_entangled = {}
        for seed in seeds:
            ents = self.get_entangled(seed, top_k=8)
            for w, s in ents:
                if w not in all_entangled or s > all_entangled[w]:
                    all_entangled[w] = s
        sorted_ents = sorted(all_entangled.items(), key=lambda x: -x[1])
        if not sorted_ents: return {"injection": "", "network": {}, "seeds": seeds, "node_count": 0}
        return {
            "injection": f"【纠缠场】种子: {'/'.join(seeds)}\n" + '\n'.join(f"  {w} (强度{s:.3f})" for w, s in sorted_ents[:top_k]),
            "network": {w: round(s, 3) for w, s in sorted_ents},
            "seeds": seeds, "node_count": len(sorted_ents),
        }

    def save(self):
        """持久化纠缠场数据到磁盘"""
        data = {
            "vector_dim": self.vector_dim,
            "word_phases": {w: v for w, v in self.word_phases.items()},
            "entanglement_pairs": {f"{a}|{b}": s for (a, b), s in self.entanglement_pairs.items()},
            "evolved_queries": self.evolved_queries,
        }
        with open(str(ENTANGLE_FILE), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self):
        """从磁盘恢复纠缠场数据"""
        if not ENTANGLE_FILE.exists(): return
        try:
            with open(str(ENTANGLE_FILE), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vector_dim = data.get("vector_dim", self.vector_dim)
            self.word_phases = {w: v for w, v in data.get("word_phases", {}).items()}
            self.entanglement_pairs = {}
            for key, s in data.get("entanglement_pairs", {}).items():
                parts = key.split("|", 1)
                if len(parts) == 2:
                    self.entanglement_pairs[(parts[0], parts[1])] = s
            self.evolved_queries = data.get("evolved_queries", 0)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"⚠️ 纠缠场数据加载失败，将从零开始：{e}")



# 全局实例
# ===================================================================
_memory = None
_entanglement = None

def get_engine():
    global _memory
    # 本地部署版：跳过授权验证
    if _memory is None:
        _memory = MemoryEngine()
    return _memory

def get_entanglement():
    global _entanglement
    if _entanglement is None:
        _entanglement = EntanglementBridge()
        _entanglement.load()  # 从磁盘恢复纠缠数据
    return _entanglement


# ===================================================================
# 种子知识包（冷启动方案）
# ===================================================================

# 编程行业种子知识
SEED_CODING = {
    "nodes": [
        ("Python", "编程"),
        ("JavaScript", "编程"),
        ("API接口", "编程"),
        ("数据库", "编程"),
        ("Git版本控制", "编程"),
        ("Docker容器", "编程"),
        ("RESTful设计", "编程"),
        ("前端框架", "编程"),
        ("后端架构", "编程"),
        ("代码调试", "编程"),
        ("单元测试", "编程"),
        ("设计模式", "编程"),
        ("算法复杂度", "编程"),
        ("微服务", "编程"),
        ("CI/CD流水线", "编程"),
        ("TypeScript", "编程"),
        ("SQL查询", "编程"),
        ("JSON数据格式", "编程"),
    ],
    "edges": [
        ("Python", "API接口", "快速开发"),
        ("Python", "数据库", "ORM操作"),
        ("Python", "单元测试", "pytest框架"),
        ("Python", "算法复杂度", "性能分析"),
        ("JavaScript", "前端框架", "核心语言"),
        ("JavaScript", "TypeScript", "超集"),
        ("JavaScript", "JSON数据格式", "原生支持"),
        ("TypeScript", "前端框架", "类型安全"),
        ("API接口", "RESTful设计", "规范"),
        ("API接口", "后端架构", "暴露层"),
        ("API接口", "JSON数据格式", "数据交换"),
        ("数据库", "SQL查询", "操作语言"),
        ("数据库", "后端架构", "持久层"),
        ("Git版本控制", "CI/CD流水线", "触发源"),
        ("Docker容器", "微服务", "部署单元"),
        ("Docker容器", "CI/CD流水线", "构建环境"),
        ("后端架构", "微服务", "架构风格"),
        ("后端架构", "设计模式", "代码组织"),
        ("代码调试", "单元测试", "验证手段"),
        ("单元测试", "CI/CD流水线", "质量门禁"),
        ("算法复杂度", "设计模式", "权衡依据"),
    ]
}

# 通用技术种子知识
SEED_TECH = {
    "nodes": [
        ("大模型", "技术"),
        ("知识图谱", "技术"),
        ("语义哈希", "技术"),
        ("XOR加密", "技术"),
        ("HMAC签名", "技术"),
        ("Token优化", "技术"),
        ("WorkBuddy", "技术"),
        ("向量检索", "技术"),
    ],
    "edges": [
        ("大模型", "Token优化", "成本相关"),
        ("知识图谱", "语义哈希", "索引技术"),
        ("知识图谱", "向量检索", "检索方式"),
        ("XOR加密", "HMAC签名", "组合加密"),
        ("WorkBuddy", "大模型", "基于"),
        ("WorkBuddy", "知识图谱", "集成"),
    ]
}

# v3.4: 通用知识种子（30条，覆盖日常场景）
SEED_GENERAL = {
    "nodes": [
        ("Python是一种高级编程语言", "编程"),
        ("AI即人工智能，模拟人类智能的技术", "技术"),
        ("机器学习是AI的子领域，让计算机从数据中学习", "技术"),
        ("深度学习使用多层神经网络处理复杂任务", "技术"),
        ("项目管理包含需求分析、开发、测试、部署", "工作"),
        ("敏捷开发强调迭代交付和快速响应变化", "工作"),
        ("API是应用程序之间的通信接口", "编程"),
        ("SQL是用于管理关系数据库的语言", "编程"),
        ("HTTP是互联网通信的基础协议", "技术"),
        ("JSON是一种轻量级数据交换格式", "技术"),
        ("Git是最流行的版本控制系统", "编程"),
        ("Linux是开源的服务器操作系统", "技术"),
        ("Docker简化了应用的打包和部署", "技术"),
        ("云计算通过互联网提供计算资源", "技术"),
        ("React是流行的前端UI框架", "编程"),
        ("Node.js基于JavaScript的服务器运行环境", "编程"),
        ("数据库索引能大幅提升查询速度", "数据库"),
        ("缓存通过存储临时数据减少重复计算", "技术"),
        ("负载均衡将请求分发到多台服务器", "技术"),
        ("微服务将应用拆分为独立的小服务", "架构"),
        ("HTTPS加密传输保护数据安全", "安全"),
        ("JWT是无状态的用户认证令牌", "安全"),
        ("RESTful API使用标准HTTP方法操作资源", "编程"),
        ("MVC将应用分为模型、视图、控制器", "架构"),
        ("DevOps融合开发与运维提高交付效率", "工作"),
        ("单元测试验证代码的最小功能单元", "测试"),
        ("代码审查有助于发现缺陷和统一代码风格", "工作"),
        ("产品需求文档描述产品的功能和特性", "工作"),
        ("Sprint是敏捷开发中的固定周期迭代", "工作"),
        ("持续集成自动化构建和测试代码变更", "工作"),
    ],
    "edges": [
        ("AI即人工智能，模拟人类智能的技术", "机器学习是AI的子领域，让计算机从数据中学习", "子领域"),
        ("机器学习是AI的子领域，让计算机从数据中学习", "深度学习使用多层神经网络处理复杂任务", "子领域"),
        ("Python是一种高级编程语言", "机器学习是AI的子领域，让计算机从数据中学习", "常用语言"),
        ("Python是一种高级编程语言", "Docker简化了应用的打包和部署", "开发环境"),
        ("API是应用程序之间的通信接口", "RESTful API使用标准HTTP方法操作资源", "规范"),
        ("API是应用程序之间的通信接口", "HTTP是互联网通信的基础协议", "传输协议"),
        ("SQL是用于管理关系数据库的语言", "数据库索引能大幅提升查询速度", "优化"),
        ("Git是最流行的版本控制系统", "持续集成自动化构建和测试代码变更", "流水线"),
        ("Docker简化了应用的打包和部署", "微服务将应用拆分为独立的小服务", "部署方式"),
        ("微服务将应用拆分为独立的小服务", "负载均衡将请求分发到多台服务器", "配合"),
        ("项目管理包含需求分析、开发、测试、部署", "敏捷开发强调迭代交付和快速响应变化", "方法论"),
        ("敏捷开发强调迭代交付和快速响应变化", "Sprint是敏捷开发中的固定周期迭代", "实践"),
        ("项目管理包含需求分析、开发、测试、部署", "产品需求文档描述产品的功能和特性", "文档"),
        ("代码审查有助于发现缺陷和统一代码风格", "单元测试验证代码的最小功能单元", "质量保障"),
        ("DevOps融合开发与运维提高交付效率", "持续集成自动化构建和测试代码变更", "实践"),
        ("HTTPS加密传输保护数据安全", "JWT是无状态的用户认证令牌", "安全"),
        ("Linux是开源的服务器操作系统", "Docker简化了应用的打包和部署", "运行环境"),
        ("云计算通过互联网提供计算资源", "Docker简化了应用的打包和部署", "部署平台"),
    ]
}

SEED_PACKS = {
    "coding": SEED_CODING,
    "tech": SEED_TECH,
    "general": SEED_GENERAL,
}


def seed_knowledge(pack_name: str = "all") -> Dict:
    """向记忆引擎注入种子知识包，解决冷启动问题

    Args:
        pack_name: "coding"(编程), "tech"(技术), "all"(全部)
    """
    engine = get_engine()
    packs = list(SEED_PACKS.values()) if pack_name == "all" else [SEED_PACKS.get(pack_name)]
    packs = [p for p in packs if p is not None]

    if not packs:
        return {"status": "error", "reason": f"未知种子包：{pack_name}，可选：{', '.join(SEED_PACKS.keys())}, all"}

    total_nodes = 0
    total_edges = 0

    for pack in packs:
        # 批量添加节点（静默模式，不增加 learn_count / token_savings）
        idx_map = {}
        for text, category in pack["nodes"]:
            idx = engine.add_node(text, category, _silent=True)
            idx_map[text] = idx
            total_nodes += 1

        # 批量添加关联
        for a_text, b_text, relation in pack["edges"]:
            if a_text in idx_map and b_text in idx_map:
                engine.add_edge(idx_map[a_text], idx_map[b_text], relation)
                total_edges += 1

    return {
        "status": "ok",
        "action": "种子知识注入",
        "pack": pack_name,
        "nodes_added": total_nodes,
        "edges_added": total_edges,
        "message": f"✅ 已注入 {total_nodes} 个知识节点 + {total_edges} 条关联"
    }


# ===================================================================
# WorkBuddy 统一入口
# ===================================================================

def workbuddy_main(action: str, content: str = "", **kwargs) -> Dict:
    """左脑统一入口

    记忆类动作:
      learn     — 学习新知识
      query     — 查询知识
      relate    — 建立关联
      search    — 图扩散查询（核心功能）
      correct   — 纠错
      disambig  — 多义词消歧
      auto      — 自动感知模式（auto on/off，增强版）
      suggest   — 推荐相关知识
      analysis  — 使用习惯分析
      tips      — 优化建议
      auto_process — 一键自动处理（纠错+学习+分析）
      stats     — 引擎状态
      dashboard — ToKen监测助手（Token/时间节省数据）
      reset     — 重置数据

    感知增强类动作（v2 新增）:
      perceive  — 自动感知意图并处理（入口级命令）
      intent    — 查看意图分类结果（调试用）
      enhance   — 上下文增强：检索相关知识注入当前语境
      context   — 查看当前上下文栈 / 清空上下文(context clear)
      recommend — 智能推荐（基于知识图谱+纠缠场+用户画像）
      profile   — 查看用户画像
      trace     — 追溯知识来源（按需回溯source_context，方案B核心）

    推理类动作:
      analyze        — 数据分析（提取数字/趋势/对比）
      summarize      — 文章总结
      classify       — 体裁分类（增强版：8种体裁+骨架+领域+意图）
      infer          — 知识推论（因果链提取+事实提取）
      deepreason     — 深度推理指令生成
      entangle       — 纠缠场关联分析
      evolve         — 演化纠缠关系
    """
    engine = get_engine()

    # 记忆类动作
    if action == "learn":
        text = kwargs.get("text", content)
        category = kwargs.get("category", engine.auto_category(text))
        # v3.8: 自动检测genre/skeleton/domain
        genre = kwargs.get("genre", "")
        skeleton = kwargs.get("skeleton", "")
        domain = kwargs.get("domain", "")
        if not genre:
            genre, tags, _, _, conf = GenreClassifier.classify_v2(text)
            if not domain:
                domain = tags[0] if tags else ""
            if not skeleton:
                skeleton = json.dumps(GenreClassifier.extract_skeleton(text, genre), ensure_ascii=False)[:500]
        source_context = kwargs.get("source_context", engine._extract_source_context(text))
        source_turn_index = kwargs.get("source_turn_index", engine._turn_index)
        idx = engine.add_node(text, category, genre=genre, skeleton=skeleton,
                              domain=domain, source_context=source_context,
                              source_turn_index=source_turn_index)
        engine._record_today("learn", 200)
        engine._session_learned.append({"text": text[:60], "time": time.time()})
        return {"status": "ok", "action": "学习", "text": text, "category": category,
                "node_idx": idx, "genre": genre, "domain": domain}

    elif action == "query":
        text = kwargs.get("text", content)
        idx = engine.find_node(text)
        # v3.4: 精确未命中时，尝试语义搜索并返回候选让用户确认
        if idx is None:
            fuzzy_results = engine.fuzzy_find(text, top_k=3)
            if fuzzy_results:
                candidates = []
                for fi, score in fuzzy_results:
                    node = engine.nodes[fi]
                    candidates.append({
                        "text": node["text"][:80],
                        "category": node.get("category", ""),
                        "score": round(score, 2)
                    })
                return {
                    "status": "needs_clarification",
                    "query": text,
                    "message": "🔍 没找到精确匹配，以下可能是你要找的：",
                    "candidates": candidates
                }
        if idx is None:
            corrected = engine.fix_typo(text)
            idx = engine.find_node(corrected) if corrected != text else None
            if idx is not None: text = corrected
        if idx is None: return {"status": "not_found", "query": content}
        engine._record_today("query", 800)
        node = engine.nodes[idx]
        edges = [{"text": engine.nodes[e[0]]["text"], "relation": e[1]} for e in node["edges"]]
        return {"status": "ok", "text": node["text"], "edges": edges}

    elif action == "relate":
        parts = [p.strip() for p in content.split(",")]
        if len(parts) < 2: return {"status": "error", "reason": "格式: 概念A,概念B[,关系]"}
        a, b = parts[0], parts[1]
        rel = parts[2] if len(parts) >= 3 else engine.detect_relation(a, b)
        a_idx = engine.add_node(a)
        b_idx = engine.add_node(b)
        engine.add_edge(a_idx, b_idx, rel)
        return {"status": "ok", "action": "关联", "a": a, "b": b, "relation": rel}

    elif action == "search":
        text = kwargs.get("text", content)
        max_hops = int(kwargs.get("max_hops", 3))
        engine._record_today("search")
        return engine.query_diffusion(text, max_hops=max_hops)

    elif action == "correct":
        text = kwargs.get("text", content)
        corrected = engine.fix_typo(text)
        engine.correct_count += 1
        engine._record_today("correct")
        return {"status": "ok", "original": text, "corrected": corrected, "changed": text != corrected}

    elif action == "disambig":
        word = kwargs.get("text", content)
        context = kwargs.get("context", "")
        return {"status": "ok", "word": word, "meaning": engine.disambiguate(word, context)}

    elif action in ("stats", "status"):
        mem_stats = engine.stats()
        ent_count = len(get_entanglement().word_phases) if get_entanglement() else 0
        mem_stats["entanglement_words"] = ent_count
        return mem_stats

    elif action == "dashboard":
        """ToKen监测助手 — 展示实时节省数据"""
        fmt = kwargs.get("format", "text")
        if fmt == "json":
            return engine.dashboard()
        return {"status": "ok", "action": "仪表盘", "output": engine.dashboard_text()}

    elif action == "analysis":
        """使用习惯分析 — 总结用户使用模式"""
        return {"status": "ok", "action": "使用分析", "output": engine._analysis_text()}

    elif action == "tips":
        """优化建议 — 根据使用情况生成个性化建议"""
        tips = engine.optimization_tips()
        text = "\n".join(tips) if tips else "暂无建议"
        return {"status": "ok", "action": "优化建议", "count": len(tips), "output": text}

    elif action == "auto":
        """自动感知模式 — 开启后自动感知意图并智能路由"""
        cmd = content.strip().lower()
        if cmd in ("on", "开", "开启", "1", "true"):
            return engine.set_auto_mode(True)
        elif cmd in ("off", "关", "关闭", "0", "false"):
            return engine.set_auto_mode(False)
        else:
            return {"status": "ok", "auto_mode": "已开启 ✅" if engine.auto_mode else "已关闭 ❌",
                    "usage": "使用：/左脑 auto on  开启自动感知  |  /左脑 auto off  关闭自动感知"}

    elif action == "suggest":
        """根据对话自动推荐相关知识"""
        text = kwargs.get("text", content)
        return engine.suggest(text)

    elif action == "auto_process":
        """一键自动处理：学习 + 分析 + 纠错"""
        text = kwargs.get("text", content)
        return engine.auto_process(text)

    # ===== 感知增强类动作（v2 新增）=====

    elif action == "perceive":
        """自动感知意图并处理 — 入口级命令，串联感知→增强→推荐"""
        text = kwargs.get("text", content)
        result = engine.auto_process_v2(text)
        # 自动附加上下文增强
        if result.get("intent") in ("query", "analyze", "learn"):
            enhanced = engine.enhance(text)
            result["enhanced"] = enhanced.get("injection_text", "")
        # 自动附加推荐
        if result.get("intent") in ("query", "chat"):
            rec = engine.recommend(text)
            if rec.get("recommendations"):
                result["recommended"] = rec["recommendations"]
        return result

    elif action == "intent":
        """查看意图分类结果（调试用）"""
        text = kwargs.get("text", content)
        scene = engine.auto_category(text)
        intent = IntentClassifier.classify(text, scene)
        return {"status": "ok", "text": text, "scene": scene, "intent": intent}

    elif action == "enhance":
        """上下文增强：检索相关知识注入当前语境"""
        text = kwargs.get("text", content)
        return engine.enhance(text)

    elif action == "context":
        """查看当前上下文栈 / 清空上下文"""
        cmd = content.strip().lower()
        if cmd in ("clear", "清空", "重置", "reset"):
            engine.context_stack.clear()
            engine._save()  # v3.5: 持久化清空状态
            return {"status": "ok", "action": "上下文已清空"}
        else:
            ctx_kw = engine.context_stack.get_context_keywords(top_k=8)
            stack_size = len(engine.context_stack.stack)
            current_scene = engine.context_stack.get_current_scene()
            return {
                "status": "ok",
                "stack_size": stack_size,
                "current_scene": current_scene,
                "context_keywords": [(kw, f"{w:.2f}") for kw, w in ctx_kw],
                "tip": "使用 /左脑 context clear 清空上下文" if stack_size > 0 else "上下文为空",
            }

    elif action == "recommend":
        """智能推荐 — 基于知识图谱+纠缠场+用户画像"""
        text = kwargs.get("text", content)
        cmd = content.strip().lower()
        if cmd in ("reset", "重置"):
            engine.user_profile.reset_recommendations()
            return {"status": "ok", "action": "推荐历史已重置"}
        return engine.recommend(text)

    elif action == "profile":
        """查看用户画像"""
        engine.user_profile.build()
        p = engine.user_profile
        depth_names = {
            "memory_heavy": "偏记忆型 🧠",
            "analysis_heavy": "偏分析型 🔍",
            "balanced": "均衡型 ⚖️",
        }
        top_interests = sorted(p.interest_vector.items(), key=lambda x: -x[1])[:5]
        return {
            "status": "ok",
            "depth_type": p.depth_type,
            "depth_type_display": depth_names.get(p.depth_type, p.depth_type),
            "top_interests": top_interests,
            "active_hours_count": len(p.active_hours),
            "recommend_history_size": len(p.recommended_history),
        }

    elif action == "reset":
        return engine.reset()

    elif action == "seed":
        """注入种子知识包（冷启动）"""
        return seed_knowledge(content.strip() or "all")

    elif action == "export":
        """数据导出为CSV"""
        if _DB_AVAILABLE:
            data_type = content.strip() or "all"
            filepath = str(DATA_DIR / f"left_brain_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            export_csv_to_file(filepath, data_type)
            return {"status": "ok", "action": "数据导出", "file": filepath, "message": f"✅ 数据已导出到 {filepath}"}
        return {"status": "error", "reason": "token_db模块未加载"}

    elif action == "achievements":
        """查看成就系统"""
        if _DB_AVAILABLE:
            achs = get_achievements()
            unlocked = [a for a in achs if a["unlocked"]]
            locked = [a for a in achs if not a["unlocked"]]
            lines = [f"🏆 成就系统 ({len(unlocked)}/{len(achs)} 已解锁)", "=" * 35]
            if unlocked:
                lines.append("✅ 已解锁：")
                for a in unlocked:
                    lines.append(f"  {a['icon']} {a['name']} — {a['description']}")
            if locked:
                lines.append("🔒 未解锁：")
                for a in locked[:8]:
                    lines.append(f"  {a['icon']} {a['name']} — {a['description']}")
            return {"status": "ok", "action": "成就系统", "output": "\n".join(lines),
                    "unlocked_count": len(unlocked), "total_count": len(achs)}
        return {"status": "error", "reason": "token_db模块未加载"}

    elif action == "history":
        """历史趋势数据"""
        if _DB_AVAILABLE:
            days = int(content.strip() or "30")
            snapshots = get_daily_snapshots(days)
            if not snapshots:
                return {"status": "ok", "message": "暂无历史数据", "snapshots": []}
            lines = [f"📈 最近{days}天趋势", "=" * 40]
            for s in reversed(snapshots):
                lines.append(f"  {s['date']}  Token:{s['token_savings']}  操作:{s['total_ops']}  ¥{s['money_saved']:.2f}")
            return {"status": "ok", "action": "历史趋势", "output": "\n".join(lines), "snapshots": snapshots}
        return {"status": "error", "reason": "token_db模块未加载"}

    elif action == "heatmap":
        """时段热力图"""
        if _DB_AVAILABLE:
            date = content.strip() or None
            data = get_hourly_data(date)
            display_date = date or datetime.now().strftime("%Y-%m-%d")
            lines = [f"🔥 时段热力图 ({display_date})", "=" * 35]
            max_ops = max(h["ops"] for h in data) if data else 0
            for h in data:
                bar_len = int(h["ops"] / max(max_ops, 1) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {h['hour']:02d}:00 {bar} {h['ops']}次/{h['tokens']}t")
            return {"status": "ok", "action": "时段热力图", "output": "\n".join(lines), "data": data}
        return {"status": "error", "reason": "token_db模块未加载"}

    # 推理类动作
    elif action == "analyze":
        text = kwargs.get("text", content)
        engine.analyze_count += 1
        engine._record_today("analyze")
        return DataAnalyzer.analyze(text)

    elif action == "summarize":
        text = kwargs.get("text", content)
        engine.summarize_count += 1
        engine._record_today("summarize")
        return Summarizer.summarize(text)

    elif action == "classify":
        text = kwargs.get("text", content)
        # v3.9: 统一管线classify，返回所有维度
        genre, tags, action_intent, content_intent, conf = GenreClassifier.classify_v2(text)
        skeleton = GenreClassifier.extract_skeleton(text, genre)
        return {"status": "ok", "genre": genre, "tags": tags,
                "action_intent": action_intent, "content_intent": content_intent,
                "confidence": conf, "skeleton": skeleton}

    elif action == "infer":
        """v3.8: 知识推论 — 因果链提取+事实提取"""
        text = kwargs.get("text", content)
        return KnowledgeInferrer.infer(text, query=text)

    elif action == "deepreason":
        """v3.8: 深度推理指令生成"""
        text = kwargs.get("text", content)
        analysis = DataAnalyzer.analyze(text) if re.search(r'\d', text) else None
        return DeepReason.generate_instruction(text, analysis)

    elif action == "trace":
        """v3.8: 追溯知识来源 — 按需回溯source_context（方案B核心）"""
        keyword = kwargs.get("text", content).strip()
        return engine.trace_source_context(keyword)

    elif action == "context_layers":
        """v3.8: 查看三层上下文状态"""
        return engine._context_memory.stat()

    elif action == "entangle":
        text = kwargs.get("text", content)
        engine.entangle_count += 1
        engine._record_today("entangle")
        eb = get_entanglement()
        return eb.generate_injection(text)

    elif action == "evolve":
        parts = content.split("|")
        query = parts[0] if parts else content
        words = [w.strip() for w in parts[1].split(",")] if len(parts) > 1 else re.findall(r'[\u4e00-\u9fff]{2,4}', query)
        eb = get_entanglement()
        n = eb.evolve(query, words)
        return {"status": "ok", "added_pairs": n, "total_pairs": len(eb.entanglement_pairs)}

    # ====================== v3.4 新增：知识管理 ======================

    elif action in ("backup", "备份"):
        """导出全部知识为加密备份包"""
        import base64 as _b64
        backup_data = {
            "version": "3.4",
            "nodes": engine.nodes,
            "categories": engine.categories,
            "entanglement": get_entanglement().entanglement_pairs if get_entanglement() else [],
            "learn_count": engine.learn_count,
            "query_count": engine.query_count,
            "stats": engine.stats(),
            "backup_at": datetime.now().isoformat()
        }
        raw = json.dumps(backup_data, ensure_ascii=False).encode("utf-8")
        # 简单混淆（非安全加密，仅防止肉眼阅读）
        key = hashlib.sha256(SECRET_KEY + b"BACKUP_SALT_v3.4").digest()
        obfuscated = bytes(b ^ key[i % len(key)] for i, b in enumerate(raw))
        encoded = _b64.b64encode(obfuscated).decode()
        return {
            "status": "ok", "action": "备份",
            "format": "base64_xor",
            "node_count": len(engine.nodes),
            "edge_count": sum(len(n.get("edges",[])) for n in engine.nodes),
            "data": encoded,
            "tip": "请复制上述 data 字段的全部内容保存为文本文件。恢复时使用 /左脑 恢复 <粘贴数据>"
        }

    elif action in ("restore", "恢复"):
        """从备份包恢复知识"""
        import base64 as _b64
        data_str = kwargs.get("text", content).strip()
        if not data_str:
            return {"status": "error", "reason": "请提供备份数据：/左脑 恢复 <粘贴备份内容>"}
        try:
            key = hashlib.sha256(SECRET_KEY + b"BACKUP_SALT_v3.4").digest()
            obfuscated = _b64.b64decode(data_str)
            raw = bytes(b ^ key[i % len(key)] for i, b in enumerate(obfuscated))
            backup = json.loads(raw.decode("utf-8"))
        except Exception:
            return {"status": "error", "reason": "备份数据无效或已损坏，请确认复制完整"}
        
        old_node_count = len(engine.nodes)
        # 合并节点
        for node in backup.get("nodes", []):
            text = node.get("text", "")
            if text and engine.find_node(text) is None:
                engine.add_node(text, node.get("category", "general"))
        # 合并纠缠数据
        eb = get_entanglement()
        for pair in backup.get("entanglement", []):
            eb._entangle_pair(pair[0], pair[1]) if len(pair) >= 2 else None
        
        engine._save()
        new_count = len(engine.nodes)
        return {
            "status": "ok", "action": "恢复",
            "old_nodes": old_node_count,
            "new_nodes": new_count,
            "added": new_count - old_node_count,
            "backup_date": backup.get("backup_at", "未知")
        }

    elif action in ("knowledge_map", "知识地图"):
        """知识图谱概览"""
        lines = ["🧠 知识地图", "=" * 40]
        lines.append(f"节点总数: {len(engine.nodes)}")
        total_edges = sum(len(n.get("edges",[])) for n in engine.nodes)
        lines.append(f"连接总数: {total_edges}")
        lines.append(f"分类数: {len(engine.categories)}")
        lines.append("")
        
        # 核心节点排行
        lines.append("📌 核心节点（连接最多）:")
        ranked = sorted(enumerate(engine.nodes), key=lambda x: len(x[1].get("edges",[])), reverse=True)
        for i, (idx, node) in enumerate(ranked[:10]):
            edge_count = len(node.get("edges",[]))
            if edge_count == 0: break
            text = node.get("text","")[:30]
            lines.append(f"  {i+1}. {text} ({edge_count}条关联)")
        
        # 分类统计
        if engine.categories:
            lines.append(f"\n📂 分类分布:")
            for cat in engine.categories[:8]:
                count = sum(1 for n in engine.nodes if n.get("category") == cat)
                lines.append(f"  {cat}: {count}个节点")
        
        return {"status": "ok", "action": "知识地图", "output": "\n".join(lines)}

    elif action in ("recent", "最近学到的"):
        """最近学到的知识"""
        days = int(content.strip() or "7")
        # 从 nodes 中找最近添加的（按索引倒序）
        recent_nodes = []
        for i in range(len(engine.nodes)-1, max(-1, len(engine.nodes)-51), -1):
            node = engine.nodes[i]
            recent_nodes.append(node.get("text","")[:60])
        
        lines = [f"📝 最近学到的知识（共{len(recent_nodes)}条）", "=" * 40]
        for i, text in enumerate(recent_nodes[:20], 1):
            lines.append(f"  {i}. {text}")
        if len(recent_nodes) > 20:
            lines.append(f"  ... 还有 {len(recent_nodes)-20} 条")
        
        return {"status": "ok", "action": "最近学到的", "count": len(recent_nodes), "output": "\n".join(lines)}

    elif action in ("feedback", "反馈"):
        """记录用户反馈"""
        text = content.strip()
        if not text:
            return {"status": "error", "reason": "请提供反馈内容"}
        engine._feedback_log = getattr(engine, '_feedback_log', [])
        engine._feedback_log.append({"text": text, "time": datetime.now().isoformat()})
        engine._save()
        return {"status": "ok", "action": "反馈", "message": "感谢反馈！已记录"}

    elif action in ("settings", "设置"):
        """查看/修改配置"""
        auto_mode = getattr(engine, 'auto_mode', False)
        lines = [
            "⚙️ 左脑设置",
            "=" * 30,
            f"自动感知: {'🟢 开启' if auto_mode else '⚫ 关闭'}",
            f"知识节点: {len(engine.nodes)}",
            f"上下文轮数: {ContextStack.MAX_ROUNDS if hasattr(ContextStack, 'MAX_ROUNDS') else 10}",
            "授权状态: ✅ 本地部署版（无DRM）",
            "",
            "可用操作:",
            "  /左脑 自动 开/关    切换自动感知",
            "  /左脑 备份           导出知识",
            "  /左脑 恢复           导入知识",
        ]
        return {"status": "ok", "action": "设置", "output": "\n".join(lines)}

    elif action in ("export_knowledge", "导出知识"):
        """v3.4: 导出知识图谱为 Markdown + Mermaid 格式"""
        fmt = kwargs.get("format", content.strip() or "markdown")
        
        if fmt in ("markdown", "md"):
            lines = ["# 🧠 左脑知识图谱", f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f"节点总数: {len(engine.nodes)}", ""]
            # 按分类组织
            by_cat = {}
            for node in engine.nodes:
                cat = node.get("category", "未分类")
                by_cat.setdefault(cat, []).append(node)
            for cat, nodes in by_cat.items():
                lines.append(f"## {cat} ({len(nodes)}条)")
                for node in nodes[:20]:
                    lines.append(f"- {node['text'][:80]}")
                if len(nodes) > 20:
                    lines.append(f"- ... 还有 {len(nodes)-20} 条")
                lines.append("")
            return {"status": "ok", "action": "导出知识", "format": "markdown", "output": "\n".join(lines)}
        
        elif fmt in ("mermaid", "mmd"):
            lines = ["```mermaid", "graph TD"]
            # Top nodes
            ranked = sorted(enumerate(engine.nodes), key=lambda x: len(x[1].get("edges",[])), reverse=True)
            shown = set()
            count = 0
            for idx, node in ranked[:30]:
                if count >= 50: break
                nid = f"n{idx}"
                label = node["text"][:20].replace('"',"'")
                lines.append(f'    {nid}["{label}"]')
                shown.add(idx)
                count += 1
                for edge in node.get("edges",[])[:3]:
                    if edge[0] not in shown:
                        label2 = engine.nodes[edge[0]]["text"][:20].replace('"',"'")
                        lines.append(f'    n{edge[0]}["{label2}"]')
                        shown.add(edge[0])
                    lines.append(f'    {nid} -->|{edge[1]}| n{edge[0]}')
                    count += 1
            lines.append("```")
            return {"status": "ok", "action": "导出知识", "format": "mermaid", "output": "\n".join(lines)}
        
        return {"status": "error", "reason": f"不支持的格式: {fmt}，可选: markdown, mermaid"}

    # ====================== v3.5: 增删改查 ======================

    # ====================== v3.5: 全局搜索 ======================

    elif action == "_set_workspace":
        """v3.6: 内部动作 — 设置当前工作区"""
        old_ws = getattr(engine, '_current_workspace', 'global')
        new_ws = engine.get_workspace_key(content)
        engine._current_workspace = new_ws
        engine._saved_workspace = old_ws  # v3.6: 保存上一个 workspace 供全局搜索恢复
        return {"status": "ok", "workspace": new_ws}

    elif action in ("gsearch", "全局搜索"):
        """v3.6: 跨 workspace 搜索，不做隔离过滤"""
        text = kwargs.get("text", content)
        saved_ws = getattr(engine, '_current_workspace', 'global')
        engine._current_workspace = None  # 取消过滤
        result = engine.query_diffusion(text, max_hops=int(kwargs.get("max_hops", 3)))
        engine._current_workspace = saved_ws  # v3.6: 恢复之前的 workspace
        return result

    elif action in ("edit", "修改"):
        text = kwargs.get("text", content)
        if "|" in text:
            old, new = text.split("|", 1)
        elif "→" in text or "->" in text:
            parts = text.replace("->", "→").split("→", 1)
            old, new = parts[0], parts[1]
        else:
            return {"status": "error", "reason": "格式：/左脑 修改 旧文本|新文本"}
        result = engine.update_node(old.strip(), new.strip())
        if result["status"] == "ok" and "new" in result:
            result["message"] = f"已修改：{result['old'][:40]} → {result['new'][:40]}"
        return result

    elif action in ("delete", "删除"):
        text = kwargs.get("text", content)
        if not text:
            return {"status": "error", "reason": "请指定要删除的关键词"}
        result = engine.delete_node(text.strip())
        if result["status"] == "ok":
            result["message"] = f"已删除：{result.get('text', text)[:40]}"
        return result

    elif action in ("list", "记忆列表"):
        page = int(content.strip() or "1")
        result = engine.list_nodes(page=page, page_size=15)
        if result["status"] == "ok":
            lines = [f"记忆列表（共{result['total']}条，第{result['page']}/{result['total_pages']}页）"]
            for item in result["items"]:
                lines.append(f"  [{item['category']}] {item['text']}")
            if result["total_pages"] > 1:
                lines.append(f"  输入 /左脑 记忆列表 {page+1} 查看下一页")
            result["output"] = "\n".join(lines)
        return result

    # ====================== v3.6: Workspace 管理 ======================

    elif action in ("workspace", "工作区"):
        """v3.6: 查看/切换 workspace"""
        cmd = content.strip()
        current_ws = getattr(engine, '_current_workspace', 'global')
        if not cmd or cmd in ("status", "状态", "查看"):
            # 统计各 workspace 的知识数量
            ws_counts = {}
            for n in engine.nodes:
                if n is None:
                    continue
                ws_name = n.get("workspace", "global")
                ws_counts[ws_name] = ws_counts.get(ws_name, 0) + 1
            lines = [f"当前工作区：{current_ws}"]
            lines.append(f"知识分布：")
            for ws_name, count in sorted(ws_counts.items(), key=lambda x: -x[1]):
                marker = " ← 当前" if ws_name == current_ws else ""
                lines.append(f"  {ws_name}: {count}条{marker}")
            return {"status": "ok", "workspace": current_ws, "distribution": ws_counts, "output": "\n".join(lines)}
        elif cmd == "global" or cmd.startswith("ws_"):
            old_ws = engine._current_workspace
            engine._current_workspace = cmd
            engine._saved_workspace = old_ws
            return {"status": "ok", "message": f"已切换到工作区：{cmd}", "workspace": cmd}
        else:
            return {"status": "ok", "output": f"当前工作区：{current_ws}\n用法：/左脑 workspace 查看分布 | /左脑 workspace global 切换全局"}

    # ====================== v3.5: 待确认队列 ======================

    elif action in ("pending", "待确认"):
        engine._pending_queue = getattr(engine, '_pending_queue', [])
        cmd = content.strip()
        if cmd in ("全部保留", "保留全部"):
            count = len(engine._pending_queue)
            for item in engine._pending_queue:
                cat = engine.auto_category(item["text"])
                engine.add_node(item["text"], cat)
            engine._pending_queue = []
            return {"status": "ok", "message": f"已保留全部 {count} 条知识"}
        elif cmd in ("全部忽略", "忽略全部"):
            count = len(engine._pending_queue)
            engine._pending_queue = []
            return {"status": "ok", "message": f"已忽略 {count} 条待确认知识"}
        elif not engine._pending_queue:
            return {"status": "ok", "message": "暂无待确认的知识", "count": 0}
        else:
            lines = [f"待确认知识（{len(engine._pending_queue)}条）"]
            for i, item in enumerate(engine._pending_queue, 1):
                lines.append(f"  {i}. [{item.get('category','')}] {item['text'][:60]}")
            lines.append("操作：/左脑 待确认 全部保留  或  /左脑 待确认 全部忽略")
            return {"status": "ok", "output": "\n".join(lines), "count": len(engine._pending_queue)}

    # ====================== v3.5: WorkBuddy 联动 ======================

    elif action in ("session", "会话"):
        """会话摘要 — v3.6.3: 返回完整上下文（分类/高频/最近/更新/workspace分布）"""
        return engine.session_start()

    elif action in ("inject", "注入"):
        """自动注入相关知识到上下文"""
        return engine.inject_context(content or kwargs.get("text", ""))

    elif action in ("extract", "提取"):
        """从内容中提取知识（搜索结果、文件等）"""
        source = kwargs.get("source", "user")
        return engine.learn_from_content(content, source)

    elif action in ("suggest", "建议"):
        """主动介入建议"""
        return engine.suggest_if_relevant(content or kwargs.get("text", ""))

    return {"status": "unknown", "action": action}


# ====================== CLI入口 ======================
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "--help", "-h"):
        print("🧠 左脑引擎 v3.6 — 记忆 + 推理 + 关联 + 感知增强 四位一体")
        print()
        print("用法: python engine.py <动作> [参数...]")
        print()
        print("记忆类动作:")
        print("  learn <内容>         学习新知识")
        print("  query <内容>         查询知识")
        print("  relate <A,B[,关系]>  建立关联")
        print("  search <内容> [跳数]  图扩散查询")
        print("  correct <内容>       纠错")
        print("  seed [coding|tech|all] 注入种子知识包")
        print("  stats                引擎状态")
        print("  dashboard            ToKen监测助手（Token/时间节省）")
        print("  auto on/off          自动联动模式（和WorkBuddy无缝配合）")
        print("  suggest <文本>        推荐相关知识")
        print("  analysis             使用习惯分析")
        print("  tips                 优化建议")
        print()
        print("推理类动作:")
        print("  analyze <文本>       数据分析（数字/趋势/对比）")
        print("  summarize <文本>     文章总结")
        print("  classify <文本>      体裁分类")
        print("  entangle <词>        纠缠场分析")
        print("  evolve <词|词1,词2>  演化纠缠")
        print()
        sys.exit(0)

    action = sys.argv[1]
    content = sys.argv[2] if len(sys.argv) > 2 else ""
    kwargs = {}
    if action == "search" and len(sys.argv) > 3: kwargs["max_hops"] = sys.argv[3]
    if action == "disambig" and len(sys.argv) > 3: kwargs["context"] = sys.argv[3]

    try:
        result = workbuddy_main(action, content, **kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
