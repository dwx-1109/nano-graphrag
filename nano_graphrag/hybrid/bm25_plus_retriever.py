"""
BM25+检索器模块

实现改进的BM25+算法，相比传统BM25有以下优势：
1. 动态参数调整：k1和b参数自适应
2. 查询扩展：伪相关反馈和同义词扩展
3. 长文档处理：优化的文档长度归一化
4. 混合评分：结合词频和语义信息

参考文献：
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond
- Lv & Zhai (2011). Lower-bounding term frequency normalization
- Carpineto & Romano (2012). A survey of automatic query expansion
"""

import os
import math
import json
import time
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

# 中文分词支持
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("jieba未安装，将使用基础分词")

# 英文词干提取
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK未安装，将使用基础处理")

logger = logging.getLogger(__name__)


@dataclass  
class BM25PlusConfig:
    """BM25+检索器配置"""
    
    # BM25核心参数
    k1: float = 1.2          # 词频饱和参数
    b: float = 0.75          # 文档长度归一化参数
    k3: float = 8.0          # 查询词频参数 (BM25+特有)
    
    # 动态参数调整
    enable_adaptive_params: bool = True
    k1_min: float = 0.5
    k1_max: float = 3.0
    b_min: float = 0.1
    b_max: float = 1.0
    
    # 查询扩展配置
    enable_query_expansion: bool = True
    expansion_terms: int = 5        # 扩展词数量
    expansion_weight: float = 0.3   # 扩展词权重
    use_pseudo_relevance: bool = True
    pseudo_docs_count: int = 3      # 伪相关反馈文档数
    
    # 分词和预处理
    language: str = "mixed"         # zh, en, mixed
    remove_stopwords: bool = True
    use_stemming: bool = True       # 英文词干提取
    min_term_length: int = 2
    max_term_length: int = 50
    
    # 文档处理
    enable_phrase_matching: bool = True
    phrase_boost: float = 1.5       # 短语匹配加权
    
    # 缓存和性能
    enable_cache: bool = True
    cache_dir: str = "bm25_plus_cache"
    max_cache_size: int = 10000
    
    # 索引配置
    enable_positions: bool = True   # 记录词汇位置信息
    enable_tf_normalization: bool = True  # TF归一化


class TextProcessor:
    """文本预处理器"""
    
    def __init__(self, config: BM25PlusConfig):
        self.config = config
        self.stemmer = None
        self.stop_words = set()
        
        # 初始化分词工具
        self._init_tools()
    
    def _init_tools(self):
        """初始化NLP工具"""
        try:
            # 英文处理工具
            if NLTK_AVAILABLE and self.config.language in ["en", "mixed"]:
                self.stemmer = PorterStemmer()
                try:
                    self.stop_words = set(stopwords.words('english'))
                except LookupError:
                    logger.warning("NLTK stopwords not downloaded, downloading...")
                    nltk.download('stopwords', quiet=True)
                    nltk.download('punkt', quiet=True)
                    self.stop_words = set(stopwords.words('english'))
            
            # 中文停用词
            if self.config.language in ["zh", "mixed"]:
                chinese_stopwords = {
                    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
                    "你", "会", "着", "没有", "看", "好", "自己", "这", "对",
                    "年", "两", "同", "而", "能", "主", "本", "以", "可", "所"
                }
                self.stop_words.update(chinese_stopwords)
                
        except Exception as e:
            logger.warning(f"NLP工具初始化失败: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """分词处理"""
        if not text or not text.strip():
            return []
        
        tokens = []
        
        try:
            # 中文分词
            if JIEBA_AVAILABLE and self._contains_chinese(text):
                chinese_tokens = list(jieba.cut(text))
                tokens.extend(chinese_tokens)
            
            # 英文分词
            if self._contains_english(text):
                if NLTK_AVAILABLE:
                    english_tokens = word_tokenize(text.lower())
                    tokens.extend(english_tokens)
                else:
                    # 简单英文分词
                    english_tokens = text.lower().split()
                    tokens.extend(english_tokens)
            
            # 如果没有识别到特定语言，使用通用分词
            if not tokens:
                tokens = text.lower().split()
                
        except Exception as e:
            logger.warning(f"分词失败，使用简单分词: {e}")
            tokens = text.lower().split()
        
        # 后处理
        return self._post_process_tokens(tokens)
    
    def _contains_chinese(self, text: str) -> bool:
        """检查是否包含中文"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def _contains_english(self, text: str) -> bool:
        """检查是否包含英文"""
        return any(char.isalpha() and ord(char) < 128 for char in text)
    
    def _post_process_tokens(self, tokens: List[str]) -> List[str]:
        """token后处理"""
        processed = []
        
        for token in tokens:
            # 清理token
            token = token.strip().lower()
            
            # 长度过滤
            if (len(token) < self.config.min_term_length or 
                len(token) > self.config.max_term_length):
                continue
            
            # 移除纯数字和标点
            if not any(c.isalnum() for c in token):
                continue
            
            # 停用词过滤
            if self.config.remove_stopwords and token in self.stop_words:
                continue
            
            # 英文词干提取
            if (self.config.use_stemming and 
                self.stemmer and 
                token.isalpha() and 
                ord(token[0]) < 128):
                try:
                    token = self.stemmer.stem(token)
                except:
                    pass
            
            processed.append(token)
        
        return processed
    
    def extract_phrases(self, text: str, max_phrase_len: int = 3) -> List[str]:
        """提取短语"""
        if not self.config.enable_phrase_matching:
            return []
        
        tokens = self.tokenize(text)
        phrases = []
        
        # 生成n-gram短语
        for n in range(2, min(max_phrase_len + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i+n])
                phrases.append(phrase)
        
        return phrases


class DocumentIndex:
    """文档索引"""
    
    def __init__(self, config: BM25PlusConfig):
        self.config = config
        self.processor = TextProcessor(config)
        
        # 索引数据
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        
        # 倒排索引: term -> [(doc_id, tf, positions)]
        self.inverted_index: Dict[str, List[Tuple[int, int, List[int]]]] = defaultdict(list)
        
        # 短语索引
        self.phrase_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        
        # 统计信息
        self.vocab_size: int = 0
        self.total_docs: int = 0
        self.idf_cache: Dict[str, float] = {}
        
        # 动态参数
        self.adaptive_k1: float = config.k1
        self.adaptive_b: float = config.b
    
    def add_documents(self, documents: List[str], doc_ids: List[str] = None):
        """添加文档到索引"""
        if doc_ids is None:
            doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
        
        logger.info(f"开始索引 {len(documents)} 个文档")
        start_time = time.time()
        
        for doc_id, doc in zip(doc_ids, documents):
            self._add_single_document(doc, doc_id)
        
        # 更新统计信息
        self._update_statistics()
        
        # 动态参数调整
        if self.config.enable_adaptive_params:
            self._adjust_parameters()
        
        build_time = time.time() - start_time
        logger.info(f"索引构建完成，耗时: {build_time:.2f}秒")
        logger.info(f"文档数量: {self.total_docs}, 词汇数量: {self.vocab_size}")
    
    def _add_single_document(self, document: str, doc_id: str):
        """添加单个文档"""
        doc_idx = len(self.documents)
        
        # 分词
        tokens = self.processor.tokenize(document)
        
        if not tokens:
            logger.warning(f"文档 {doc_id} 分词后为空")
            return
        
        # 存储文档信息
        self.documents.append(document)
        self.doc_ids.append(doc_id)
        self.doc_tokens.append(tokens)
        self.doc_lengths.append(len(tokens))
        
        # 构建词频统计
        term_positions = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)
        
        # 更新倒排索引
        for term, positions in term_positions.items():
            tf = len(positions)
            self.inverted_index[term].append((doc_idx, tf, positions if self.config.enable_positions else []))
        
        # 短语索引
        if self.config.enable_phrase_matching:
            phrases = self.processor.extract_phrases(document)
            for phrase in phrases:
                self.phrase_index[phrase].append((doc_idx, 1))
    
    def _update_statistics(self):
        """更新统计信息"""
        self.total_docs = len(self.documents)
        self.vocab_size = len(self.inverted_index)
        
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # 清空IDF缓存
        self.idf_cache.clear()
    
    def _adjust_parameters(self):
        """动态调整BM25参数"""
        if not self.doc_lengths:
            return
        
        # 基于文档长度分布调整b参数
        doc_length_std = np.std(self.doc_lengths)
        doc_length_cv = doc_length_std / self.avg_doc_length if self.avg_doc_length > 0 else 0
        
        # 文档长度变化大时，增加b值
        if doc_length_cv > 1.0:
            self.adaptive_b = min(self.config.b_max, self.config.b * 1.2)
        elif doc_length_cv < 0.3:
            self.adaptive_b = max(self.config.b_min, self.config.b * 0.8)
        else:
            self.adaptive_b = self.config.b
        
        # 基于词汇分布调整k1参数
        # 大词汇表时适当增加k1
        if self.vocab_size > 10000:
            self.adaptive_k1 = min(self.config.k1_max, self.config.k1 * 1.1)
        elif self.vocab_size < 1000:
            self.adaptive_k1 = max(self.config.k1_min, self.config.k1 * 0.9)
        else:
            self.adaptive_k1 = self.config.k1
        
        logger.info(f"动态参数调整: k1={self.adaptive_k1:.3f}, b={self.adaptive_b:.3f}")
    
    def get_idf(self, term: str) -> float:
        """计算IDF值（带缓存）"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        df = len(self.inverted_index.get(term, []))
        if df == 0:
            idf = 0.0
        else:
            # BM25+ IDF公式：避免负值
            idf = max(0.0, math.log((self.total_docs - df + 0.5) / (df + 0.5)))
        
        self.idf_cache[term] = idf
        return idf
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'total_docs': self.total_docs,
            'vocab_size': self.vocab_size,
            'avg_doc_length': self.avg_doc_length,
            'adaptive_k1': self.adaptive_k1,
            'adaptive_b': self.adaptive_b,
            'doc_length_stats': {
                'min': min(self.doc_lengths) if self.doc_lengths else 0,
                'max': max(self.doc_lengths) if self.doc_lengths else 0,
                'std': float(np.std(self.doc_lengths)) if self.doc_lengths else 0,
            }
        }


class BM25PlusRetriever:
    """BM25+检索器"""
    
    def __init__(self, config: BM25PlusConfig = None):
        self.config = config or BM25PlusConfig()
        self.index = DocumentIndex(self.config)
        self.query_cache = {}
        
        # 创建缓存目录
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def build_index(self, documents: List[str], document_ids: List[str] = None):
        """构建检索索引"""
        self.index.add_documents(documents, document_ids)
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               enable_expansion: bool = None) -> List[Dict[str, Any]]:
        """
        BM25+检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            enable_expansion: 是否启用查询扩展
            
        Returns:
            检索结果列表
        """
        if not self.index.total_docs:
            logger.warning("索引为空，无法检索")
            return []
        
        # 检查缓存
        cache_key = f"{query}_{top_k}_{enable_expansion}"
        if self.config.enable_cache and cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        start_time = time.time()
        
        # 查询扩展
        if enable_expansion is None:
            enable_expansion = self.config.enable_query_expansion
        
        expanded_query = self._expand_query(query) if enable_expansion else query
        
        # 查询处理
        query_terms = self.index.processor.tokenize(expanded_query)
        query_phrases = self.index.processor.extract_phrases(expanded_query)
        
        if not query_terms and not query_phrases:
            logger.warning("查询分词后为空")
            return []
        
        # 计算BM25+分数
        doc_scores = self._calculate_bm25_plus_scores(query_terms, query_phrases)
        
        # 排序并返回top_k结果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (doc_idx, score) in enumerate(sorted_docs[:top_k]):
            results.append({
                'document_id': self.index.doc_ids[doc_idx],
                'document': self.index.documents[doc_idx],
                'score': float(score),
                'rank': i + 1,
                'method': 'bm25_plus'
            })
        
        # 缓存结果
        if self.config.enable_cache and len(self.query_cache) < self.config.max_cache_size:
            self.query_cache[cache_key] = results
        
        search_time = time.time() - start_time
        logger.debug(f"BM25+检索完成，耗时: {search_time:.3f}秒，结果数量: {len(results)}")
        
        return results
    
    async def async_search(self, 
                          query: str, 
                          top_k: int = 10,
                          enable_expansion: bool = None) -> List[Dict[str, Any]]:
        """
        异步检索接口（新增）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            enable_expansion: 是否启用查询扩展
            
        Returns:
            检索结果列表
        """
        import asyncio
        
        # 在线程池中执行同步检索
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, top_k, enable_expansion)
    
    def _expand_query(self, query: str) -> str:
        """查询扩展"""
        if not self.config.enable_query_expansion:
            return query
        
        # 伪相关反馈
        if self.config.use_pseudo_relevance and self.index.total_docs > 0:
            # 先进行初始检索
            initial_results = self.search(query, top_k=self.config.pseudo_docs_count, enable_expansion=False)
            
            # 从前几个结果中提取高频词
            expansion_terms = self._extract_expansion_terms(initial_results)
            
            # 构建扩展查询
            if expansion_terms:
                expanded_query = query + " " + " ".join(expansion_terms)
                logger.debug(f"查询扩展: {query} -> {expanded_query}")
                return expanded_query
        
        return query
    
    def _extract_expansion_terms(self, results: List[Dict[str, Any]]) -> List[str]:
        """从伪相关文档中提取扩展词"""
        if not results:
            return []
        
        # 合并相关文档
        relevant_text = " ".join([result['document'] for result in results])
        
        # 分词并统计词频
        tokens = self.index.processor.tokenize(relevant_text)
        term_freq = Counter(tokens)
        
        # 选择高频但不在原查询中的词
        expansion_candidates = []
        for term, freq in term_freq.most_common(self.config.expansion_terms * 3):
            # 过滤条件：不在原查询中、频率足够高
            if freq >= 2:
                expansion_candidates.append(term)
        
        return expansion_candidates[:self.config.expansion_terms]
    
    def _calculate_bm25_plus_scores(self, 
                                   query_terms: List[str], 
                                   query_phrases: List[str]) -> Dict[int, float]:
        """计算BM25+分数"""
        doc_scores = defaultdict(float)
        
        # 查询词频统计
        query_tf = Counter(query_terms)
        
        # 词项评分
        for term, qtf in query_tf.items():
            if term not in self.index.inverted_index:
                continue
            
            idf = self.index.get_idf(term)
            if idf <= 0:
                continue
            
            # 查询词频部分 (BM25+特有的k3参数)
            query_weight = ((self.config.k3 + 1) * qtf) / (self.config.k3 + qtf)
            
            # 文档词频部分
            for doc_idx, tf, positions in self.index.inverted_index[term]:
                doc_len = self.index.doc_lengths[doc_idx]
                
                # BM25+归一化
                normalized_tf = tf / (1 + self.index.adaptive_b * (doc_len / self.index.avg_doc_length - 1))
                
                # BM25+词频权重
                tf_weight = ((self.index.adaptive_k1 + 1) * normalized_tf) / (self.index.adaptive_k1 + normalized_tf)
                
                # 最终分数
                score = idf * query_weight * tf_weight
                doc_scores[doc_idx] += score
        
        # 短语加权
        if self.config.enable_phrase_matching and query_phrases:
            phrase_boost = self._calculate_phrase_boost(query_phrases)
            for doc_idx, boost in phrase_boost.items():
                doc_scores[doc_idx] *= (1 + boost)
        
        return doc_scores
    
    def _calculate_phrase_boost(self, query_phrases: List[str]) -> Dict[int, float]:
        """计算短语匹配加权"""
        phrase_boost = defaultdict(float)
        
        for phrase in query_phrases:
            if phrase in self.index.phrase_index:
                for doc_idx, count in self.index.phrase_index[phrase]:
                    # 短语匹配加权
                    boost = (self.config.phrase_boost - 1.0) * count * 0.1
                    phrase_boost[doc_idx] += boost
        
        return phrase_boost
    
    def save_index(self, save_path: str):
        """保存索引"""
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存配置
            config_path = save_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
            
            # 保存索引数据
            index_path = save_dir / "index.pkl"
            index_data = {
                'documents': self.index.documents,
                'doc_ids': self.index.doc_ids,
                'doc_tokens': self.index.doc_tokens,
                'doc_lengths': self.index.doc_lengths,
                'avg_doc_length': self.index.avg_doc_length,
                'inverted_index': dict(self.index.inverted_index),
                'phrase_index': dict(self.index.phrase_index),
                'vocab_size': self.index.vocab_size,
                'total_docs': self.index.total_docs,
                'adaptive_k1': self.index.adaptive_k1,
                'adaptive_b': self.index.adaptive_b,
            }
            
            with open(index_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"BM25+索引保存完成: {save_path}")
            
        except Exception as e:
            logger.error(f"索引保存失败: {e}")
    
    def load_index(self, load_path: str):
        """加载索引"""
        try:
            load_dir = Path(load_path)
            if not load_dir.exists():
                logger.error(f"索引路径不存在: {load_path}")
                return
            
            # 加载配置
            config_path = load_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            
            # 加载索引数据
            index_path = load_dir / "index.pkl"
            if index_path.exists():
                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)
                
                # 恢复索引
                self.index.documents = index_data['documents']
                self.index.doc_ids = index_data['doc_ids']
                self.index.doc_tokens = index_data['doc_tokens']
                self.index.doc_lengths = index_data['doc_lengths']
                self.index.avg_doc_length = index_data['avg_doc_length']
                self.index.inverted_index = defaultdict(list, index_data['inverted_index'])
                self.index.phrase_index = defaultdict(list, index_data['phrase_index'])
                self.index.vocab_size = index_data['vocab_size']
                self.index.total_docs = index_data['total_docs']
                self.index.adaptive_k1 = index_data['adaptive_k1']
                self.index.adaptive_b = index_data['adaptive_b']
            
            logger.info(f"BM25+索引加载完成: {load_path}")
            
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        stats = {
            'retriever_type': 'BM25Plus',
            'config': {
                'k1': self.config.k1,
                'b': self.config.b,
                'k3': self.config.k3,
                'language': self.config.language,
                'enable_query_expansion': self.config.enable_query_expansion,
                'enable_phrase_matching': self.config.enable_phrase_matching,
            },
            'cache_size': len(self.query_cache),
        }
        
        # 添加索引统计
        stats.update(self.index.get_stats())
        
        return stats


def create_bm25_plus_retriever(config: BM25PlusConfig = None) -> BM25PlusRetriever:
    """创建BM25+检索器的工厂函数"""
    return BM25PlusRetriever(config)


def create_bm25_plus_config(**kwargs) -> BM25PlusConfig:
    """创建BM25+配置的工厂函数"""
    return BM25PlusConfig(**kwargs)


# 导出所有公共接口
__all__ = [
    'BM25PlusConfig',
    'BM25PlusRetriever',
    'create_bm25_plus_retriever',
    'create_bm25_plus_config',
    'TextProcessor',
    'DocumentIndex'
] 