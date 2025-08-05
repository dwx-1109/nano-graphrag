"""
BGE-M3多功能检索器

基于BAAI/bge-m3模型实现的高级检索器，支持：
- Dense Retrieval: 密集向量检索
- Sparse Retrieval: 稀疏关键词检索  
- Multi-Vector Retrieval: ColBERT风格的细粒度检索
- Hybrid Ranking: 多种检索方法的组合

参考文献：
- Chen et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, 
  Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216

模型特点：
- 支持100+种语言
- 最大输入长度8192 tokens
- 模型大小：569M参数，2.27GB
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

# 检查FlagEmbedding依赖
BGE_M3_AVAILABLE = False
try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    print("FlagEmbedding未安装，将使用基础检索器")
    print("请安装: pip install FlagEmbedding")

import faiss

logger = logging.getLogger(__name__)


@dataclass
class BGEM3RetrieverConfig:
    """BGE-M3检索器配置"""
    
    # 模型配置
    model_path: str = "nano_graphrag/models/bge-m3"
    model_name_or_path: str = "BAAI/bge-m3"  # 在线模型路径
    use_local_model: bool = True  # 优先使用本地模型
    
    # 检索功能开关
    enable_dense: bool = True      # 密集检索
    enable_sparse: bool = True     # 稀疏检索
    enable_colbert: bool = True    # ColBERT多向量检索
    
    # 密集检索配置
    dense_embedding_dim: int = 1024  # BGE-M3嵌入维度
    dense_batch_size: int = 32
    dense_normalize: bool = True
    
    # 稀疏检索配置
    sparse_top_k: int = 100
    sparse_threshold: float = 0.01
    
    # ColBERT配置  
    colbert_dim: int = 1024
    colbert_batch_size: int = 16
    
    # 混合检索权重
    dense_weight: float = 0.4
    sparse_weight: float = 0.2
    colbert_weight: float = 0.4
    
    # FAISS配置
    faiss_index_type: str = "IVF"  # Flat, IVF, HNSW
    faiss_nlist: int = 1024
    faiss_nprobe: int = 32
    
    # 性能配置
    max_seq_length: int = 8192  # BGE-M3最大支持长度
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads: int = 4
    
    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "bge_m3_cache"


class BGEM3Retriever:
    """
    BGE-M3多功能检索器
    
    集成了BGE-M3的三种检索模式：
    1. Dense Retrieval: 使用[CLS]标记的归一化隐藏状态
    2. Sparse Retrieval: 基于词项权重的稀疏表示
    3. Multi-Vector Retrieval: ColBERT风格的细粒度匹配
    """
    
    def __init__(self, config: BGEM3RetrieverConfig = None):
        self.config = config or BGEM3RetrieverConfig()
        self.model = None
        self.dense_index = None
        self.sparse_index = None
        self.documents = []
        self.document_embeddings = {}
        
        # 检查依赖
        if not BGE_M3_AVAILABLE:
            logger.warning("FlagEmbedding未安装，检索器功能受限")
            logger.warning("请安装: pip install FlagEmbedding")
            return
        
        # 初始化模型
        self._load_model()
        
        # 创建缓存目录
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def _load_model(self):
        """加载BGE-M3模型"""
        if not BGE_M3_AVAILABLE:
            return
        
        try:
            # 优先尝试本地模型
            if self.config.use_local_model and os.path.exists(self.config.model_path):
                logger.info(f"加载本地BGE-M3模型: {self.config.model_path}")
                self.model = BGEM3FlagModel(
                    self.config.model_path,
                    use_fp16=True if self.config.device == "cuda" else False,
                    device=self.config.device
                )
            else:
                # 使用在线模型
                logger.info(f"加载在线BGE-M3模型: {self.config.model_name_or_path}")
                self.model = BGEM3FlagModel(
                    self.config.model_name_or_path,
                    use_fp16=True if self.config.device == "cuda" else False,
                    device=self.config.device
                )
            
            logger.info("BGE-M3模型加载成功")
            
        except Exception as e:
            logger.error(f"BGE-M3模型加载失败: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """检查检索器是否可用"""
        return BGE_M3_AVAILABLE and self.model is not None
    
    def encode_queries(self, 
                      queries: Union[str, List[str]], 
                      **kwargs) -> Dict[str, Any]:
        """
        编码查询文本
        
        Args:
            queries: 查询文本或文本列表
            **kwargs: 额外参数
            
        Returns:
            包含不同类型嵌入的字典
        """
        if not self.is_available():
            raise RuntimeError("BGE-M3模型未加载")
        
        if isinstance(queries, str):
            queries = [queries]
        
        try:
            # 使用BGE-M3编码
            output = self.model.encode(
                queries,
                batch_size=self.config.dense_batch_size,
                max_length=self.config.max_seq_length,
                return_dense=self.config.enable_dense,
                return_sparse=self.config.enable_sparse,
                return_colbert_vecs=self.config.enable_colbert,
                **kwargs
            )
            
            results = {}
            
            # 密集嵌入
            if self.config.enable_dense and 'dense_vecs' in output:
                dense_vecs = output['dense_vecs']
                if self.config.dense_normalize:
                    dense_vecs = dense_vecs / np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                results['dense'] = dense_vecs
            
            # 稀疏权重  
            if self.config.enable_sparse and 'lexical_weights' in output:
                results['sparse'] = output['lexical_weights']
            
            # ColBERT向量
            if self.config.enable_colbert and 'colbert_vecs' in output:
                results['colbert'] = output['colbert_vecs']
            
            return results
            
        except Exception as e:
            logger.error(f"查询编码失败: {e}")
            return {}
    
    def encode_documents(self, 
                        documents: List[str],
                        **kwargs) -> Dict[str, Any]:
        """
        编码文档文本
        
        Args:
            documents: 文档列表
            **kwargs: 额外参数
            
        Returns:
            包含不同类型嵌入的字典
        """
        if not self.is_available():
            raise RuntimeError("BGE-M3模型未加载")
        
        try:
            # 批量编码文档
            output = self.model.encode(
                documents,
                batch_size=self.config.dense_batch_size,
                max_length=self.config.max_seq_length,
                return_dense=self.config.enable_dense,
                return_sparse=self.config.enable_sparse,
                return_colbert_vecs=self.config.enable_colbert,
                **kwargs
            )
            
            results = {}
            
            # 密集嵌入
            if self.config.enable_dense and 'dense_vecs' in output:
                dense_vecs = output['dense_vecs']
                if self.config.dense_normalize:
                    dense_vecs = dense_vecs / np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                results['dense'] = dense_vecs
            
            # 稀疏权重
            if self.config.enable_sparse and 'lexical_weights' in output:
                results['sparse'] = output['lexical_weights']
            
            # ColBERT向量
            if self.config.enable_colbert and 'colbert_vecs' in output:
                results['colbert'] = output['colbert_vecs']
            
            return results
            
        except Exception as e:
            logger.error(f"文档编码失败: {e}")
            return {}
    
    def build_index(self, documents: List[str], document_ids: List[str] = None):
        """
        构建检索索引
        
        Args:
            documents: 文档列表
            document_ids: 文档ID列表，如果为None则自动生成
        """
        if not self.is_available():
            logger.error("BGE-M3模型未加载，无法构建索引")
            return
        
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        logger.info(f"开始构建BGE-M3索引，文档数量: {len(documents)}")
        start_time = time.time()
        
        # 保存文档
        self.documents = documents
        
        try:
            # 编码所有文档
            embeddings = self.encode_documents(documents)
            self.document_embeddings = embeddings
            
            # 构建密集索引
            if self.config.enable_dense and 'dense' in embeddings:
                self._build_dense_index(embeddings['dense'])
            
            # 构建稀疏索引
            if self.config.enable_sparse and 'sparse' in embeddings:
                self._build_sparse_index(embeddings['sparse'])
            
            # ColBERT索引构建较复杂，这里简化处理
            if self.config.enable_colbert and 'colbert' in embeddings:
                logger.info("ColBERT向量已保存，将在检索时进行实时计算")
            
            build_time = time.time() - start_time
            logger.info(f"BGE-M3索引构建完成，耗时: {build_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
    
    def _build_dense_index(self, dense_embeddings: np.ndarray):
        """构建密集向量索引"""
        try:
            dim = dense_embeddings.shape[1]
            
            if self.config.faiss_index_type == "Flat":
                # 精确搜索
                self.dense_index = faiss.IndexFlatIP(dim)  # 内积索引
            elif self.config.faiss_index_type == "IVF":
                # IVF索引
                quantizer = faiss.IndexFlatIP(dim)
                self.dense_index = faiss.IndexIVFFlat(quantizer, dim, self.config.faiss_nlist)
                self.dense_index.train(dense_embeddings)
                self.dense_index.nprobe = self.config.faiss_nprobe
            elif self.config.faiss_index_type == "HNSW":
                # HNSW索引
                self.dense_index = faiss.IndexHNSWFlat(dim, 32)
                self.dense_index.hnsw.efConstruction = 200
                self.dense_index.hnsw.efSearch = 50
            else:
                # 默认使用Flat
                self.dense_index = faiss.IndexFlatIP(dim)
            
            # 添加向量到索引
            self.dense_index.add(dense_embeddings.astype(np.float32))
            logger.info(f"密集索引构建完成，索引类型: {self.config.faiss_index_type}, 向量数量: {dense_embeddings.shape[0]}")
            
        except Exception as e:
            logger.error(f"密集索引构建失败: {e}")
    
    def _build_sparse_index(self, sparse_weights: List[Dict[str, float]]):
        """构建稀疏索引（简化版本）"""
        try:
            # 这里简化处理，实际应该构建倒排索引
            self.sparse_index = sparse_weights
            logger.info(f"稀疏索引构建完成，文档数量: {len(sparse_weights)}")
        except Exception as e:
            logger.error(f"稀疏索引构建失败: {e}")
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        执行检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            method: 检索方法 ("dense", "sparse", "colbert", "hybrid")
            
        Returns:
            检索结果列表
        """
        if not self.is_available():
            logger.error("BGE-M3模型未加载")
            return []
        
        if not self.documents:
            logger.error("未构建索引")
            return []
        
        try:
            # 编码查询
            query_embeddings = self.encode_queries([query])
            
            if method == "dense":
                return self._dense_search(query_embeddings, top_k)
            elif method == "sparse":
                return self._sparse_search(query_embeddings, top_k)
            elif method == "colbert":
                return self._colbert_search(query_embeddings, top_k)
            elif method == "hybrid":
                return self._hybrid_search(query_embeddings, top_k)
            else:
                logger.warning(f"未知检索方法: {method}, 使用混合检索")
                return self._hybrid_search(query_embeddings, top_k)
                
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    async def async_search(self, 
                          query: str, 
                          top_k: int = 10,
                          method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        异步检索接口（新增）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            method: 检索方法
            
        Returns:
            检索结果列表
        """
        import asyncio
        
        # 在线程池中执行同步检索
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, top_k, method)
    
    def _dense_search(self, query_embeddings: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """密集检索"""
        if not self.config.enable_dense or 'dense' not in query_embeddings:
            return []
        
        if self.dense_index is None:
            logger.error("密集索引未构建")
            return []
        
        try:
            query_vec = query_embeddings['dense'][0].astype(np.float32)
            scores, indices = self.dense_index.search(query_vec.reshape(1, -1), top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # 有效索引
                    results.append({
                        'document_id': f"doc_{idx}",
                        'document': self.documents[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'method': 'dense'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"密集检索失败: {e}")
            return []
    
    def _sparse_search(self, query_embeddings: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """稀疏检索"""
        if not self.config.enable_sparse or 'sparse' not in query_embeddings:
            return []
        
        if self.sparse_index is None:
            logger.error("稀疏索引未构建")
            return []
        
        try:
            query_weights = query_embeddings['sparse'][0]
            scores = []
            
            # 计算与每个文档的稀疏相似度
            for doc_idx, doc_weights in enumerate(self.sparse_index):
                score = 0.0
                # 计算共现词项的权重乘积和
                for term, q_weight in query_weights.items():
                    if term in doc_weights:
                        score += q_weight * doc_weights[term]
                scores.append((score, doc_idx))
            
            # 排序并返回top_k
            scores.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for i, (score, idx) in enumerate(scores[:top_k]):
                if score > self.config.sparse_threshold:
                    results.append({
                        'document_id': f"doc_{idx}",
                        'document': self.documents[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'method': 'sparse'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"稀疏检索失败: {e}")
            return []
    
    def _colbert_search(self, query_embeddings: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """ColBERT多向量检索"""
        if not self.config.enable_colbert or 'colbert' not in query_embeddings:
            return []
        
        if 'colbert' not in self.document_embeddings:
            logger.error("ColBERT文档嵌入未构建")
            return []
        
        try:
            query_vecs = query_embeddings['colbert'][0]  # [seq_len, hidden_dim]
            doc_vecs_list = self.document_embeddings['colbert']
            
            scores = []
            
            # 对每个文档计算ColBERT相似度
            for doc_idx, doc_vecs in enumerate(doc_vecs_list):
                # 计算最大相似度矩阵
                sim_matrix = np.dot(query_vecs, doc_vecs.T)  # [q_len, d_len]
                max_sims = np.max(sim_matrix, axis=1)  # [q_len]
                score = np.mean(max_sims)  # ColBERT平均最大相似度
                scores.append((score, doc_idx))
            
            # 排序并返回top_k
            scores.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for i, (score, idx) in enumerate(scores[:top_k]):
                results.append({
                    'document_id': f"doc_{idx}",
                    'document': self.documents[idx],
                    'score': float(score),
                    'rank': i + 1,
                    'method': 'colbert'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"ColBERT检索失败: {e}")
            return []
    
    def _hybrid_search(self, query_embeddings: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """混合检索"""
        try:
            # 获取各种检索结果
            dense_results = self._dense_search(query_embeddings, top_k * 2)
            sparse_results = self._sparse_search(query_embeddings, top_k * 2)
            colbert_results = self._colbert_search(query_embeddings, top_k * 2)
            
            # 合并分数
            doc_scores = {}
            
            # 加权合并分数
            for results, weight in [
                (dense_results, self.config.dense_weight),
                (sparse_results, self.config.sparse_weight),
                (colbert_results, self.config.colbert_weight)
            ]:
                for result in results:
                    doc_id = result['document_id']
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            'document': result['document'],
                            'scores': {},
                            'final_score': 0.0
                        }
                    doc_scores[doc_id]['scores'][result['method']] = result['score']
                    doc_scores[doc_id]['final_score'] += result['score'] * weight
            
            # 排序并构建最终结果
            sorted_docs = sorted(doc_scores.items(), 
                               key=lambda x: x[1]['final_score'], 
                               reverse=True)
            
            results = []
            for i, (doc_id, doc_info) in enumerate(sorted_docs[:top_k]):
                results.append({
                    'document_id': doc_id,
                    'document': doc_info['document'],
                    'score': doc_info['final_score'],
                    'rank': i + 1,
                    'method': 'hybrid',
                    'detail_scores': doc_info['scores']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []
    
    def save_index(self, save_path: str):
        """保存索引"""
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存配置
            config_path = save_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
            
            # 保存密集索引
            if self.dense_index is not None:
                dense_path = save_dir / "dense_index.faiss"
                faiss.write_index(self.dense_index, str(dense_path))
            
            # 保存稀疏索引
            if self.sparse_index is not None:
                sparse_path = save_dir / "sparse_index.json"
                with open(sparse_path, 'w', encoding='utf-8') as f:
                    json.dump(self.sparse_index, f, indent=2, ensure_ascii=False)
            
            # 保存文档和嵌入
            docs_path = save_dir / "documents.json"
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            
            logger.info(f"索引保存完成: {save_path}")
            
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
            
            # 加载密集索引
            dense_path = load_dir / "dense_index.faiss"
            if dense_path.exists():
                self.dense_index = faiss.read_index(str(dense_path))
            
            # 加载稀疏索引
            sparse_path = load_dir / "sparse_index.json"
            if sparse_path.exists():
                with open(sparse_path, 'r', encoding='utf-8') as f:
                    self.sparse_index = json.load(f)
            
            # 加载文档
            docs_path = load_dir / "documents.json"
            if docs_path.exists():
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            logger.info(f"索引加载完成: {load_path}")
            
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        stats = {
            'model_available': self.is_available(),
            'model_path': self.config.model_path,
            'document_count': len(self.documents),
            'config': {
                'enable_dense': self.config.enable_dense,
                'enable_sparse': self.config.enable_sparse,
                'enable_colbert': self.config.enable_colbert,
                'dense_weight': self.config.dense_weight,
                'sparse_weight': self.config.sparse_weight,
                'colbert_weight': self.config.colbert_weight,
            }
        }
        
        if self.dense_index is not None:
            stats['dense_index_size'] = self.dense_index.ntotal
            
        if self.sparse_index is not None:
            stats['sparse_index_size'] = len(self.sparse_index)
        
        return stats


def create_bge_m3_retriever(config: BGEM3RetrieverConfig = None) -> BGEM3Retriever:
    """创建BGE-M3检索器的工厂函数"""
    return BGEM3Retriever(config)


def create_bge_m3_config(**kwargs) -> BGEM3RetrieverConfig:
    """创建BGE-M3配置的工厂函数"""
    return BGEM3RetrieverConfig(**kwargs)


# 导出所有公共接口
__all__ = [
    'BGEM3RetrieverConfig',
    'BGEM3Retriever',
    'create_bge_m3_retriever',
    'create_bge_m3_config',
    'BGE_M3_AVAILABLE'
] 