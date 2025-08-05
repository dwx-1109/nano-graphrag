"""
并行检索管理器

解决多源检索的性能瓶颈，实现真正的异步并行检索。
基于现有的BGE-M3和BM25+检索器进行集成。
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ..cache.lru_cache import LRUCache

logger = logging.getLogger(__name__)

@dataclass
class ParallelRetrievalConfig:
    """并行检索配置"""
    
    # 并发控制
    max_concurrent_retrievers: int = 4
    retrieval_timeout: float = 30.0
    enable_timeout: bool = True
    
    # 缓存配置
    enable_result_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1小时
    
    # 性能监控
    enable_performance_tracking: bool = True
    log_slow_retrievals: bool = True
    slow_retrieval_threshold: float = 5.0  # 5秒
    
    # 错误处理
    max_retries: int = 2
    retry_delay: float = 1.0
    continue_on_partial_failure: bool = True
    
    # 结果融合
    enable_result_deduplication: bool = True
    similarity_threshold: float = 0.8

class ParallelRetrievalManager:
    """
    并行检索管理器
    
    核心功能：
    1. 异步并行执行多个检索器
    2. 超时控制和错误恢复
    3. 结果缓存和性能监控
    4. 结果去重和融合
    """
    
    def __init__(self, config: ParallelRetrievalConfig = None):
        self.config = config or ParallelRetrievalConfig()
        self.retrievers: Dict[str, Any] = {}
        self.performance_stats = {}
        
        # 初始化缓存
        if self.config.enable_result_cache:
            self.result_cache = LRUCache(
                max_size=self.config.cache_size,
                ttl=self.config.cache_ttl
            )
        else:
            self.result_cache = None
        
        # 线程池（用于同步检索器的异步包装）
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_retrievers
        )
        
        logger.info("并行检索管理器初始化完成")
    
    def register_retriever(self, 
                          name: str, 
                          retriever: Any, 
                          weight: float = 1.0,
                          enabled: bool = True):
        """
        注册检索器
        
        Args:
            name: 检索器名称
            retriever: 检索器实例
            weight: 检索器权重
            enabled: 是否启用
        """
        self.retrievers[name] = {
            'instance': retriever,
            'weight': weight,
            'enabled': enabled,
            'stats': {
                'total_calls': 0,
                'total_time': 0.0,
                'error_count': 0,
                'cache_hits': 0
            }
        }
        logger.info(f"注册检索器: {name}, 权重: {weight}, 启用: {enabled}")
    
    async def parallel_search(self, 
                            query: str, 
                            top_k: int = 10,
                            retriever_names: Optional[List[str]] = None,
                            **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        并行检索
        
        Args:
            query: 查询文本
            top_k: 每个检索器返回的结果数量
            retriever_names: 指定使用的检索器名称列表
            **kwargs: 传递给检索器的额外参数
            
        Returns:
            各检索器的结果字典
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"{query}_{top_k}_{sorted(retriever_names or [])}"
        if self.result_cache and self.result_cache.contains(cache_key):
            logger.debug(f"缓存命中: {cache_key}")
            return self.result_cache.get(cache_key)
        
        # 确定要使用的检索器
        if retriever_names is None:
            retriever_names = [name for name, info in self.retrievers.items() 
                             if info['enabled']]
        
        if not retriever_names:
            logger.warning("没有可用的检索器")
            return {}
        
        # 创建异步任务
        tasks = []
        for name in retriever_names:
            if name not in self.retrievers:
                logger.warning(f"检索器 {name} 未注册")
                continue
            
            task = asyncio.create_task(
                self._single_retriever_search(name, query, top_k, **kwargs)
            )
            tasks.append((name, task))
        
        # 执行并行检索
        results = {}
        completed_tasks = 0
        
        try:
            # 等待所有任务完成（带超时）
            if self.config.enable_timeout:
                done, pending = await asyncio.wait(
                    [task for _, task in tasks],
                    timeout=self.config.retrieval_timeout,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # 取消未完成的任务
                for task in pending:
                    task.cancel()
                    logger.warning("检索任务超时被取消")
            else:
                done = [task for _, task in tasks]
            
            # 收集结果
            for name, task in tasks:
                if task in done:
                    try:
                        result = await task
                        results[name] = result
                        completed_tasks += 1
                    except Exception as e:
                        logger.error(f"检索器 {name} 执行失败: {e}")
                        results[name] = []
                        self.retrievers[name]['stats']['error_count'] += 1
                else:
                    results[name] = []
                    logger.warning(f"检索器 {name} 未完成")
        
        except Exception as e:
            logger.error(f"并行检索失败: {e}")
            return {}
        
        # 记录性能统计
        total_time = time.time() - start_time
        if self.config.enable_performance_tracking:
            self.performance_stats[cache_key] = {
                'total_time': total_time,
                'completed_tasks': completed_tasks,
                'total_tasks': len(tasks),
                'timestamp': time.time()
            }
            
            if (self.config.log_slow_retrievals and 
                total_time > self.config.slow_retrieval_threshold):
                logger.warning(f"慢检索检测: {total_time:.2f}s > {self.config.slow_retrieval_threshold}s")
        
        # 缓存结果
        if self.result_cache:
            self.result_cache.put(cache_key, results)
        
        logger.info(f"并行检索完成: {completed_tasks}/{len(tasks)} 成功, 耗时 {total_time:.2f}s")
        return results
    
    async def _single_retriever_search(self, 
                                     name: str, 
                                     query: str, 
                                     top_k: int,
                                     **kwargs) -> List[Dict[str, Any]]:
        """
        单个检索器搜索（带重试）
        
        Args:
            name: 检索器名称
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 额外参数
            
        Returns:
            检索结果列表
        """
        retriever_info = self.retrievers[name]
        retriever = retriever_info['instance']
        
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                # 检查检索器是否有异步接口
                if hasattr(retriever, 'async_search'):
                    results = await retriever.async_search(query, top_k=top_k, **kwargs)
                elif hasattr(retriever, 'search'):
                    # 同步接口，使用线程池包装
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        self.thread_pool, 
                        retriever.search, 
                        query, 
                        top_k
                    )
                else:
                    logger.error(f"检索器 {name} 没有可用的搜索接口")
                    return []
                
                # 更新统计
                elapsed_time = time.time() - start_time
                retriever_info['stats']['total_calls'] += 1
                retriever_info['stats']['total_time'] += elapsed_time
                
                return results if results else []
                
            except Exception as e:
                logger.warning(f"检索器 {name} 第 {attempt + 1} 次尝试失败: {e}")
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    retriever_info['stats']['error_count'] += 1
                    if self.config.continue_on_partial_failure:
                        return []
                    else:
                        raise
        
        return []
    
    def fuse_results(self, 
                    results: Dict[str, List[Dict[str, Any]]], 
                    fusion_method: str = "weighted_rrf") -> List[Dict[str, Any]]:
        """
        融合多源检索结果
        
        Args:
            results: 各检索器的结果
            fusion_method: 融合方法 ("weighted_rrf", "simple_concat", "score_based")
            
        Returns:
            融合后的结果列表
        """
        if not results:
            return []
        
        try:
            if fusion_method == "weighted_rrf":
                return self._weighted_reciprocal_rank_fusion(results)
            elif fusion_method == "simple_concat":
                return self._simple_concatenation(results)
            elif fusion_method == "score_based":
                return self._score_based_fusion(results)
            else:
                logger.warning(f"未知融合方法: {fusion_method}, 使用加权RRF")
                return self._weighted_reciprocal_rank_fusion(results)
                
        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            # 回退到简单连接
            return self._simple_concatenation(results)
    
    def _weighted_reciprocal_rank_fusion(self, 
                                       results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """加权倒数排名融合"""
        k = 60  # RRF参数
        doc_scores = {}
        
        for retriever_name, docs in results.items():
            if not docs:
                continue
                
            weight = self.retrievers[retriever_name]['weight']
            
            for rank, doc in enumerate(docs):
                doc_id = doc.get('id', doc.get('content', str(rank)))
                rrf_score = weight / (k + rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'score': 0.0,
                        'sources': []
                    }
                
                doc_scores[doc_id]['score'] += rrf_score
                doc_scores[doc_id]['sources'].append(retriever_name)
        
        # 按分数排序
        sorted_docs = sorted(doc_scores.items(), 
                           key=lambda x: x[1]['score'], 
                           reverse=True)
        
        # 构建最终结果
        fused_results = []
        for doc_id, info in sorted_docs:
            doc = info['doc'].copy()
            doc['fusion_score'] = info['score']
            doc['sources'] = info['sources']
            fused_results.append(doc)
        
        return fused_results
    
    def _simple_concatenation(self, 
                            results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """简单连接融合"""
        all_results = []
        
        for retriever_name, docs in results.items():
            for doc in docs:
                doc_copy = doc.copy()
                doc_copy['source'] = retriever_name
                all_results.append(doc_copy)
        
        return all_results
    
    def _score_based_fusion(self, 
                          results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """基于分数的融合"""
        all_results = []
        
        for retriever_name, docs in results.items():
            weight = self.retrievers[retriever_name]['weight']
            
            for doc in docs:
                doc_copy = doc.copy()
                original_score = doc.get('score', 0.0)
                doc_copy['weighted_score'] = original_score * weight
                doc_copy['source'] = retriever_name
                all_results.append(doc_copy)
        
        # 按加权分数排序
        all_results.sort(key=lambda x: x.get('weighted_score', 0.0), reverse=True)
        return all_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            'retrievers': {},
            'cache_stats': {},
            'recent_queries': self.performance_stats
        }
        
        # 检索器统计
        for name, info in self.retrievers.items():
            retriever_stats = info['stats'].copy()
            if retriever_stats['total_calls'] > 0:
                retriever_stats['avg_time'] = retriever_stats['total_time'] / retriever_stats['total_calls']
                retriever_stats['error_rate'] = retriever_stats['error_count'] / retriever_stats['total_calls']
            else:
                retriever_stats['avg_time'] = 0.0
                retriever_stats['error_rate'] = 0.0
            
            stats['retrievers'][name] = retriever_stats
        
        # 缓存统计
        if self.result_cache:
            cache_stats = self.result_cache.get_stats()
            stats['cache_stats'] = cache_stats
        
        return stats
    
    def clear_cache(self):
        """清除缓存"""
        if self.result_cache:
            self.result_cache.clear()
            logger.info("检索结果缓存已清除")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

def create_parallel_retrieval_manager(config: ParallelRetrievalConfig = None) -> ParallelRetrievalManager:
    """创建并行检索管理器的工厂函数"""
    return ParallelRetrievalManager(config) 