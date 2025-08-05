"""
并行检索实现模块

提供多源检索的并行调度与执行功能
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, TypeVar, Awaitable

from .._utils import logger

T = TypeVar('T')


class TimeoutHandler:
    """超时处理器"""
    
    @staticmethod
    async def with_timeout(
        coro: Awaitable[T], 
        timeout: float, 
        default_value: Any = None
    ) -> T:
        """带超时的异步执行
        
        Args:
            coro: 异步协程
            timeout: 超时时间(秒)
            default_value: 超时时返回的默认值
            
        Returns:
            协程结果或默认值（如果超时）
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"操作超时 ({timeout}秒)")
            return default_value
    
    @staticmethod
    async def gather_with_concurrency(
        concurrency: int, 
        *tasks, 
        return_exceptions: bool = False
    ):
        """限制并发数的异步任务收集器
        
        Args:
            concurrency: 最大并发数
            tasks: 异步任务列表
            return_exceptions: 是否返回异常而不是抛出
            
        Returns:
            任务结果列表
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def sem_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(
            *(sem_task(task) for task in tasks), 
            return_exceptions=return_exceptions
        )


class RetrievalResult:
    """检索结果"""
    
    def __init__(
        self, 
        documents: List[Dict[str, Any]],
        retriever_name: str,
        query: str,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """初始化检索结果
        
        Args:
            documents: 检索到的文档列表
            retriever_name: 检索器名称
            query: 查询文本
            metrics: 检索指标
        """
        self.documents = documents
        self.retriever_name = retriever_name
        self.query = query
        self.metrics = metrics or {}
        
    @property
    def count(self) -> int:
        """获取检索结果数量"""
        return len(self.documents)
    
    def __repr__(self) -> str:
        return f"RetrievalResult(retriever={self.retriever_name}, count={self.count}, metrics={self.metrics})"


class ParallelRetriever:
    """并行检索控制器"""
    
    def __init__(
        self,
        retrievers: Dict[str, Callable],
        max_concurrency: int = 5,
        timeout: Optional[float] = 30.0,
        default_top_k: int = 10
    ):
        """初始化并行检索控制器
        
        Args:
            retrievers: 检索器字典，键为检索器名称，值为检索函数
            max_concurrency: 最大并发数
            timeout: 检索超时时间(秒)
            default_top_k: 默认检索文档数
        """
        self.retrievers = retrievers
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.default_top_k = default_top_k
    
    def add_retriever(self, name: str, retriever_func: Callable):
        """添加检索器
        
        Args:
            name: 检索器名称
            retriever_func: 检索函数，应接受query和top_k参数
        """
        self.retrievers[name] = retriever_func
    
    def remove_retriever(self, name: str):
        """移除检索器
        
        Args:
            name: 检索器名称
        """
        if name in self.retrievers:
            del self.retrievers[name]
    
    async def retrieve_all(
        self,
        query: str,
        retriever_names: Optional[List[str]] = None,
        top_k: Union[int, Dict[str, int]] = None,
        **kwargs
    ) -> Dict[str, RetrievalResult]:
        """并行执行多个检索器
        
        Args:
            query: 查询文本
            retriever_names: 要使用的检索器名称列表，如果为None则使用所有检索器
            top_k: 每个检索器返回的最大文档数，可以是整数或每个检索器的字典
            **kwargs: 传递给检索器的其他参数
            
        Returns:
            检索结果字典，键为检索器名称，值为检索结果
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.default_top_k
        
        # 确定要使用的检索器
        if retriever_names:
            retrievers = {
                name: self.retrievers[name]
                for name in retriever_names
                if name in self.retrievers
            }
        else:
            retrievers = self.retrievers
        
        if not retrievers:
            logger.warning("没有可用的检索器")
            return {}
        
        # 创建检索任务
        tasks = {}
        for name, retriever_func in retrievers.items():
            # 确定top_k
            if isinstance(top_k, dict):
                k = top_k.get(name, self.default_top_k)
            else:
                k = top_k
            
            # 创建检索任务
            if self.timeout:
                task = TimeoutHandler.with_timeout(
                    retriever_func(query, top_k=k, **kwargs),
                    self.timeout,
                    default_value=RetrievalResult(
                        documents=[],
                        retriever_name=name,
                        query=query,
                        metrics={"error": "timeout"}
                    )
                )
            else:
                task = retriever_func(query, top_k=k, **kwargs)
            
            tasks[name] = task
        
        # 并行执行任务
        results = {}
        completed_tasks = await TimeoutHandler.gather_with_concurrency(
            self.max_concurrency,
            *tasks.values(),
            return_exceptions=True
        )
        
        # 处理结果
        for name, task_result in zip(tasks.keys(), completed_tasks):
            if isinstance(task_result, Exception):
                logger.error(f"检索器{name}执行失败: {str(task_result)}")
                results[name] = RetrievalResult(
                    documents=[],
                    retriever_name=name,
                    query=query,
                    metrics={"error": str(task_result)}
                )
            else:
                results[name] = task_result
        
        # 记录总执行时间
        elapsed = time.time() - start_time
        logger.info(f"多检索器并行执行完成，耗时{elapsed:.4f}秒")
        
        return results 