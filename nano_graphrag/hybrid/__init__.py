"""
混合检索模块

提供多源检索融合和并行检索调度功能:
- 并行检索控制器
- 结果融合策略
- 基于置信度的自适应路由
- 自适应权重管理
"""

# 导出所有函数以保持向后兼容
__all__ = [
    # 多源检索
    'ParallelRetriever',
    'TimeoutHandler',
    'RetrievalResult',
    
    # 融合策略
    'FusionStrategy',
    'LinearFusionStrategy',
    'RRFFusionStrategy',
    'AdaptiveFusionStrategy',
    'DynamicWeightFusionStrategy',
    'MultiHeadAttentionFusionStrategy',
    'EnhancedFusionStrategy',
    
    # 混合GraphRAG
    'GraphRAGHybrid',
    'GraphRAGHybridConfig',
    'create_graphrag_hybrid',
    
    # 自适应管理
    'AdaptiveManager',
    'AdaptiveManagerConfig',
    'create_adaptive_manager',
    
    # 归一化
    'NormalizationStrategy',
    'MinMaxNormalizer',
    'ZScoreNormalizer',
    'SoftmaxNormalizer',
    
    # BGE-M3检索器
    'BGEM3Retriever',
    'BGEM3RetrieverConfig',
    'create_bge_m3_retriever',
    'create_bge_m3_config',
    'BGE_M3_AVAILABLE',
    
    # BM25+检索器 (新增)
    'BM25PlusRetriever',
    'BM25PlusConfig',
    'create_bm25_plus_retriever',
    'create_bm25_plus_config',
    'TextProcessor',
    'DocumentIndex',
]

from .multi_retrieval import (
    ParallelRetriever,
    TimeoutHandler,
    RetrievalResult,
)

# 注释掉不存在的导入
# from .fusion_strategies import (
#     FusionStrategy,
#     LinearFusionStrategy,
#     RRFFusionStrategy,
#     AdaptiveFusionStrategy,
#     DynamicWeightFusionStrategy,
#     MultiHeadAttentionFusionStrategy,
#     EnhancedFusionStrategy,
# )

# from .graphrag_hybrid import (
#     GraphRAGHybrid,
#     GraphRAGHybridConfig,
#     create_graphrag_hybrid,
# )

# from .adaptive_management import (
#     AdaptiveManager,
#     AdaptiveManagerConfig,
#     create_adaptive_manager,
# )

# from .normalization import (
#     NormalizationStrategy,
#     MinMaxNormalizer,
#     ZScoreNormalizer,
#     SoftmaxNormalizer,
# )

# 导入BGE-M3检索器
from .bge_m3_retriever import (
    BGEM3Retriever,
    BGEM3RetrieverConfig,
    create_bge_m3_retriever,
    create_bge_m3_config,
    BGE_M3_AVAILABLE,
)

# 导入BM25+检索器
from .bm25_plus_retriever import (
    BM25PlusRetriever,
    BM25PlusConfig,
    create_bm25_plus_retriever,
    create_bm25_plus_config,
    TextProcessor,
    DocumentIndex,
) 