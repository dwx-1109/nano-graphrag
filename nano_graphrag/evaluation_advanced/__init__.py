"""
现代评估模块 (整合版)

包含：
- Modern Evaluator: 基于RAGAS的现代RAG评估器
- Comprehensive Evaluator: 综合评估器（完整功能）
- Basic Evaluator: 传统检索评估器 (已从evaluation模块整合，原模块已清理)
- 各种评估指标和工具
"""

# ============= 现代评估器 (主要推荐) =============
from .modern_evaluator import (
    ModernEvaluator,
    ModernEvaluatorConfig,
    RAGASEvaluator,
    SemanticEvaluator,
    ComplexityAwareEvaluator,
    create_modern_evaluator,
    create_modern_evaluator_config,
)

# ============= 综合评估器 (完整功能) =============
from .comprehensive_evaluator import (
    ComprehensiveEvaluator,
    ComprehensiveEvaluatorConfig,
    BLEUEvaluator,
    ROUGEEvaluator,
    BERTScoreEvaluator,
    NDCGEvaluator,
    MRREvaluator,
    PrecisionRecallEvaluator,
    StatisticalTester,
    EvaluationReport,
    create_comprehensive_evaluator,
    create_comprehensive_evaluator_config,
)

# ============= 基线对比器 =============
from .baseline_comparator import (
    BaselineComparator,
    BaselineConfig,
    ComparisonResult,
    create_baseline_comparator,
)

# ============= 基础评估器 (已整合，原模块已清理) =============
# 原evaluation模块已删除，基础评估功能已整合到现代评估器中
BASIC_EVALUATION_AVAILABLE = False

# 为向后兼容保留的占位符
RetrievalMetrics = None
RetrievalEvaluator = None
FusionEvaluator = None
BasicFeedbackProcessor = None
create_retrieval_evaluator = None
create_fusion_evaluator = None
create_basic_feedback_processor = None
precision_at_k = None
recall_at_k = None
ndcg_at_k = None
map_score = None
f1_score_at_k = None
dcg_at_k = None
average_precision = None
mrr_score = None

__all__ = [
    # ============= 现代评估器 (主要推荐，轻量级) =============
    "ModernEvaluator",
    "ModernEvaluatorConfig", 
    "RAGASEvaluator",
    "SemanticEvaluator",
    "ComplexityAwareEvaluator",
    "create_modern_evaluator",
    "create_modern_evaluator_config",
    
    # ============= 综合评估器 (完整功能，重量级) =============
    "ComprehensiveEvaluator",
    "ComprehensiveEvaluatorConfig",
    "BLEUEvaluator",
    "ROUGEEvaluator", 
    "BERTScoreEvaluator",
    "NDCGEvaluator",
    "MRREvaluator",
    "PrecisionRecallEvaluator", 
    "StatisticalTester",
    "EvaluationReport",
    "create_comprehensive_evaluator",
    "create_comprehensive_evaluator_config",
    
    # ============= 基线对比器 =============
    "BaselineComparator",
    "BaselineConfig",
    "ComparisonResult", 
    "create_baseline_comparator",
    
    # ============= 基础评估器 (整合，向后兼容占位符) =============
    "RetrievalMetrics",
    "RetrievalEvaluator",
    "FusionEvaluator", 
    "BasicFeedbackProcessor",
    "create_retrieval_evaluator",
    "create_fusion_evaluator",
    "create_basic_feedback_processor",
    
    # ============= 基础评估指标 (向后兼容占位符) =============
    "precision_at_k",
    "recall_at_k",
    "f1_score_at_k",
    "ndcg_at_k", 
    "dcg_at_k",
    "map_score",
    "average_precision",
    "mrr_score",
    
    # ============= 可用性标志 =============
    "BASIC_EVALUATION_AVAILABLE",
]

# ============= 便捷函数 =============
def create_evaluator(evaluator_type: str = "modern", **kwargs):
    """创建评估器的便捷函数
    
    Args:
        evaluator_type: 评估器类型 ("modern", "comprehensive", "basic")
        **kwargs: 评估器配置参数
        
    Returns:
        对应的评估器实例
        
    Note:
        "basic" 评估器已整合到现代评估器中，推荐使用 "modern" 或 "comprehensive"
    """
    if evaluator_type == "modern":
        return create_modern_evaluator(**kwargs)
    elif evaluator_type == "comprehensive":
        return create_comprehensive_evaluator(**kwargs)
    elif evaluator_type == "basic":
        # 引导用户使用现代评估器
        import warnings
        warnings.warn(
            "Basic evaluator has been integrated into modern evaluator. "
            "Using 'modern' evaluator instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        return create_modern_evaluator(**kwargs)
    else:
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

# 添加便捷函数到导出列表
__all__.append("create_evaluator") 