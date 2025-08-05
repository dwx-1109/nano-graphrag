"""
融合策略模块 (整合版)

包含：
- Enhanced Fusion: 集成多种先进融合策略的增强融合器
- Multi-Head Attention Fusion: 基于Transformer的注意力融合
- Confidence Calibration: 置信度校准和不确定性量化
- Traditional Fusion: 传统融合策略 (从hybrid模块整合)
- 各种融合组件和工具
"""

# ============= 高级融合策略 =============

# 导入Multi-Head Attention Fusion
from .multihead_attention_fusion import (
    MultiHeadAttentionFusionConfig,
    MultiHeadAttentionFusion,
    create_multihead_attention_fusion,
    create_multihead_attention_fusion_config,
)

# 导入Enhanced Fusion
from .enhanced_fusion import (
    EnhancedFusionConfig,
    EnhancedFusion,
    create_enhanced_fusion,
    create_enhanced_fusion_config,
)

# 导入置信度校准（原有）
from .confidence_calibration import (
    ConfidenceCalibratorConfig,
    ConfidenceCalibrator,
    create_confidence_calibrator,
    create_confidence_calibrator_config,
)

# 导入置信度感知融合器（新增）
try:
    from .confidence_aware_fusion import (
        ConfidenceAwareFusionConfig,
        ConfidenceWeightingModule,
        ConfidenceAwareFusion,
        create_confidence_aware_fusion,
        create_confidence_aware_fusion_config,
    )
    CONFIDENCE_AWARE_FUSION_AVAILABLE = True
except ImportError:
    # 置信度感知融合器可选，导入失败不影响主要功能
    ConfidenceAwareFusionConfig = None
    ConfidenceWeightingModule = None
    ConfidenceAwareFusion = None
    create_confidence_aware_fusion = None
    create_confidence_aware_fusion_config = None
    CONFIDENCE_AWARE_FUSION_AVAILABLE = False

# ============= 传统融合策略 (从hybrid模块整合) =============
try:
    from ..hybrid.fusion_strategies import (
        FusionStrategy,
        FusionResult,
        LinearFusion,
        RRFFusion,
        WeightedFusion,
        DynamicWeightFusion,
        ConfidenceAwareFusionStrategy,
        HybridFusionManager,
        create_fusion_strategy,
        create_fusion_manager,
    )
    from ..hybrid.normalization import (
        ScoreNormalizer,
        normalize_retrieval_scores,
    )
    TRADITIONAL_FUSION_AVAILABLE = True
except ImportError:
    # 传统融合策略可选，导入失败不影响高级功能
    FusionStrategy = None
    FusionResult = None
    LinearFusion = None
    RRFFusion = None
    WeightedFusion = None
    DynamicWeightFusion = None
    ConfidenceAwareFusionStrategy = None
    HybridFusionManager = None
    create_fusion_strategy = None
    create_fusion_manager = None
    ScoreNormalizer = None
    normalize_retrieval_scores = None
    TRADITIONAL_FUSION_AVAILABLE = False

__all__ = [
    # ============= 高级融合策略 =============
    # Multi-Head Attention Fusion
    'MultiHeadAttentionFusionConfig',
    'MultiHeadAttentionFusion',
    'create_multihead_attention_fusion',
    'create_multihead_attention_fusion_config',
    
    # Enhanced Fusion
    'EnhancedFusionConfig',
    'EnhancedFusion',
    'create_enhanced_fusion',
    'create_enhanced_fusion_config',
    
    # 置信度校准（原有）
    'ConfidenceCalibratorConfig',
    'ConfidenceCalibrator',
    'create_confidence_calibrator',
    'create_confidence_calibrator_config',
    
    # 置信度感知融合器（新增）
    'ConfidenceAwareFusionConfig',
    'ConfidenceWeightingModule',
    'ConfidenceAwareFusion',
    'create_confidence_aware_fusion',
    'create_confidence_aware_fusion_config',
    
    # ============= 传统融合策略 (整合) =============
    'FusionStrategy',
    'FusionResult',
    'LinearFusion',
    'RRFFusion',
    'WeightedFusion',
    'DynamicWeightFusion',
    'ConfidenceAwareFusionStrategy',
    'HybridFusionManager',
    'create_fusion_strategy',
    'create_fusion_manager',
    
    # 归一化工具
    'ScoreNormalizer',
    'normalize_retrieval_scores',
    
    # ============= 可用性标志 =============
    'CONFIDENCE_AWARE_FUSION_AVAILABLE',
    'TRADITIONAL_FUSION_AVAILABLE',
]

# ============= 统一融合接口 =============
def create_fusion_pipeline(fusion_type: str = "enhanced", **kwargs):
    """创建融合管道的便捷函数
    
    Args:
        fusion_type: 融合类型 ("enhanced", "attention", "traditional", "linear", "rrf")
        **kwargs: 融合器配置参数
        
    Returns:
        对应的融合器实例
    """
    if fusion_type == "enhanced":
        return create_enhanced_fusion(**kwargs)
    elif fusion_type == "attention":
        return create_multihead_attention_fusion(**kwargs)
    elif fusion_type == "confidence_aware" and CONFIDENCE_AWARE_FUSION_AVAILABLE:
        return create_confidence_aware_fusion(**kwargs)
    elif fusion_type == "traditional" and TRADITIONAL_FUSION_AVAILABLE:
        return create_fusion_manager(**kwargs)
    elif fusion_type == "linear" and TRADITIONAL_FUSION_AVAILABLE:
        return LinearFusion(**kwargs)
    elif fusion_type == "rrf" and TRADITIONAL_FUSION_AVAILABLE:
        return RRFFusion(**kwargs)
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")

def get_available_fusion_types():
    """获取可用的融合类型列表"""
    available_types = ["enhanced", "attention"]
    
    if CONFIDENCE_AWARE_FUSION_AVAILABLE:
        available_types.append("confidence_aware")
    
    if TRADITIONAL_FUSION_AVAILABLE:
        available_types.extend(["traditional", "linear", "rrf"])
    
    return available_types

# 添加便捷函数到导出列表
__all__.extend(["create_fusion_pipeline", "get_available_fusion_types"]) 