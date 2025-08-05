"""
图神经网络模块

包含：
- Graph Transformer: 现代化的图Transformer实现
- GraphSAGE: 传统图神经网络（保留兼容性）
- Complexity Aware: 复杂度感知层
- 工具函数和配置类
"""

# 导入GraphSAGE相关
try:
    from .graphsage import (
        GraphSAGEConfig,
        GraphSAGE,
        create_graphsage_model,
        create_graphsage_config,
    )
    GRAPHSAGE_AVAILABLE = True
except ImportError:
    GRAPHSAGE_AVAILABLE = False
    GraphSAGEConfig = None
    GraphSAGE = None
    create_graphsage_model = None
    create_graphsage_config = None

# 导入Graph Transformer相关
try:
    from .graph_transformer import (
        GraphTransformerConfig,
        GraphTransformer,
        create_graph_transformer,
        create_graph_transformer_config,
    )
    GRAPH_TRANSFORMER_AVAILABLE = True
except ImportError:
    GRAPH_TRANSFORMER_AVAILABLE = False
    GraphTransformerConfig = None
    GraphTransformer = None
    create_graph_transformer = None
    create_graph_transformer_config = None

# 导入复杂度感知模块
try:
    from .complexity_aware import (
        ComplexityAwareConfig,
        ComplexityAwareLayer,
        create_complexity_aware_layer,
        create_complexity_aware_config,
    )
    COMPLEXITY_AWARE_AVAILABLE = True
except ImportError:
    COMPLEXITY_AWARE_AVAILABLE = False
    ComplexityAwareConfig = None
    ComplexityAwareLayer = None
    create_complexity_aware_layer = None
    create_complexity_aware_config = None

# 导入轻量化集成器
try:
    from .lightweight_integration import (
        LightweightGraphTransformerIntegrator,
        create_lightweight_graph_transformer,
    )
    LIGHTWEIGHT_INTEGRATION_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_INTEGRATION_AVAILABLE = False
    LightweightGraphTransformerIntegrator = None
    create_lightweight_graph_transformer = None

# 导入工具函数
try:
    from .utils import (
        prepare_graph_data,
        compute_graph_metrics,
        visualize_attention_weights,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    prepare_graph_data = None
    compute_graph_metrics = None
    visualize_attention_weights = None

# 导出列表
__all__ = []

if GRAPHSAGE_AVAILABLE:
    __all__.extend([
        'GraphSAGEConfig',
        'GraphSAGE', 
        'create_graphsage_model',
        'create_graphsage_config',
    ])
    
if GRAPH_TRANSFORMER_AVAILABLE:
    __all__.extend([
        'GraphTransformerConfig',
        'GraphTransformer',
        'create_graph_transformer',
        'create_graph_transformer_config',
    ])
    
if COMPLEXITY_AWARE_AVAILABLE:
    __all__.extend([
        'ComplexityAwareConfig',
        'ComplexityAwareLayer',
        'create_complexity_aware_layer',
        'create_complexity_aware_config',
    ])
    
if LIGHTWEIGHT_INTEGRATION_AVAILABLE:
    __all__.extend([
        'LightweightGraphTransformerIntegrator',
        'create_lightweight_graph_transformer',
    ])
    
if UTILS_AVAILABLE:
    __all__.extend([
        'prepare_graph_data',
        'compute_graph_metrics',
        'visualize_attention_weights',
    ])

# 添加可用性标志
__all__.extend([
    'GRAPHSAGE_AVAILABLE',
    'GRAPH_TRANSFORMER_AVAILABLE',
    'COMPLEXITY_AWARE_AVAILABLE',
    'LIGHTWEIGHT_INTEGRATION_AVAILABLE',
    'UTILS_AVAILABLE',
]) 