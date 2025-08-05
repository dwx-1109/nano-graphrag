"""
优化器模块

提供各种优化策略和参数调优功能:
- 简单优化器：基础参数优化
- 进化优化器：遗传算法优化
- 贝叶斯优化器：高效超参数搜索
"""

# 导出所有函数以保持向后兼容
__all__ = [
    # 简单优化器
    'SimpleOptimizer',
    'SimpleOptimizerConfig',
    'create_simple_optimizer',
    'create_simple_optimizer_config',
    
    # 进化优化器 (可选)
    'EvolutionaryOptimizer',
    'EvolutionaryOptimizerConfig',
    'create_evolutionary_optimizer',
    'create_evolutionary_optimizer_config',
    
    # 贝叶斯优化器 (可选)
    'BayesianOptimizer',
    'BayesianOptimizerConfig',
    'create_bayesian_optimizer',
    'create_bayesian_optimizer_config',
]

from .simple_optimizer import (
    SimpleOptimizer,
    SimpleOptimizerConfig,
    create_simple_optimizer,
    create_simple_optimizer_config,
)

# 进化优化器 (可选依赖)
try:
    from .evolution import (
        EvolutionaryOptimizer,
        EvolutionaryOptimizerConfig,
        create_evolutionary_optimizer,
        create_evolutionary_optimizer_config,
    )
except ImportError:
    # 如果依赖不可用，创建占位符
    EvolutionaryOptimizer = None
    EvolutionaryOptimizerConfig = None
    create_evolutionary_optimizer = None
    create_evolutionary_optimizer_config = None

# 贝叶斯优化器 (可选依赖)
try:
    from .bayesian import (
        BayesianOptimizer,
        BayesianOptimizerConfig,
        create_bayesian_optimizer,
        create_bayesian_optimizer_config,
    )
except ImportError:
    # 如果依赖不可用，创建占位符
    BayesianOptimizer = None
    BayesianOptimizerConfig = None
    create_bayesian_optimizer = None
    create_bayesian_optimizer_config = None 