"""
去重模块

提供文档去重、相似度计算和质量过滤功能。
"""

from .deduplication import (
    SimilarityCalculator,
    DocumentDeduplicator,
    DeduplicationResult,
    QualityFilter,
    QualityFilterResult,
    calculate_text_similarity,
    deduplicate_documents,
    filter_documents_by_quality
)

# 添加别名以保持与integration.py中引用一致
Deduplicator = DocumentDeduplicator

__all__ = [
    'SimilarityCalculator',
    'DocumentDeduplicator',
    'Deduplicator',  # 添加别名到导出列表
    'DeduplicationResult',
    'QualityFilter',
    'QualityFilterResult',
    'calculate_text_similarity',
    'deduplicate_documents',
    'filter_documents_by_quality'
] 