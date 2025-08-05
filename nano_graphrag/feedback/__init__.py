"""
反馈模块

提供用户反馈收集和处理功能
"""

from .feedback import (
    FeedbackCollector,
    FeedbackProcessor,
    create_feedback_collector,
    create_feedback_processor
)
from .storage import FeedbackStorage

__all__ = [
    'FeedbackCollector',
    'FeedbackProcessor',
    'FeedbackStorage',
    'create_feedback_collector',
    'create_feedback_processor'
] 