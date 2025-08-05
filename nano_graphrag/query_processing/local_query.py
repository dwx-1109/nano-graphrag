"""
局部图检索查询

提供基于局部图结构的查询功能。
"""

from typing import Dict, Any
from ..base import BaseGraphStorage, BaseKVStorage, CommunitySchema, TextChunkSchema, QueryParam, BaseVectorStorage
from ..answer_generation.prompts import PROMPTS
from ..context.context_builder import _build_local_query_context
from .._utils import logger


async def local_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: Dict[str, Any],
    return_context: bool = False
) -> str:
    """
    局部图检索查询
    
    Args:
        query: 查询文本
        knowledge_graph_inst: 知识图谱实例
        entities_vdb: 实体向量数据库
        community_reports: 社区报告存储
        text_chunks_db: 文本块存储
        query_param: 查询参数
        global_config: 全局配置
        return_context: 是否只返回上下文
        
    Returns:
        回答或上下文
    """
    use_model_func = global_config["best_model_func"]
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
    )
    if query_param.only_need_context or return_context:
        return context
    if context is None or not context.strip():
        logger.warning(f"No valid context found for local query: {query}")
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response 