"""
BM25检索查询

提供基于BM25算法的文档检索功能。
"""

import tiktoken
from typing import Dict, Any
from ..base import BaseKVStorage, TextChunkSchema, QueryParam
from .._utils import truncate_list_by_token_size, logger
from ..answer_generation.prompts import PROMPTS


async def bm25_query(
    query: str,
    bm25_storage: Any,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: Dict[str, Any],
) -> str:
    """
    BM25检索查询
    
    Args:
        query: 查询文本
        bm25_storage: BM25存储实例
        text_chunks_db: 文本块键值存储
        query_param: 查询参数
        global_config: 全局配置
        
    Returns:
        回答或上下文
    """
    use_model_func = global_config["best_model_func"]
    
    # 通过BM25检索相关文档
    results = await bm25_storage.search(query, top_k=query_param.top_k)
    
    if not len(results):
        return PROMPTS["fail_response"]
    
    # 如果只需要上下文，则直接返回检索结果
    # 创建tiktoken编码器
    tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
    maybe_trun_chunks = truncate_list_by_token_size(
        results,
        query_param.bm25_max_token_for_text_unit,
        tiktoken_model,
        key=lambda x: x["content"]
    )
    logger.info(f"BM25搜索 - 截断 {len(results)} 到 {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    
    if query_param.only_need_context:
        return section
    
    # 使用LLM生成回答
    sys_prompt_temp = PROMPTS["naive_rag_response"]  # 复用naive模式的提示模板
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response 