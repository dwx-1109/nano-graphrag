from typing import Union
import pickle
import asyncio
from openai import BadRequestError
from collections import defaultdict
import dspy
from nano_graphrag.base import (
    BaseGraphStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from nano_graphrag.answer_generation.prompts import PROMPTS
from nano_graphrag._utils import logger, compute_mdhash_id
from nano_graphrag.entity_extraction.module import TypedEntityRelationshipExtractor
# 这些函数需要重新实现或从其他模块导入
# from nano_graphrag._op import _merge_edges_then_upsert, _merge_nodes_then_upsert

# # 临时实现这些函数
# async def _merge_nodes_then_upsert(
#     nodes: list,
#     knwoledge_graph_inst: BaseGraphStorage,
#     global_config: dict = {},
# ) -> None:
#     """临时实现：合并节点并更新到图存储"""
#     if not nodes:
#         return
    
#     # 批量添加节点
#     await knwoledge_graph_inst.add_nodes_batch(nodes)

# async def _merge_edges_then_upsert(
#     source: str,
#     target: str,
#     edges: list,
#     knwoledge_graph_inst: BaseGraphStorage,
#     global_config: dict = {},
# ) -> None:
#     """临时实现：合并边并更新到图存储"""
#     if not edges:
#         return
    
#     # 批量添加边
#     await knwoledge_graph_inst.add_edges_batch(edges)

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """合并节点数据并更新到图数据库"""
    from collections import Counter
    from nano_graphrag._utils import split_string_by_multi_markers
    from nano_graphrag.answer_generation.prompts import GRAPH_FIELD_SEP
    
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    
    node_data = dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(entity_name, node_data)
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """合并边数据并更新到图数据库"""
    from nano_graphrag._utils import split_string_by_multi_markers
    from nano_graphrag.answer_generation.prompts import GRAPH_FIELD_SEP
    
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


async def generate_dataset(
    chunks: dict[str, TextChunkSchema],
    filepath: str,
    save_dataset: bool = True,
    global_config: dict = {},
) -> list[dspy.Example]:
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=True)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(
        chunk_key_dp: tuple[str, TextChunkSchema]
    ) -> dspy.Example:
        nonlocal already_processed, already_entities, already_relations
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            prediction = await asyncio.to_thread(entity_extractor, input_text=content)
            entities, relationships = prediction.entities, prediction.relationships
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []
        example = dspy.Example(
            input_text=content, entities=entities, relationships=relationships
        ).with_inputs("input_text")
        already_entities += len(entities)
        already_relations += len(relationships)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return example

    examples = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    filtered_examples = [
        example
        for example in examples
        if len(example.entities) > 0 and len(example.relationships) > 0
    ]
    num_filtered_examples = len(examples) - len(filtered_examples)
    if save_dataset:
        with open(filepath, "wb") as f:
            pickle.dump(filtered_examples, f)
            logger.info(
                f"Saved {len(filtered_examples)} examples with keys: {filtered_examples[0].keys()}, filtered {num_filtered_examples} examples"
            )

    return filtered_examples


async def extract_entities_dspy(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    using_amazon_bedrock: bool = False,  # 添加此参数以保持兼容性
) -> Union[BaseGraphStorage, None]:
    # 临时禁用self_refine以避免字段不匹配问题
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=False)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            prediction = await asyncio.to_thread(entity_extractor, input_text=content)
            entities, relationships = prediction.entities, prediction.relationships
            logger.debug(f"成功提取 {len(entities)} 个实体和 {len(relationships)} 个关系")
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []
        except Exception as e:
            logger.error(f"意外错误在实体提取中: {e}")
            logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
            entities, relationships = [], []

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for entity in entities:
            entity["source_id"] = chunk_key
            maybe_nodes[entity["entity_name"]].append(entity)
            already_entities += 1

        for relationship in relationships:
            relationship["source_id"] = chunk_key
            maybe_edges[(relationship["src_id"], relationship["tgt_id"])].append(
                relationship
            )
            already_relations += 1

        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knwoledge_graph_inst


# 创建别名以保持向后兼容
extract_entities = extract_entities_dspy
