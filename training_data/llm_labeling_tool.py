#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM辅助问题复杂度标注工具
使用云服务LLM对问题进行zero_hop、one_hop、multi_hop分类标注
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
import requests
from dataclasses import dataclass
import openai
from openai import OpenAI
import sys
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_labeling.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LabelingConfig:
    """标注配置"""
    api_key: str
    model: str = "gpt-4"
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    temperature: float = 0.1
    max_tokens: int = 500
    base_url: str = None  # 新增

class LLMLabelingTool:
    """LLM辅助标注工具"""
    
    def __init__(self, config: LabelingConfig):
        self.config = config
        if config.base_url:
            self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = OpenAI(api_key=config.api_key)
        
        self.labeling_criteria = """
You are a senior question classification expert specializing in "retrieval hop" classification tasks. Your goal is to strictly classify input natural language questions into one of the following three categories and provide brief but precise reasoning.

## Classification Definitions

### 1. zero-hop
**Core Characteristics**: The answer can be directly provided by common sense or the model's internal knowledge base in one go, **requiring no retrieval whatsoever**; typically very basic attribute queries or common sense judgments.
**Key Requirements**: Limited to basic facts that any ordinary adult can answer directly. Any question involving professional terms, academic knowledge, technical concepts, or requiring consultation of encyclopedias/professional books/networks is NOT zero-hop.
**Typical Examples**: 
- Colors, shapes, simple arithmetic, common sense facts
- "What color is the sky on a clear day?" → zero-hop (common sense observation)
- "What is 2+2?" → zero-hop (basic arithmetic)
- "Is water wet?" → zero-hop (common sense)
- "What is the capital of France?" → zero-hop (common knowledge)

**Negative Examples** (NOT zero-hop):
- "What is an NP problem?" → one-hop (requires professional knowledge)
- "What is best-case time complexity?" → one-hop (technical concept)
- "What is computational complexity theory?" → one-hop (academic knowledge)
- "What is photosynthesis?" → one-hop (requires encyclopedia/textbook)

### 2. one-hop
**Core Characteristics**: Requires **one** external retrieval or consultation of a specific structured/textual source, **one-step process**. The query object is usually an attribute of an entity or a single fact.
**Key Requirements**: Involves technical concepts, professional terms, or academic knowledge that requires external lookup.
**Typical Examples**: 
- Entity attribute queries or single topics
- "Who directed the movie Titanic?" → one-hop (single fact lookup)
- "What is the capital of Brazil?" → one-hop (specific fact)
- "When was the Declaration of Independence signed?" → one-hop (historical fact)
- "What is artificial intelligence?" → one-hop (requires encyclopedia/professional books)
- "What is an NP problem?" → one-hop (requires professional knowledge)
- "What is best-case time complexity?" → one-hop (technical concept, requires professional knowledge)

### 3. multi-hop
**Core Characteristics**: Requires crossing relationships between **two or more** entities or attributes, or performing **exclusion, filtering, statistics** and other multi-step logic before arriving at an answer.
**Typical Examples**: 
- "First find all authors who wrote about Chopin, then filter for French nationality and exclude George Sands" → multi-hop (multiple steps with filtering)
- "Who is the spouse of the author who wrote the book that won the Pulitzer Prize in 2020?" → multi-hop (multiple entity relationships)
- "Which movies won both Oscar Best Picture and Golden Globe?" → multi-hop (need to find two award lists + intersection)
- "What is the time complexity of the fastest known algorithm for NP-complete problems?" → multi-hop (need to find NP-complete problems + fastest algorithms + their complexities)

## Classification Decision Tree
1. **Does the question involve any technical terms, professional concepts, or academic knowledge?**
   - YES → NOT zero-hop (go to step 2)
   - NO → Consider zero-hop (if it's truly common sense)

2. **Does the question require looking up information from external sources (encyclopedia, database, textbook, etc.)?**
   - YES → one-hop or multi-hop (go to step 3)
   - NO → Re-evaluate step 1

3. **Does the question require combining information from multiple sources or performing calculations/comparisons?**
   - YES → multi-hop
   - NO → one-hop

## Key Rules
- **Technical terms = one-hop**: Any question mentioning "complexity", "algorithm", "theory", "analysis", "optimization", "efficiency", etc. is one-hop
- **Professional knowledge = one-hop**: Any question requiring domain-specific knowledge is one-hop
- **Academic concepts = one-hop**: Any question about scientific, mathematical, or academic concepts is one-hop
- **Only pure common sense = zero-hop**: Limited to basic facts everyone knows
- **Multiple entities/relationships = multi-hop**: Questions involving multiple entities or requiring filtering/comparison are multi-hop

## Guidelines
- **Output JSON only**: Do not add extra text or comments
- **Precise reasoning**: 1-2 sentences, must point out the "hop count" basis, such as "single attribute retrieval", "multi-entity filtering", etc.
- **Avoid ambiguity**: If the question contains exclusion, statistics, or mentions multiple entities, classify as multi-hop
- **Cover boundary cases**: Clearly indicate the classification for easily confused boundary scenarios

## Judgment Points
1. **Number of information sources**: How many independent information sources need to be queried?
2. **Reasoning steps**: Is combination, comparison, or calculation needed?
3. **Complexity**: Is understanding relationships or context required?
4. **Common sense level**: Is it something any ordinary adult can answer directly?
5. **Professional terms/academic knowledge**: Any question involving professional terms, academic knowledge, technical concepts, or requiring consultation of encyclopedias/networks is NOT zero-hop.
6. **Technical concepts**: Questions about algorithms, complexity, computational theory, scientific processes, etc. are NEVER zero-hop.

Please carefully analyze each question and ensure accurate classification. Output only a JSON array in the following format:
[{"query": "question text", "label": "zero-hop/one-hop/multi-hop"}]
"""
        
        self.labeling_prompt_template = """
Please classify the complexity of the following questions:

{questions}

Output only a JSON array, no other content.
"""
    
    def create_labeling_prompt(self, questions: list) -> str:
        # 只发送query，减少token使用
        question_list = "\n".join([f"{i+1}. {q['query']}" for i, q in enumerate(questions)])
        return self.labeling_prompt_template.format(questions=question_list)
    
    def parse_llm_response(self, response: str) -> list:
        import re
        import json
        try:
            # 尝试直接解析完整JSON
            start = response.find('[')
            end = response.rfind(']')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # 如果找不到完整的JSON数组，尝试解析单个对象
            results = []
            # 查找所有可能的JSON对象
            pattern = r'\{[^{}]*"query"[^{}]*"label"[^{}]*\}'
            matches = re.findall(pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    # 清理可能的格式问题
                    cleaned = match.replace('\n', ' ').replace('\r', ' ')
                    # 移除可能的尾随逗号
                    if cleaned.endswith(','):
                        cleaned = cleaned[:-1]
                    # 尝试修复常见的JSON问题
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    
                    obj = json.loads(cleaned)
                    if 'query' in obj and 'label' in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    try:
                        # 移除所有换行符和多余空格
                        cleaned = re.sub(r'\s+', ' ', match.strip())
                        # 确保引号配对
                        cleaned = re.sub(r'([^"])"([^"]*?)([^"])"', r'\1"\2\3"', cleaned)
                        obj = json.loads(cleaned)
                        if 'query' in obj and 'label' in obj:
                            results.append(obj)
                    except:
                        continue
            
            if results:
                logger.info(f"从截断响应中成功解析 {len(results)} 个结果")
                return results
            
            logger.error(f"无法解析LLM响应: {response[:200]}...")
            return []
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}, 响应: {response[:200]}...")
            return []
    
    def call_llm_api(self, prompt: str) -> str:
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.labeling_criteria},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                result = response.choices[0].message.content.strip()
                
                # 验证响应是否包含完整的JSON结构
                if '[' in result and ']' in result:
                    start = result.find('[')
                    end = result.rfind(']')
                    if start != -1 and end != -1 and end > start:
                        return result
                    else:
                        logger.warning(f"响应JSON不完整，尝试重试 (尝试 {attempt + 1}/{self.config.max_retries})")
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                else:
                    logger.warning(f"响应格式异常，尝试重试 (尝试 {attempt + 1}/{self.config.max_retries})")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                
                return result
                
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"API调用最终失败: {e}")
                    return None
    
    def label_single_question(self, question: str) -> Optional[Dict[str, Any]]:
        """标注单个问题"""
        prompt = self.create_labeling_prompt(question)
        response = self.call_llm_api(prompt)
        
        if response is None:
            return None
        
        result = self.parse_llm_response(response)
        if result is None:
            return None
        
        return {
            "question": question,
            "label": result.get("complexity", "UNKNOWN"),
            "analysis": result.get("analysis", ""),
            "reasoning": result.get("reasoning", ""),
            "llm_response": response
        }
    
    def label_batch_questions(self, questions: list) -> list:
        """批量标注问题"""
        results = []
        failed = []
        total_questions = len(questions)
        pbar = tqdm(total=total_questions, desc="LLM标注进度", ncols=80)
        for i, question in enumerate(questions, 1):
            result = self.label_single_question(question)
            if result:
                results.append(result)
                logger.info(f"标注: {result.get('question')} | label: {result.get('label')}")
            else:
                logger.warning(f"标注失败: {question}")
                failed.append(question)
                results.append({
                    "question": question,
                    "label": "FAILED",
                    "analysis": "",
                    "reasoning": "",
                    "llm_response": ""
                })
            pbar.update(1)
            if i % self.config.batch_size == 0:
                time.sleep(1)
        pbar.close()
        if failed:
            with open("failed_samples.json", "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            logger.warning(f"有{len(failed)}条样本标注失败，已导出到failed_samples.json")
        logger.info(f"批量标注完成，成功标注 {len([r for r in results if r['label'] != 'FAILED'])}/{total_questions} 个问题")
        return results
    
    def load_questions_from_json(self, file_path: str) -> List[str]:
        """从JSON文件加载问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持多种JSON格式
            if isinstance(data, list):
                # 格式1: [{"question": "..."}, ...]
                if data and isinstance(data[0], dict) and "question" in data[0]:
                    return [item["question"] for item in data]
                # 格式2: ["question1", "question2", ...]
                elif data and isinstance(data[0], str):
                    return data
            elif isinstance(data, dict):
                # 格式3: {"questions": ["question1", "question2", ...]}
                if "questions" in data:
                    return data["questions"]
                # 格式4: {"data": [{"question": "..."}, ...]}
                elif "data" in data and isinstance(data["data"], list):
                    return [item["question"] for item in data["data"] if "question" in item]
            
            raise ValueError("不支持的JSON格式")
            
        except Exception as e:
            logger.error(f"加载问题文件失败: {e}")
            raise
    
    def save_results_to_json(self, results: List[Dict[str, Any]], output_path: str):
        """保存结果到JSON文件"""
        try:
            # 创建输出目录（如果路径包含目录）
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 准备输出数据
            output_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_questions": len(results),
                    "successful_labels": len([r for r in results if r['label'] != 'FAILED']),
                    "failed_labels": len([r for r in results if r['label'] == 'FAILED']),
                    "model": self.config.model,
                    "label_distribution": self.get_label_distribution(results)
                },
                "results": results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def get_label_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取标注分布"""
        distribution = {}
        for result in results:
            label = result.get('label', 'unknown')
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印标注摘要"""
        if not results:
            print("没有标注结果")
            return
        
        distribution = self.get_label_distribution(results)
        total = len(results)
        
        print("="*60)
        print("标注结果摘要")
        print("="*60)
        print(f"总样本数: {total}")
        print(f"标注分布:")
        for label, count in sorted(distribution.items()):
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        print("="*60)

    def overwrite_existing_labels(self, data: list, batch_size=3) -> list:  # 进一步减少批处理大小
        all_queries = [{"query": item["query"]} for item in data]
        results = []
        failed = []
        pbar = tqdm(total=len(all_queries), desc="LLM标注进度", ncols=80)
        
        for i in range(0, len(all_queries), batch_size):
            batch_queries = all_queries[i:i+batch_size]
            prompt = self.create_labeling_prompt(batch_queries)
            response = self.call_llm_api(prompt)
            batch_result = self.parse_llm_response(response) if response else []
            
            # 如果批量处理失败，尝试逐个处理
            if not batch_result and len(batch_queries) > 1:
                logger.info(f"批量处理失败，尝试逐个处理 {len(batch_queries)} 个样本")
                for single_query in batch_queries:
                    single_prompt = self.create_labeling_prompt([single_query])
                    single_response = self.call_llm_api(single_prompt)
                    single_result = self.parse_llm_response(single_response) if single_response else []
                    if single_result:
                        batch_result.extend(single_result)
                        logger.info(f"单个处理成功: {single_result[0].get('query')}")
                    else:
                        logger.warning(f"单个处理也失败: {single_query['query']}")
                        failed.append(single_query['query'])
                    time.sleep(0.5)
            
            results.extend(batch_result)
            # 日志细节
            for res in batch_result:
                logger.info(f"标注: {res.get('query')} | label: {res.get('label')}")
            
            # 记录失败的样本
            processed_queries = {res.get('query') for res in batch_result}
            for q in batch_queries:
                if q['query'] not in processed_queries:
                    logger.warning(f"标注失败: {q['query']}")
                    failed.append(q['query'])
            
            pbar.update(len(batch_queries))
            time.sleep(1)
        
        pbar.close()
        
        # 只更新complexity字段，保留原有explanation
        query2label = {item["query"]: item["label"] for item in results}
        new_data = []
        for item in data:
            q = item["query"]
            if q in query2label:
                item["complexity"] = query2label[q]
                # 保留原有explanation，不修改
            new_data.append(item)
        
        # 导出失败样本
        if failed:
            with open("failed_samples.json", "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            logger.warning(f"有{len(failed)}条样本标注失败，已导出到failed_samples.json")
        
        return new_data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM辅助问题复杂度标注工具")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--api-key", required=True, help="OpenAI API密钥")
    parser.add_argument("--model", default="gpt-4", help="使用的模型 (默认: gpt-4)")
    parser.add_argument("--batch-size", type=int, default=10, help="批处理大小 (默认: 10)")
    parser.add_argument("--temperature", type=float, default=0.1, help="温度参数 (默认: 0.1)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI API base_url (如Qwen)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已标注数据，仅更新complexity和explanation")
    args = parser.parse_args()
    
    # 创建配置
    config = LabelingConfig(
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        base_url=args.base_url
    )
    
    # 创建标注工具
    labeling_tool = LLMLabelingTool(config)
    
    try:
        # 加载问题
        logger.info(f"加载问题文件: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if args.overwrite:
            logger.info("覆盖标注模式：仅更新complexity和explanation字段")
            results = labeling_tool.overwrite_existing_labels(data, batch_size=args.batch_size)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存到: {args.output}")
        else:
            # 原有流程...
            questions = labeling_tool.load_questions_from_json(args.input)
            results = labeling_tool.label_batch_questions(questions)
            labeling_tool.save_results_to_json(results, args.output)
            labeling_tool.print_summary(results)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main() 