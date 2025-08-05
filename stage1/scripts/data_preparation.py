#!/usr/bin/env python3
"""
阶段1数据预处理脚本
下载、清理和标注数据集，为复杂度感知路由实验做准备
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import re
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nano_graphrag.complexity.classifier import classify_query_complexity_sync

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, data_dir: str = "stage1/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.annotations_dir = self.data_dir / "annotations"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.annotations_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def download_ms_marco(self, subset_size: int = 10000) -> str:
        """下载MS MARCO数据集"""
        logger.info(f"下载MS MARCO数据集 (子集大小: {subset_size})")
        
        try:
            # 下载MS MARCO QA数据集
            dataset = load_dataset("ms_marco", "v2.1", split="train")
            
            # 取子集以加快实验
            if subset_size and len(dataset) > subset_size:
                dataset = dataset.select(range(subset_size))
            
            # 保存到本地
            output_file = self.raw_dir / "ms_marco.json"
            
            processed_data = []
            for item in tqdm(dataset, desc="处理MS MARCO"):
                if item['answers'] and len(item['answers']) > 0:
                    processed_data.append({
                        'query_id': item['query_id'],
                        'query': item['query'],
                        'answers': item['answers'],
                        'passages': item.get('passages', []),
                        'source': 'ms_marco'
                    })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"MS MARCO数据集已保存到: {output_file}")
            logger.info(f"样本数量: {len(processed_data)}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"下载MS MARCO失败: {e}")
            return None
    
    def download_natural_questions(self, subset_size: int = 5000) -> str:
        """下载Natural Questions数据集"""
        logger.info(f"下载Natural Questions数据集 (子集大小: {subset_size})")
        
        try:
            # 下载Natural Questions数据集
            dataset = load_dataset("natural_questions", "default", split="train")
            
            if subset_size and len(dataset) > subset_size:
                dataset = dataset.select(range(subset_size))
            
            output_file = self.raw_dir / "natural_questions.json"
            
            processed_data = []
            for item in tqdm(dataset, desc="处理Natural Questions"):
                question = item['question']['text']
                
                # 提取答案
                annotations = item['annotations']
                if annotations and len(annotations) > 0:
                    short_answers = []
                    long_answers = []
                    
                    for ann in annotations:
                        if ann['short_answers']:
                            for sa in ann['short_answers']:
                                if sa['text']:
                                    short_answers.append(sa['text'])
                        
                        if ann['long_answer'] and ann['long_answer']['candidate_index'] >= 0:
                            long_answers.append("Long answer available")
                    
                    if short_answers or long_answers:
                        processed_data.append({
                            'query_id': f"nq_{item['id']}",
                            'query': question,
                            'short_answers': short_answers,
                            'long_answers': long_answers,
                            'document': item['document']['html'] if item['document'] else "",
                            'source': 'natural_questions'
                        })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Natural Questions数据集已保存到: {output_file}")
            logger.info(f"样本数量: {len(processed_data)}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"下载Natural Questions失败: {e}")
            return None
    
    def download_hotpot_qa(self, subset_size: int = 5000) -> str:
        """下载HotpotQA数据集"""
        logger.info(f"下载HotpotQA数据集 (子集大小: {subset_size})")
        
        try:
            # 下载HotpotQA数据集
            dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
            
            if subset_size and len(dataset) > subset_size:
                dataset = dataset.select(range(subset_size))
            
            output_file = self.raw_dir / "hotpot_qa.json"
            
            processed_data = []
            for item in tqdm(dataset, desc="处理HotpotQA"):
                processed_data.append({
                    'query_id': item['id'],
                    'query': item['question'],
                    'answer': item['answer'],
                    'context': item['context'],
                    'supporting_facts': item['supporting_facts'],
                    'level': item['level'],  # easy or hard
                    'type': item['type'],    # bridge or comparison
                    'source': 'hotpot_qa'
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"HotpotQA数据集已保存到: {output_file}")
            logger.info(f"样本数量: {len(processed_data)}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"下载HotpotQA失败: {e}")
            return None
    
    def download_squad_v2(self, subset_size: int = 5000) -> str:
        """下载SQuAD 2.0数据集"""
        logger.info(f"下载SQuAD 2.0数据集 (子集大小: {subset_size})")
        
        try:
            # 下载SQuAD 2.0数据集
            dataset = load_dataset("squad_v2", split="train")
            
            if subset_size and len(dataset) > subset_size:
                dataset = dataset.select(range(subset_size))
            
            output_file = self.raw_dir / "squad_v2.json"
            
            processed_data = []
            for item in tqdm(dataset, desc="处理SQuAD 2.0"):
                answers = []
                if item['answers']['text']:
                    answers = item['answers']['text']
                
                processed_data.append({
                    'query_id': item['id'],
                    'query': item['question'],
                    'answers': answers,
                    'context': item['context'],
                    'title': item['title'],
                    'is_impossible': item['answers']['answer_start'] == [],
                    'source': 'squad_v2'
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"SQuAD 2.0数据集已保存到: {output_file}")
            logger.info(f"样本数量: {len(processed_data)}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"下载SQuAD 2.0失败: {e}")
            return None


class ComplexityAnnotator:
    """复杂度标注器"""
    
    def __init__(self):
        self.complexity_stats = {
            'zero_hop': 0,
            'one_hop': 0,
            'multi_hop': 0
        }
    
    def annotate_query_complexity(self, query: str) -> str:
        """标注查询复杂度"""
        try:
            # 使用项目的复杂度分类器
            complexity = classify_query_complexity_sync(query)
            self.complexity_stats[complexity] += 1
            return complexity
        except Exception as e:
            logger.warning(f"复杂度分类失败，使用启发式方法: {e}")
            return self._heuristic_complexity(query)
    
    def _heuristic_complexity(self, query: str) -> str:
        """启发式复杂度分类"""
        query_lower = query.lower()
        
        # Zero-hop: 定义性问题
        zero_hop_patterns = [
            r'\bwhat is\b', r'\bwho is\b', r'\bwhen is\b', r'\bwhere is\b',
            r'\bdefine\b', r'\bmean\b', r'\bmeaning\b'
        ]
        
        # Multi-hop: 比较、关系、推理问题
        multi_hop_patterns = [
            r'\bcompare\b', r'\bdifference\b', r'\bsimilar\b', r'\brelationship\b',
            r'\bhow does.*affect\b', r'\bwhy does\b', r'\bcause.*effect\b',
            r'\bboth.*and\b', r'\beither.*or\b'
        ]
        
        for pattern in zero_hop_patterns:
            if re.search(pattern, query_lower):
                if len(query.split()) <= 6:  # 简短的定义性问题
                    self.complexity_stats['zero_hop'] += 1
                    return 'zero_hop'
                break
        
        for pattern in multi_hop_patterns:
            if re.search(pattern, query_lower):
                self.complexity_stats['multi_hop'] += 1
                return 'multi_hop'
        
        # 默认为one-hop
        self.complexity_stats['one_hop'] += 1
        return 'one_hop'
    
    def get_stats(self) -> Dict[str, int]:
        """获取标注统计"""
        return self.complexity_stats.copy()


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, data_dir: str = "stage1/data"):
        self.data_dir = Path(data_dir)
        self.downloader = DatasetDownloader(data_dir)
        self.annotator = ComplexityAnnotator()
    
    def process_dataset(self, dataset_file: str) -> List[Dict[str, Any]]:
        """处理单个数据集文件"""
        logger.info(f"处理数据集: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        for item in tqdm(data, desc=f"标注复杂度"):
            query = item['query']
            complexity = self.annotator.annotate_query_complexity(query)
            
            processed_item = {
                **item,
                'complexity': complexity,
                'query_length': len(query.split()),
                'query_type': self._classify_query_type(query)
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'meaning']):
            return 'definition'
        elif any(word in query_lower for word in ['how', 'explain', 'describe']):
            return 'explanation'
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in query_lower for word in ['compare', 'difference', 'similar']):
            return 'comparison'
        elif any(word in query_lower for word in ['who', 'when', 'where']):
            return 'factual'
        else:
            return 'other'
    
    def create_unified_dataset(self, dataset_files: List[str]) -> str:
        """创建统一的数据集"""
        logger.info("创建统一数据集...")
        
        all_data = []
        for dataset_file in dataset_files:
            if os.path.exists(dataset_file):
                processed_data = self.process_dataset(dataset_file)
                all_data.extend(processed_data)
        
        # 保存统一数据集
        output_file = self.data_dir / "processed" / "unified_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # 保存复杂度统计
        stats_file = self.data_dir / "annotations" / "complexity_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotator.get_stats(), f, indent=2)
        
        logger.info(f"统一数据集已保存到: {output_file}")
        logger.info(f"总样本数: {len(all_data)}")
        logger.info(f"复杂度分布: {self.annotator.get_stats()}")
        
        return str(output_file)
    
    def create_test_splits(self, unified_dataset_file: str, 
                          test_ratio: float = 0.2, 
                          val_ratio: float = 0.1) -> Dict[str, str]:
        """创建训练/验证/测试划分"""
        logger.info("创建数据划分...")
        
        with open(unified_dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 按复杂度分层抽样
        complexity_groups = {}
        for item in data:
            complexity = item['complexity']
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(item)
        
        train_data, val_data, test_data = [], [], []
        
        for complexity, items in complexity_groups.items():
            np.random.shuffle(items)
            n_items = len(items)
            
            n_test = int(n_items * test_ratio)
            n_val = int(n_items * val_ratio)
            n_train = n_items - n_test - n_val
            
            train_data.extend(items[:n_train])
            val_data.extend(items[n_train:n_train + n_val])
            test_data.extend(items[n_train + n_val:])
        
        # 保存划分
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        output_files = {}
        for split_name, split_data in splits.items():
            output_file = self.data_dir / "processed" / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            output_files[split_name] = str(output_file)
            logger.info(f"{split_name} 集: {len(split_data)} 样本")
        
        return output_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="阶段1数据预处理")
    parser.add_argument("--download", action="store_true", help="下载数据集")
    parser.add_argument("--annotate", action="store_true", help="标注复杂度")
    parser.add_argument("--data_dir", default="stage1/data", help="数据目录")
    parser.add_argument("--subset_size", type=int, default=5000, help="数据集子集大小")
    
    args = parser.parse_args()
    
    # 确保在项目根目录
    if not os.path.exists("nano_graphrag"):
        logger.error("请在项目根目录运行此脚本")
        sys.exit(1)
    
    processor = DataProcessor(args.data_dir)
    
    if args.download:
        logger.info("开始下载数据集...")
        
        dataset_files = []
        
        # 下载各个数据集
        ms_marco_file = processor.downloader.download_ms_marco(args.subset_size)
        if ms_marco_file:
            dataset_files.append(ms_marco_file)
        
        nq_file = processor.downloader.download_natural_questions(args.subset_size // 2)
        if nq_file:
            dataset_files.append(nq_file)
        
        hotpot_file = processor.downloader.download_hotpot_qa(args.subset_size // 2)
        if hotpot_file:
            dataset_files.append(hotpot_file)
        
        squad_file = processor.downloader.download_squad_v2(args.subset_size // 2)
        if squad_file:
            dataset_files.append(squad_file)
        
        logger.info(f"成功下载 {len(dataset_files)} 个数据集")
        
        # 保存数据集文件列表
        files_list = processor.data_dir / "raw" / "dataset_files.json"
        with open(files_list, 'w') as f:
            json.dump(dataset_files, f, indent=2)
    
    if args.annotate:
        logger.info("开始标注复杂度...")
        
        # 读取数据集文件列表
        files_list = processor.data_dir / "raw" / "dataset_files.json"
        if not files_list.exists():
            logger.error("未找到数据集文件列表，请先运行 --download")
            sys.exit(1)
        
        with open(files_list, 'r') as f:
            dataset_files = json.load(f)
        
        # 创建统一数据集
        unified_file = processor.create_unified_dataset(dataset_files)
        
        # 创建数据划分
        splits = processor.create_test_splits(unified_file)
        
        logger.info("数据预处理完成！")
        logger.info(f"训练集: {splits['train']}")
        logger.info(f"验证集: {splits['validation']}")
        logger.info(f"测试集: {splits['test']}")
        
        # 显示复杂度分布
        stats_file = processor.data_dir / "annotations" / "complexity_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            logger.info(f"复杂度分布: {stats}")


if __name__ == "__main__":
    main()