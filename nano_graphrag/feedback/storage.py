"""
反馈存储模块

提供反馈数据的存储和检索功能
"""

import os
import json
import time
import pickle
import logging
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime

import numpy as np


class FeedbackStorage:
    """反馈存储器"""
    
    def __init__(self, storage_dir: str = "feedback_data", use_json: bool = True):
        """
        初始化反馈存储器
        
        Args:
            storage_dir: 存储目录
            use_json: 是否使用JSON格式存储
        """
        self.storage_dir = storage_dir
        self.use_json = use_json
        
        # 内存缓存
        self.cache = {}
        
        # 会话索引
        self.session_index = {}
        
        # 特征索引
        self.feature_index = {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
    
    def store(self, query_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        存储反馈数据
        
        Args:
            query_id: 查询ID
            feedback_data: 反馈数据
            
        Returns:
            是否成功
        """
        if not query_id:
            self.logger.error("查询ID不能为空")
            return False
        
        # 更新内存缓存
        if query_id not in self.cache:
            self.cache[query_id] = []
        
        self.cache[query_id].append(feedback_data)
        
        # 更新会话索引
        session_id = feedback_data.get('session_id', None)
        if session_id:
            if session_id not in self.session_index:
                self.session_index[session_id] = set()
            self.session_index[session_id].add(query_id)
        
        # 更新特征索引
        features = feedback_data.get('features', {})
        for feature_name, feature_value in features.items():
            if feature_name not in self.feature_index:
                self.feature_index[feature_name] = {}
            
            # 将特征值转换为字符串用作索引键
            feature_key = str(feature_value)
            
            if feature_key not in self.feature_index[feature_name]:
                self.feature_index[feature_name][feature_key] = set()
            
            self.feature_index[feature_name][feature_key].add(query_id)
        
        # 保存到文件
        return self._save_to_file(query_id, self.cache[query_id])
    
    def _save_to_file(self, query_id: str, feedback_list: List[Dict[str, Any]]) -> bool:
        """
        将反馈数据保存到文件
        
        Args:
            query_id: 查询ID
            feedback_list: 反馈数据列表
            
        Returns:
            是否成功
        """
        # 创建文件名
        file_name = f"{query_id.replace('/', '_').replace(' ', '_')}"
        
        if self.use_json:
            file_path = os.path.join(self.storage_dir, f"{file_name}.json")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(feedback_list, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                self.logger.error(f"保存反馈数据失败: {e}")
                return False
        else:
            file_path = os.path.join(self.storage_dir, f"{file_name}.pkl")
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(feedback_list, f)
                return True
            except Exception as e:
                self.logger.error(f"保存反馈数据失败: {e}")
                return False
    
    def get(self, query_id: str) -> List[Dict[str, Any]]:
        """
        获取查询的反馈数据
        
        Args:
            query_id: 查询ID
            
        Returns:
            反馈数据列表
        """
        # 首先尝试从缓存获取
        if query_id in self.cache:
            return self.cache[query_id]
        
        # 尝试从文件加载
        file_name = f"{query_id.replace('/', '_').replace(' ', '_')}"
        
        if self.use_json:
            file_path = os.path.join(self.storage_dir, f"{file_name}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        feedback_list = json.load(f)
                        
                        # 更新缓存
                        self.cache[query_id] = feedback_list
                        
                        return feedback_list
                except Exception as e:
                    self.logger.error(f"加载反馈数据失败: {e}")
        else:
            file_path = os.path.join(self.storage_dir, f"{file_name}.pkl")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        feedback_list = pickle.load(f)
                        
                        # 更新缓存
                        self.cache[query_id] = feedback_list
                        
                        return feedback_list
                except Exception as e:
                    self.logger.error(f"加载反馈数据失败: {e}")
        
        return []
    
    def get_by_session(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取会话的所有反馈数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            查询ID到反馈数据列表的映射
        """
        if session_id not in self.session_index:
            return {}
        
        result = {}
        
        for query_id in self.session_index[session_id]:
            feedback_list = self.get(query_id)
            if feedback_list:
                result[query_id] = feedback_list
        
        return result
    
    def get_count_by_session(self, session_id: str) -> int:
        """
        获取会话的反馈数量
        
        Args:
            session_id: 会话ID
            
        Returns:
            反馈数量
        """
        if session_id not in self.session_index:
            return 0
        
        count = 0
        
        for query_id in self.session_index[session_id]:
            count += len(self.get(query_id))
        
        return count
    
    def get_by_feature(self, feature_name: str, feature_value: Any) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取具有特定特征的所有反馈数据
        
        Args:
            feature_name: 特征名称
            feature_value: 特征值
            
        Returns:
            查询ID到反馈数据列表的映射
        """
        if feature_name not in self.feature_index:
            return {}
        
        feature_key = str(feature_value)
        if feature_key not in self.feature_index[feature_name]:
            return {}
        
        result = {}
        
        for query_id in self.feature_index[feature_name][feature_key]:
            all_feedback = self.get(query_id)
            
            # 过滤出具有匹配特征的反馈
            matching_feedback = []
            for feedback in all_feedback:
                features = feedback.get('features', {})
                if features.get(feature_name) == feature_value:
                    matching_feedback.append(feedback)
            
            if matching_feedback:
                result[query_id] = matching_feedback
        
        return result
    
    def delete(self, query_id: str) -> bool:
        """
        删除查询的反馈数据
        
        Args:
            query_id: 查询ID
            
        Returns:
            是否成功
        """
        if query_id not in self.cache:
            return False
        
        # 获取反馈数据
        feedback_list = self.cache[query_id]
        
        # 更新会话索引
        for feedback in feedback_list:
            session_id = feedback.get('session_id', None)
            if session_id and session_id in self.session_index:
                self.session_index[session_id].discard(query_id)
                
                # 如果会话没有反馈了，删除该会话
                if not self.session_index[session_id]:
                    del self.session_index[session_id]
        
        # 更新特征索引
        for feedback in feedback_list:
            features = feedback.get('features', {})
            for feature_name, feature_value in features.items():
                if feature_name in self.feature_index:
                    feature_key = str(feature_value)
                    if feature_key in self.feature_index[feature_name]:
                        self.feature_index[feature_name][feature_key].discard(query_id)
                        
                        # 如果特征值没有反馈了，删除该特征值
                        if not self.feature_index[feature_name][feature_key]:
                            del self.feature_index[feature_name][feature_key]
                            
                            # 如果特征没有值了，删除该特征
                            if not self.feature_index[feature_name]:
                                del self.feature_index[feature_name]
        
        # 从缓存中删除
        del self.cache[query_id]
        
        # 删除文件
        file_name = f"{query_id.replace('/', '_').replace(' ', '_')}"
        
        if self.use_json:
            file_path = os.path.join(self.storage_dir, f"{file_name}.json")
        else:
            file_path = os.path.join(self.storage_dir, f"{file_name}.pkl")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                self.logger.error(f"删除反馈数据失败: {e}")
                return False
        
        return True
    
    def load_all(self) -> None:
        """加载所有反馈数据到缓存"""
        # 清空缓存和索引
        self.cache = {}
        self.session_index = {}
        self.feature_index = {}
        
        # 遍历存储目录
        for file_name in os.listdir(self.storage_dir):
            file_path = os.path.join(self.storage_dir, file_name)
            
            if os.path.isfile(file_path):
                # 提取查询ID
                if self.use_json and file_name.endswith('.json'):
                    query_id = file_name[:-5].replace('_', ' ')
                elif not self.use_json and file_name.endswith('.pkl'):
                    query_id = file_name[:-4].replace('_', ' ')
                else:
                    continue
                
                # 加载反馈数据
                self.get(query_id)
        
        self.logger.info(f"已加载 {len(self.cache)} 个查询的反馈数据")
    
    def search(
        self, 
        feature_filters: Optional[Dict[str, Any]] = None, 
        date_range: Optional[Tuple[datetime, datetime]] = None, 
        session_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        搜索反馈数据
        
        Args:
            feature_filters: 特征过滤条件
            date_range: 日期范围 (开始时间, 结束时间)
            session_id: 会话ID
            
        Returns:
            查询ID到反馈数据列表的映射
        """
        # 初始候选集
        candidate_query_ids = set(self.cache.keys())
        
        # 按会话过滤
        if session_id:
            if session_id not in self.session_index:
                return {}
            
            candidate_query_ids &= self.session_index[session_id]
        
        # 按特征过滤
        if feature_filters:
            for feature_name, feature_value in feature_filters.items():
                if feature_name not in self.feature_index:
                    return {}
                
                feature_key = str(feature_value)
                if feature_key not in self.feature_index[feature_name]:
                    return {}
                
                candidate_query_ids &= self.feature_index[feature_name][feature_key]
        
        # 收集结果
        result = {}
        
        for query_id in candidate_query_ids:
            feedback_list = self.get(query_id)
            
            # 按日期过滤
            if date_range:
                start_time, end_time = date_range
                filtered_feedback = []
                
                for feedback in feedback_list:
                    try:
                        timestamp = feedback.get('timestamp')
                        if not timestamp:
                            continue
                            
                        feedback_time = datetime.fromisoformat(timestamp)
                        if start_time <= feedback_time <= end_time:
                            filtered_feedback.append(feedback)
                    except:
                        continue
                
                if filtered_feedback:
                    result[query_id] = filtered_feedback
            else:
                result[query_id] = feedback_list
        
        return result 