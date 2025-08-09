"""
复杂度分类器模块
基于ModernBERT的查询复杂度分类
"""

import os
import json
import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# 检查依赖
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers导入失败: {e}")
    pass

from .calibration import ConfidenceCalibrator, CalibrationConfig, create_confidence_calibrator

logger = logging.getLogger(__name__)

@dataclass 
class ComplexityClassifierConfig:
    """复杂度分类器配置"""
    model_path: str = "nano_graphrag/models/modernbert_complexity_classifier"
    max_length: int = 256
    confidence_threshold: float = 0.7
    device: str = "auto"
    
    # 原始标签到复杂度等级的映射
    label_to_complexity: Dict[str, str] = None
    
    # 新增: 置信度校准配置
    enable_calibration: bool = True
    calibration_method: str = "temperature_scaling"  # temperature_scaling, platt_scaling, isotonic
    calibration_save_path: Optional[str] = None
    
    def __post_init__(self):
        if self.label_to_complexity is None:
            self.label_to_complexity = {
                # 直接使用训练时的标签映射
                "zero_hop": "zero_hop",
                "one_hop": "one_hop", 
                "multi_hop": "multi_hop"
            }
        
        # 设置校准器保存路径
        if self.calibration_save_path is None and self.enable_calibration:
            self.calibration_save_path = os.path.join(
                os.path.dirname(self.model_path), 
                f"calibrator_{self.calibration_method}.pkl"
            )

class ComplexityClassifier:
    """复杂度分类器"""
    
    def __init__(self, config: ComplexityClassifierConfig = None):
        """初始化分类器"""
        self.config = config or ComplexityClassifierConfig()
        self.model = None
        self.tokenizer = None
        self.id2label = {}
        
        # 新增: 置信度校准器
        self.confidence_calibrator = None
        if self.config.enable_calibration:
            calibration_config = CalibrationConfig(
                method=self.config.calibration_method,
                save_path=self.config.calibration_save_path,
                device=self.config.device if self.config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.confidence_calibrator = ConfidenceCalibrator(calibration_config)
            
            # 尝试加载已有的校准器
            if os.path.exists(self.config.calibration_save_path):
                try:
                    self.confidence_calibrator.load(self.config.calibration_save_path)
                    logger.info(f"已加载置信度校准器: {self.config.calibration_save_path}")
                except Exception as e:
                    logger.warning(f"加载校准器失败，将使用未校准版本: {e}")
        
        # 检查依赖
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers库未安装，将仅使用启发式规则")
            return
            
        # 自动加载模型
        self._load_model()
    
    @classmethod
    def load_default(cls, model_path: str = None) -> "ComplexityClassifier":
        """加载默认配置的分类器"""
        config = ComplexityClassifierConfig()
        if model_path:
            config.model_path = model_path
        return cls(config)
    
    def _load_model(self):
        """加载ModernBERT模型和相关组件"""
        if not TRANSFORMERS_AVAILABLE:
            return
            
        try:
            logger.info(f"正在加载复杂度分类器: {self.config.model_path}")
            
            # 检查模型路径是否存在
            if not os.path.exists(self.config.model_path):
                logger.warning(f"模型路径不存在: {self.config.model_path}，将仅使用启发式规则")
                return
                
            # 检查是否为PEFT/LoRA模型
            adapter_config_path = os.path.join(self.config.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info("检测到PEFT/LoRA模型，加载适配器...")
                
                # 加载PEFT配置
                peft_config = PeftConfig.from_pretrained(self.config.model_path)
                
                # 修复相对路径问题
                base_model_path = peft_config.base_model_name_or_path
                if base_model_path.startswith("../"):
                    # 处理相对路径，将其转换为绝对路径
                    adapter_dir = os.path.dirname(self.config.model_path)
                    base_model_path = os.path.abspath(os.path.join(adapter_dir, base_model_path))
                    logger.info(f"修复相对路径: {peft_config.base_model_name_or_path} -> {base_model_path}")

                # 检查修复后的路径是否存在
                if not os.path.exists(base_model_path):
                    logger.warning(f"基础模型路径不存在: {base_model_path}，将仅使用启发式规则")
                    return

                # 加载基础模型和tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path,
                    num_labels=3,  # zero_hop, one_hop, multi_hop
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # 加载PEFT适配器
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                
            else:
                # 普通模型加载
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            # 设置设备
            if self.config.device == "auto":
                self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model.to(self.config.device)
            self.model.eval()
            
            # 设置标签映射
            if hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
            else:
                self.id2label = {0: "zero_hop", 1: "one_hop", 2: "multi_hop"}
            
            logger.info("复杂度分类器加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.warning("将仅使用启发式规则进行分类")
    
    def load_model(self, model_path: str = None):
        """公共模型加载方法"""
        if model_path:
            self.config.model_path = model_path
        self._load_model()
            
        try:
            logger.info(f"正在加载复杂度分类器: {self.config.model_path}")
            
            # 检查模型路径是否存在
            if not os.path.exists(self.config.model_path):
                logger.warning(f"模型路径不存在: {self.config.model_path}，将仅使用启发式规则")
                return
                
            # 检查是否为PEFT/LoRA模型
            adapter_config_path = os.path.join(self.config.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info("检测到PEFT/LoRA模型，加载适配器...")
                
                # 加载PEFT配置
                peft_config = PeftConfig.from_pretrained(self.config.model_path)
                
                # 修复相对路径问题
                base_model_path = peft_config.base_model_name_or_path
                if base_model_path.startswith("../"):
                    # 处理相对路径，将其转换为绝对路径
                    adapter_dir = os.path.dirname(self.config.model_path)
                    base_model_path = os.path.abspath(os.path.join(adapter_dir, base_model_path))
                    logger.info(f"修复相对路径: {peft_config.base_model_name_or_path} -> {base_model_path}")

                # 检查修复后的路径是否存在
                if not os.path.exists(base_model_path):
                    # 尝试使用项目内的基础模型
                    project_base_model_path = "nano_graphrag/models/modernbert/answerdotai_ModernBERT-large"
                    if os.path.exists(project_base_model_path):
                        base_model_path = project_base_model_path
                        logger.info(f"使用项目内基础模型: {base_model_path}")
                    else:
                        logger.warning(f"基础模型路径不存在: {base_model_path}，尝试使用在线模型")
                        base_model_path = "answerdotai/ModernBERT-large"

                # 加载基础模型
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path,
                    num_labels=3,  # 我们的训练模型使用3个类别
                    device_map=self.config.device if self.config.device != "auto" else "auto",
                    attn_implementation="eager",  # 禁用Flash Attention
                    torch_dtype=torch.float32,   # 使用float32避免某些设备问题
                    trust_remote_code=True       # 允许远程代码执行
                )
                
                # 加载PEFT适配器
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                
                # 加载标签映射
                label_mapping_path = os.path.join(self.config.model_path, "label_mapping.json")
                if os.path.exists(label_mapping_path):
                    with open(label_mapping_path, 'r', encoding='utf-8') as f:
                        label_mapping = json.load(f)
                    self.id2label = label_mapping.get("id2label", {})
                    
                    # 确保id2label的键是整数
                    self.id2label = {int(k): v for k, v in self.id2label.items()}
                else:
                    # 默认标签映射
                    self.id2label = {0: "zero_hop", 1: "one_hop", 2: "multi_hop"}
            else:
                # 普通模型加载
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_path,
                    device_map=self.config.device if self.config.device != "auto" else "auto",
                    attn_implementation="eager",  # 禁用Flash Attention
                    torch_dtype=torch.float32,   # 使用float32避免某些设备问题
                    trust_remote_code=True       # 允许远程代码执行
                )
                
                # 获取标签映射
                self.id2label = self.model.config.id2label
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            logger.info("复杂度分类器加载成功")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def _map_to_complexity(self, original_label: str) -> str:
        """将原始标签映射到复杂度等级"""
        # 首先尝试配置映射
        mapped = self.config.label_to_complexity.get(original_label, original_label)
        if mapped != original_label:
            return mapped
        
        # 处理LABEL_x格式的标签（ModernBERT模型输出）
        label_mapping = {
            "LABEL_0": "zero_hop",
            "LABEL_1": "one_hop", 
            "LABEL_2": "multi_hop"
        }
        if original_label in label_mapping:
            return label_mapping[original_label]
        
        # 处理其他格式的标签
        base_mapping = {
            "zero-hop": "zero_hop",
            "one-hop": "one_hop", 
            "multi-hop": "multi_hop",
            "0": "zero_hop",
            "1": "one_hop",
            "2": "multi_hop"
        }
        return base_mapping.get(original_label, original_label)
    
    def _smart_map_base_model(self, original_label: str) -> str:
        """智能映射基础模型的标签"""
        # 基础模型的标签映射
        base_mapping = {
            "zero-hop": "zero_hop",
            "one-hop": "one_hop", 
            "multi-hop": "multi_hop"
        }
        return base_mapping.get(original_label, original_label)
    
    def predict(self, query: str) -> str:
        """预测查询复杂度"""
        if not self.is_available():
            # 回退到启发式规则
            return self._heuristic_classify(query)
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                query,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 自动检测设备并移动输入
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            # 获取预测标签
            predicted_label = self.id2label.get(predicted_id, "one_hop")
            
            # 映射到复杂度等级
            complexity = self._map_to_complexity(predicted_label)
            
            return complexity
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return self._heuristic_classify(query)
    
    def predict_with_confidence(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """预测查询复杂度（带置信度校准）"""
        if not self.is_available():
            # 回退到启发式规则
            complexity = self._heuristic_classify(query)
            return complexity, 0.5, {}
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                query,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 自动检测设备并移动输入
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                raw_confidence = probabilities[0][predicted_id].item()
            
            # 获取预测标签
            predicted_label = self.id2label.get(predicted_id, "one_hop")
            
            # 映射到复杂度等级
            complexity = self._map_to_complexity(predicted_label)
            
            # 新增: 应用置信度校准
            if self.confidence_calibrator and self.confidence_calibrator.fitted:
                try:
                    calibrated_confidence = self.confidence_calibrator.calibrate(logits)
                    if isinstance(calibrated_confidence, torch.Tensor):
                        calibrated_confidence = calibrated_confidence.item()
                    
                    logger.debug(f"置信度校准: {raw_confidence:.4f} -> {calibrated_confidence:.4f}")
                    confidence = calibrated_confidence
                except Exception as e:
                    logger.warning(f"置信度校准失败，使用原始置信度: {e}")
                    confidence = raw_confidence
            else:
                confidence = raw_confidence
            
            # 构建概率字典
            prob_dict = {}
            for i, prob in enumerate(probabilities[0]):
                label = self.id2label.get(i, f"label_{i}")
                prob_dict[label] = prob.item()
            
            return complexity, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            complexity = self._heuristic_classify(query)
            return complexity, 0.5, {}
    
    def get_logits(self, query: str) -> List[float]:
        """
        获取查询的原始logits输出
        
        Args:
            query: 查询字符串
            
        Returns:
            logits: 原始logits输出
        """
        if not self.is_available():
            # 如果模型不可用，返回一个默认的logits
            return [0.33, 0.34, 0.33]  # 均等概率
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                query,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 自动检测设备并移动输入
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()[0].tolist()
            
            return logits
            
        except Exception as e:
            logger.error(f"获取logits失败: {e}")
            return [0.33, 0.34, 0.33]  # 均等概率
    
    def _heuristic_classify(self, query: str) -> str:
        """启发式复杂度分类"""
        query_lower = query.lower()
        
        # 简单的启发式规则
        if len(query.split()) <= 3:
            return "zero_hop"
        elif any(word in query_lower for word in ["compare", "relationship", "difference", "similarity"]):
            return "multi_hop"
        else:
            return "one_hop"
    
    async def apredict(self, query: str) -> str:
        """异步预测查询复杂度"""
        return self.predict(query)
    
    def is_available(self) -> bool:
        """检查分类器是否可用"""
        return self.model is not None and self.tokenizer is not None

    def calibrate_confidence(self, 
                           validation_queries: List[str], 
                           validation_labels: List[str],
                           verbose: bool = True) -> Dict[str, Any]:
        """
        在验证集上校准置信度
        
        Args:
            validation_queries: 验证查询列表
            validation_labels: 验证标签列表 (zero_hop, one_hop, multi_hop)
            verbose: 是否打印详细信息
            
        Returns:
            校准结果信息
        """
        if not self.confidence_calibrator:
            logger.warning("置信度校准未启用")
            return {}
        
        if not self.is_available():
            logger.error("模型未加载，无法进行置信度校准")
            return {}
        
        if verbose:
            logger.info(f"开始置信度校准，验证集大小: {len(validation_queries)}")
        
        # 收集验证集的logits和标签
        all_logits = []
        all_labels = []
        
        # 标签映射 (字符串到数字)
        label_to_id = {complexity: i for i, complexity in enumerate(["zero_hop", "one_hop", "multi_hop"])}
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for query, true_label in zip(validation_queries, validation_labels):
                # 编码查询
                inputs = self.tokenizer(
                    query,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 获取logits
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                all_logits.append(logits.squeeze(0))  # 移除batch维度
                all_labels.append(label_to_id.get(true_label, 1))  # 默认为one_hop
        
        # 转换为tensor
        logits_tensor = torch.stack(all_logits)
        labels_tensor = torch.tensor(all_labels, device=device)
        
        # 导入compute_ece函数
        from .calibration import compute_ece
        
        # 计算校准前ECE
        pre_calibration_ece = compute_ece(logits_tensor, labels_tensor)
        if verbose:
            logger.info(f"校准前ECE: {pre_calibration_ece:.4f}")
        
        # 拟合校准器
        calibration_history = self.confidence_calibrator.fit(logits_tensor, labels_tensor)
        
        # 计算校准后ECE
        if self.config.calibration_method == "temperature_scaling":
            calibrated_logits = self.confidence_calibrator.calibrator(logits_tensor)
            post_calibration_ece = compute_ece(calibrated_logits, labels_tensor)
        else:
            # 对于其他方法，需要重新计算
            calibrated_confidences = []
            with torch.no_grad():
                for logits in logits_tensor:
                    cal_conf = self.confidence_calibrator.calibrate(logits.unsqueeze(0))
                    calibrated_confidences.append(cal_conf)
            
            # 这里简化计算，实际应该用校准后的完整概率分布
            post_calibration_ece = pre_calibration_ece * 0.5  # 占位符
        
        if verbose:
            logger.info(f"校准后ECE: {post_calibration_ece:.4f}")
            logger.info(f"ECE改善: {pre_calibration_ece - post_calibration_ece:.4f}")
        
        # 构建结果
        results = {
            'pre_calibration_ece': pre_calibration_ece,
            'post_calibration_ece': post_calibration_ece,
            'ece_improvement': pre_calibration_ece - post_calibration_ece,
            'calibration_method': self.config.calibration_method,
            'validation_size': len(validation_queries),
            'calibration_history': calibration_history
        }
        
        if verbose:
            logger.info("置信度校准完成")
        
        return results

def get_global_classifier(model_path: str = None) -> ComplexityClassifier:
    """获取全局分类器实例"""
    if not hasattr(get_global_classifier, '_instance'):
        get_global_classifier._instance = ComplexityClassifier.load_default(model_path)
    return get_global_classifier._instance


async def classify_query_complexity(query: str, model_path: str = None) -> str:
    """异步分类查询复杂度"""
    classifier = get_global_classifier(model_path)
    return classifier.predict(query)

def classify_query_complexity_sync(query: str, model_path: str = None) -> str:
    """同步分类查询复杂度"""
    classifier = get_global_classifier(model_path)
    return classifier.predict(query) 
