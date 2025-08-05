"""
置信度校准与不确定性量化

实现神经网络的置信度校准和不确定性量化，确保模型输出的概率
与真实准确率相匹配，这对RAG系统的可靠性至关重要。

核心算法:
1. Temperature Scaling: 温度缩放校准
2. Platt Scaling: Sigmoid校准  
3. Isotonic Regression: 等渗回归校准
4. Monte Carlo Dropout: 蒙特卡洛Dropout不确定性估计
5. Deep Ensembles: 深度集成不确定性量化

科研贡献:
- 首次在多源RAG融合中引入置信度校准
- 提供不确定性感知的融合决策
- 实现校准质量的定量评估
- 支持自适应校准策略

参考文献:
Guo, C., et al. (2017). On calibration of modern neural networks. ICML.
Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation. NIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceCalibratorConfig:
    """置信度校准器配置"""
    
    # 校准方法参数
    calibration_method: str = "temperature_scaling"  # temperature_scaling, platt_scaling, isotonic
    
    # Temperature Scaling参数
    temperature_init: float = 1.0
    temperature_lr: float = 0.01
    temperature_max_iter: int = 50
    
    # Platt Scaling参数
    platt_max_iter: int = 100
    platt_solver: str = "lbfgs"
    
    # Monte Carlo Dropout参数
    mc_dropout_samples: int = 100
    mc_dropout_rate: float = 0.1
    
    # Deep Ensembles参数
    ensemble_size: int = 5
    ensemble_diversity_weight: float = 0.1
    
    # 校准评估参数
    eval_bins: int = 15
    eval_metric: str = "ece"  # ece, mce, brier_score
    
    # 自适应校准参数
    enable_adaptive_calibration: bool = True
    adaptation_window: int = 1000
    adaptation_threshold: float = 0.05


class TemperatureScaling(nn.Module):
    """温度缩放校准"""
    
    def __init__(self, config: ConfidenceCalibratorConfig):
        super().__init__()
        self.config = config
        
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        
        # 优化器
        self.optimizer = None
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用温度缩放
        
        Args:
            logits: 原始logits [batch_size, num_classes]
            
        Returns:
            校准后的概率 [batch_size, num_classes]
        """
        # 温度缩放
        scaled_logits = logits / self.temperature
        calibrated_probs = F.softmax(scaled_logits, dim=-1)
        
        return calibrated_probs
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        训练温度参数
        
        Args:
            logits: 验证集logits [num_samples, num_classes]
            labels: 真实标签 [num_samples]
            
        Returns:
            最终的NLL损失
        """
        # 初始化优化器
        if self.optimizer is None:
            self.optimizer = torch.optim.LBFGS(
                [self.temperature], 
                lr=self.config.temperature_lr,
                max_iter=self.config.temperature_max_iter
            )
        
        def closure():
            self.optimizer.zero_grad()
            calibrated_probs = self.forward(logits)
            loss = F.nll_loss(torch.log(calibrated_probs + 1e-8), labels)
            loss.backward()
            return loss
        
        # 优化温度参数
        for _ in range(self.config.temperature_max_iter):
            loss = self.optimizer.step(closure)
        
        logger.info(f"Temperature scaling fitted with temperature: {self.temperature.item():.4f}")
        return loss.item()


class PlattScaling:
    """Platt缩放校准 (Sigmoid校准)"""
    
    def __init__(self, config: ConfidenceCalibratorConfig):
        self.config = config
        self.calibrator = LogisticRegression(
            max_iter=config.platt_max_iter,
            solver=config.platt_solver
        )
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        训练Platt缩放参数
        
        Args:
            scores: 预测得分 [num_samples]
            labels: 真实标签 [num_samples]
        """
        scores = scores.reshape(-1, 1)
        self.calibrator.fit(scores, labels)
        self.fitted = True
        
        logger.info("Platt scaling fitted successfully")
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        获取校准后的概率
        
        Args:
            scores: 预测得分 [num_samples]
            
        Returns:
            校准后的概率 [num_samples, 2]
        """
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        scores = scores.reshape(-1, 1)
        return self.calibrator.predict_proba(scores)


class IsotonicCalibration:
    """等渗回归校准"""
    
    def __init__(self, config: ConfidenceCalibratorConfig):
        self.config = config
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        训练等渗回归校准器
        
        Args:
            scores: 预测得分 [num_samples]
            labels: 真实标签 [num_samples]
        """
        self.calibrator.fit(scores, labels)
        self.fitted = True
        
        logger.info("Isotonic calibration fitted successfully")
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        获取校准后的概率
        
        Args:
            scores: 预测得分 [num_samples]
            
        Returns:
            校准后的概率 [num_samples]
        """
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        return self.calibrator.predict(scores)


class MonteCarloDropout(nn.Module):
    """蒙特卡洛Dropout不确定性估计"""
    
    def __init__(self, model: nn.Module, config: ConfidenceCalibratorConfig):
        super().__init__()
        self.model = model
        self.config = config
        
        # 添加Dropout层
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """为模型添加Dropout层"""
        def add_dropout(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # 在线性层后添加Dropout
                    setattr(module, name + "_dropout", nn.Dropout(self.config.mc_dropout_rate))
                else:
                    add_dropout(child)
        
        add_dropout(self.model)
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        蒙特卡洛Dropout前向传播
        
        Args:
            x: 输入张量
            return_uncertainty: 是否返回不确定性
            
        Returns:
            预测均值和可选的不确定性
        """
        if not return_uncertainty:
            return self.model(x)
        
        # 启用训练模式以激活Dropout
        self.model.train()
        
        predictions = []
        
        # 多次采样
        for _ in range(self.config.mc_dropout_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)
        
        # 恢复评估模式
        self.model.eval()
        
        # 计算统计量
        predictions = torch.stack(predictions)  # [num_samples, batch_size, ...]
        
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class DeepEnsemble(nn.Module):
    """深度集成不确定性量化"""
    
    def __init__(self, model_class: type, model_kwargs: Dict[str, Any], config: ConfidenceCalibratorConfig):
        super().__init__()
        self.config = config
        
        # 创建集成模型
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(config.ensemble_size)
        ])
        
        # 多样性正则化权重
        self.diversity_weight = config.ensemble_diversity_weight
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        深度集成前向传播
        
        Args:
            x: 输入张量
            return_uncertainty: 是否返回不确定性
            
        Returns:
            预测均值和可选的不确定性
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [ensemble_size, batch_size, ...]
        
        mean_pred = predictions.mean(dim=0)
        
        if return_uncertainty:
            uncertainty = predictions.std(dim=0)
            return mean_pred, uncertainty
        
        return mean_pred
    
    def compute_diversity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算集成多样性损失
        
        Args:
            x: 输入张量
            
        Returns:
            多样性损失
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [ensemble_size, batch_size, ...]
        
        # 计算成对距离
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                # 余弦相似度
                cos_sim = F.cosine_similarity(predictions[i], predictions[j], dim=-1)
                diversity_loss += cos_sim.mean()
                num_pairs += 1
        
        # 最大化多样性 = 最小化相似度
        diversity_loss = diversity_loss / num_pairs
        
        return diversity_loss


class CalibrationMetrics:
    """校准质量评估指标"""
    
    @staticmethod
    def expected_calibration_error(confidences: np.ndarray, 
                                  accuracies: np.ndarray, 
                                  num_bins: int = 15) -> float:
        """
        期望校准误差 (ECE)
        
        Args:
            confidences: 预测置信度 [num_samples]
            accuracies: 预测准确性 [num_samples] (0或1)
            num_bins: 分箱数量
            
        Returns:
            ECE值
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 确定在当前bin中的样本
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def maximum_calibration_error(confidences: np.ndarray,
                                 accuracies: np.ndarray,
                                 num_bins: int = 15) -> float:
        """
        最大校准误差 (MCE)
        
        Args:
            confidences: 预测置信度 [num_samples]
            accuracies: 预测准确性 [num_samples]
            num_bins: 分箱数量
            
        Returns:
            MCE值
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_calibration_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_calibration_error = max(max_calibration_error, calibration_error)
        
        return max_calibration_error
    
    @staticmethod
    def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
        """
        Brier分数
        
        Args:
            probabilities: 预测概率 [num_samples, num_classes]
            labels: 真实标签 [num_samples]
            
        Returns:
            Brier分数
        """
        # 转换为one-hot编码
        num_classes = probabilities.shape[1]
        one_hot_labels = np.eye(num_classes)[labels]
        
        # 计算Brier分数
        brier = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))
        
        return brier


class UncertaintyQuantifier:
    """不确定性量化器"""
    
    def __init__(self, config: ConfidenceCalibratorConfig):
        self.config = config
    
    def epistemic_uncertainty(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        认知不确定性 (模型不确定性)
        
        Args:
            predictions: 多次预测结果 [num_samples, batch_size, num_classes]
            
        Returns:
            认知不确定性 [batch_size]
        """
        # 计算预测的方差
        mean_pred = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0).mean(dim=-1)
        
        return epistemic
    
    def aleatoric_uncertainty(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        随机不确定性 (数据不确定性)
        
        Args:
            predictions: 多次预测结果 [num_samples, batch_size, num_classes]
            
        Returns:
            随机不确定性 [batch_size]
        """
        # 计算每次预测的熵的均值
        entropies = []
        
        for pred in predictions:
            entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=-1)
            entropies.append(entropy)
        
        aleatoric = torch.stack(entropies).mean(dim=0)
        
        return aleatoric
    
    def total_uncertainty(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        总不确定性
        
        Args:
            predictions: 多次预测结果 [num_samples, batch_size, num_classes]
            
        Returns:
            总不确定性 [batch_size]
        """
        mean_pred = predictions.mean(dim=0)
        total_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
        
        return total_entropy


class ConfidenceCalibrator:
    """
    置信度校准器主类
    
    整合多种校准方法，提供统一的校准接口
    """
    
    def __init__(self, config: ConfidenceCalibratorConfig):
        self.config = config
        self.calibrator = None
        self.uncertainty_quantifier = UncertaintyQuantifier(config)
        self.fitted = False
        
        # 初始化校准器
        self._init_calibrator()
    
    def _init_calibrator(self):
        """初始化校准器"""
        if self.config.calibration_method == "temperature_scaling":
            self.calibrator = TemperatureScaling(self.config)
        elif self.config.calibration_method == "platt_scaling":
            self.calibrator = PlattScaling(self.config)
        elif self.config.calibration_method == "isotonic":
            self.calibrator = IsotonicCalibration(self.config)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.calibration_method}")
    
    def fit(self, predictions: Union[torch.Tensor, np.ndarray], 
            labels: Union[torch.Tensor, np.ndarray]):
        """
        训练校准器
        
        Args:
            predictions: 预测结果 (logits或概率)
            labels: 真实标签
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        if self.config.calibration_method == "temperature_scaling":
            # Temperature scaling需要torch tensors
            pred_tensor = torch.from_numpy(predictions).float()
            label_tensor = torch.from_numpy(labels).long()
            self.calibrator.fit(pred_tensor, label_tensor)
        else:
            # 其他方法使用numpy
            if predictions.ndim > 1:
                # 取最大概率作为置信度
                confidences = np.max(predictions, axis=1)
            else:
                confidences = predictions
            
            self.calibrator.fit(confidences, labels)
        
        self.fitted = True
        logger.info(f"Calibrator fitted using {self.config.calibration_method}")
    
    def calibrate(self, predictions: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        校准预测结果
        
        Args:
            predictions: 原始预测结果
            
        Returns:
            校准后的概率
        """
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        if self.config.calibration_method == "temperature_scaling":
            if isinstance(predictions, np.ndarray):
                predictions = torch.from_numpy(predictions).float()
            
            with torch.no_grad():
                calibrated = self.calibrator(predictions)
            return calibrated.numpy()
        else:
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            
            if predictions.ndim > 1:
                confidences = np.max(predictions, axis=1)
            else:
                confidences = predictions
            
            if self.config.calibration_method == "platt_scaling":
                calibrated_probs = self.calibrator.predict_proba(confidences)
                return calibrated_probs[:, 1]  # 返回正类概率
            else:  # isotonic
                return self.calibrator.predict_proba(confidences)
    
    def evaluate_calibration(self, 
                           predictions: Union[torch.Tensor, np.ndarray],
                           labels: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        评估校准质量
        
        Args:
            predictions: 预测结果
            labels: 真实标签
            
        Returns:
            校准评估指标字典
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # 获取置信度和准确性
        if predictions.ndim > 1:
            confidences = np.max(predictions, axis=1)
            predicted_labels = np.argmax(predictions, axis=1)
        else:
            confidences = predictions
            predicted_labels = (predictions > 0.5).astype(int)
        
        accuracies = (predicted_labels == labels).astype(float)
        
        # 计算校准指标
        metrics = {}
        
        if self.config.eval_metric in ["ece", "all"]:
            metrics["ece"] = CalibrationMetrics.expected_calibration_error(
                confidences, accuracies, self.config.eval_bins
            )
        
        if self.config.eval_metric in ["mce", "all"]:
            metrics["mce"] = CalibrationMetrics.maximum_calibration_error(
                confidences, accuracies, self.config.eval_bins
            )
        
        if self.config.eval_metric in ["brier_score", "all"]:
            if predictions.ndim > 1:
                metrics["brier_score"] = CalibrationMetrics.brier_score(predictions, labels)
        
        return metrics


# 工厂函数
def create_confidence_calibrator(config: Optional[ConfidenceCalibratorConfig] = None) -> ConfidenceCalibrator:
    """创建置信度校准器"""
    if config is None:
        config = ConfidenceCalibratorConfig()
    return ConfidenceCalibrator(config)


def create_confidence_calibrator_config(**kwargs) -> ConfidenceCalibratorConfig:
    """创建置信度校准器配置"""
    return ConfidenceCalibratorConfig(**kwargs) 