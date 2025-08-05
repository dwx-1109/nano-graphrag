"""
置信度校准模块

实现多种置信度校准算法，提升分类器的置信度可靠性。
这是项目核心创新点的关键实现。

支持的校准方法：
- Temperature Scaling: 通过学习温度参数调整logits
- Platt Scaling: 通过sigmoid函数校准
- Isotonic Regression: 非参数校准方法

参考文献：
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Platt (1999). Probabilistic outputs for support vector machines.
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
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """校准配置类"""
    
    # 校准方法选择
    method: str = "temperature_scaling"  # temperature_scaling, platt_scaling, isotonic
    
    # Temperature Scaling参数
    temperature_init: float = 1.0
    temperature_lr: float = 0.001
    temperature_max_iter: int = 50
    temperature_patience: int = 5
    
    # Platt Scaling参数
    platt_max_iter: int = 100
    platt_C: float = 1.0
    
    # 校准评估参数
    eval_bins: int = 15
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 保存路径
    save_path: Optional[str] = None


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling校准器
    
    通过学习一个温度参数T来调整logits：
    calibrated_logits = logits / T
    
    参考: Guo et al. (2017) On Calibration of Modern Neural Networks
    """
    
    def __init__(self, config: CalibrationConfig):
        super().__init__()
        self.config = config
        
        # 温度参数 (可学习)
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature_init)
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用温度缩放
        
        Args:
            logits: 原始logits [batch_size, num_classes]
            
        Returns:
            校准后的logits
        """
        return logits / self.temperature
    
    def fit(self, 
            logits: torch.Tensor, 
            labels: torch.Tensor,
            verbose: bool = True) -> Dict[str, float]:
        """
        在验证集上拟合温度参数
        
        Args:
            logits: 验证集logits [N, num_classes]
            labels: 验证集标签 [N]
            verbose: 是否打印训练信息
            
        Returns:
            训练历史信息
        """
        # 设置设备
        device = torch.device(self.config.device)
        self.to(device)
        logits = logits.to(device)
        labels = labels.to(device)
        
        # 设置优化器
        optimizer = torch.optim.Adam([self.temperature], lr=self.config.temperature_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.config.temperature_patience
        )
        
        # 训练历史
        history = {
            'loss': [],
            'temperature': [],
            'ece': []
        }
        
        best_loss = float('inf')
        patience_count = 0
        
        for epoch in range(self.config.temperature_max_iter):
            # 前向传播
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 学习率调度
            scheduler.step(loss)
            
            # 计算ECE
            with torch.no_grad():
                ece = self._compute_ece(scaled_logits, labels)
            
            # 记录历史
            history['loss'].append(loss.item())
            history['temperature'].append(self.temperature.item())
            history['ece'].append(ece)
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={loss.item():.6f}, "
                          f"Temperature={self.temperature.item():.4f}, ECE={ece:.4f}")
            
            # 早停检查
            if loss < best_loss:
                best_loss = loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.config.temperature_patience * 2:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        if verbose:
            logger.info(f"Temperature scaling completed. Final temperature: {self.temperature.item():.4f}")
        
        return history
    
    def _compute_ece(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """计算Expected Calibration Error"""
        return compute_ece(logits, labels, n_bins=self.config.eval_bins)


class PlattScaling:
    """
    Platt Scaling校准器
    
    使用sigmoid函数将分数映射到校准的概率：
    P(y=1|s) = 1 / (1 + exp(A*s + B))
    
    参考: Platt (1999) Probabilistic outputs for support vector machines
    """
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.calibrator = LogisticRegression(
            C=config.platt_C,
            max_iter=config.platt_max_iter,
            solver='liblinear'
        )
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        拟合Platt Scaling参数
        
        Args:
            scores: 预测分数 [N]
            labels: 真实标签 [N]
            
        Returns:
            拟合信息
        """
        # 确保二分类
        if len(np.unique(labels)) > 2:
            raise ValueError("Platt Scaling currently only supports binary classification")
        
        # 拟合逻辑回归
        scores = scores.reshape(-1, 1)
        self.calibrator.fit(scores, labels)
        self.fitted = True
        
        # 计算拟合后的准确率
        calibrated_probs = self.calibrator.predict_proba(scores)[:, 1]
        ece = compute_ece_binary(calibrated_probs, labels)
        
        logger.info(f"Platt Scaling fitted. ECE: {ece:.4f}")
        
        return {
            'ece': ece,
            'coef': self.calibrator.coef_[0][0],
            'intercept': self.calibrator.intercept_[0]
        }
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """预测校准后的概率"""
        if not self.fitted:
            raise ValueError("Must fit calibrator before prediction")
        
        scores = scores.reshape(-1, 1)
        return self.calibrator.predict_proba(scores)[:, 1]


class IsotonicCalibration:
    """
    Isotonic Regression校准器
    
    使用等保序回归进行非参数校准
    """
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """拟合Isotonic校准器"""
        self.calibrator.fit(scores, labels)
        self.fitted = True
        
        # 计算校准后ECE
        calibrated_probs = self.calibrator.predict(scores)
        ece = compute_ece_binary(calibrated_probs, labels)
        
        logger.info(f"Isotonic Regression fitted. ECE: {ece:.4f}")
        
        return {'ece': ece}
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """预测校准后的概率"""
        if not self.fitted:
            raise ValueError("Must fit calibrator before prediction")
        
        return self.calibrator.predict(scores)


class ConfidenceCalibrator:
    """
    统一的置信度校准器
    
    集成多种校准方法，为ComplexityClassifier提供置信度校准功能
    """
    
    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.calibrator = None
        self.fitted = False
        
        # 初始化对应的校准器
        if self.config.method == "temperature_scaling":
            self.calibrator = TemperatureScaling(self.config)
        elif self.config.method == "platt_scaling":
            self.calibrator = PlattScaling(self.config)
        elif self.config.method == "isotonic":
            self.calibrator = IsotonicCalibration(self.config)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.method}")
        
        logger.info(f"Initialized {self.config.method} calibrator")
    
    def fit(self, 
            logits_or_scores: Union[torch.Tensor, np.ndarray], 
            labels: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        拟合校准器
        
        Args:
            logits_or_scores: logits或分数
            labels: 真实标签
            
        Returns:
            拟合信息
        """
        # 转换数据类型
        if isinstance(logits_or_scores, torch.Tensor):
            if self.config.method == "temperature_scaling":
                # Temperature scaling需要完整的logits
                history = self.calibrator.fit(logits_or_scores, labels)
            else:
                # 其他方法只需要最大概率
                probs = F.softmax(logits_or_scores, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                is_correct = torch.argmax(logits_or_scores, dim=-1) == labels
                
                max_probs_np = max_probs.cpu().numpy()
                is_correct_np = is_correct.cpu().numpy().astype(int)
                
                history = self.calibrator.fit(max_probs_np, is_correct_np)
        else:
            # numpy数组
            history = self.calibrator.fit(logits_or_scores, labels)
        
        self.fitted = True
        
        # 保存校准器
        if self.config.save_path:
            self.save(self.config.save_path)
        
        return history
    
    def calibrate(self, 
                  logits_or_scores: Union[torch.Tensor, np.ndarray], 
                  probabilities: Optional[Dict[str, float]] = None) -> Union[float, torch.Tensor]:
        """
        校准置信度
        
        Args:
            logits_or_scores: logits或分数
            probabilities: 原始概率分布（用于temperature scaling）
            
        Returns:
            校准后的置信度
        """
        if not self.fitted:
            logger.warning("Calibrator not fitted, returning original confidence")
            if isinstance(logits_or_scores, torch.Tensor):
                return torch.max(F.softmax(logits_or_scores, dim=-1))
            else:
                return float(logits_or_scores)
        
        if self.config.method == "temperature_scaling":
            if isinstance(logits_or_scores, torch.Tensor):
                # 确保设备一致性
                device = next(self.calibrator.parameters()).device
                logits_or_scores = logits_or_scores.to(device)
                scaled_logits = self.calibrator(logits_or_scores)
                calibrated_probs = F.softmax(scaled_logits, dim=-1)
                return torch.max(calibrated_probs, dim=-1)[0]
            else:
                # 如果输入是numpy数组，且是一维的confidence scores，直接使用其他方法
                # Temperature scaling需要完整的logits，不适用于单一confidence scores
                logger.warning("Temperature scaling requires logits, falling back to identity calibration for confidence scores")
                return logits_or_scores
        else:
            # Platt scaling或Isotonic
            if isinstance(logits_or_scores, torch.Tensor):
                max_prob = torch.max(F.softmax(logits_or_scores, dim=-1), dim=-1)[0]
                max_prob_np = max_prob.cpu().numpy()
            else:
                max_prob_np = logits_or_scores
            
            calibrated_prob = self.calibrator.predict_proba(max_prob_np)
            return float(calibrated_prob) if np.isscalar(calibrated_prob) else calibrated_prob
    
    def save(self, path: str):
        """保存校准器"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.config.method == "temperature_scaling":
            torch.save({
                'calibrator_state_dict': self.calibrator.state_dict(),
                'config': self.config,
                'fitted': self.fitted
            }, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'calibrator': self.calibrator,
                    'config': self.config,
                    'fitted': self.fitted
                }, f)
        
        logger.info(f"Calibrator saved to {path}")
    
    def load(self, path: str):
        """加载校准器"""
        if self.config.method == "temperature_scaling":
            # 使用weights_only=False来兼容旧版本PyTorch模型
            checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
            self.calibrator.load_state_dict(checkpoint['calibrator_state_dict'])
            self.fitted = checkpoint['fitted']
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.calibrator = data['calibrator']
                self.fitted = data['fitted']
        
        logger.info(f"Calibrator loaded from {path}")


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    计算Expected Calibration Error (ECE)
    
    Args:
        logits: 预测logits [N, num_classes]
        labels: 真实标签 [N]
        n_bins: 分箱数量
        
    Returns:
        ECE值
    """
    with torch.no_grad():
        # 计算预测概率和预测类别
        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = torch.max(probs, dim=-1)
        
        # 计算准确率
        accuracies = (predictions == labels).float()
        
        # 分箱
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 确定当前bin中的样本
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # 计算当前bin的准确率和平均置信度
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                
                # 累加ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()


def compute_ece_binary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """计算二分类ECE"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def create_confidence_calibrator(method: str = "temperature_scaling", **kwargs) -> ConfidenceCalibrator:
    """创建置信度校准器的工厂函数"""
    config = CalibrationConfig(method=method, **kwargs)
    return ConfidenceCalibrator(config)


# 导出所有公共接口
__all__ = [
    'CalibrationConfig',
    'TemperatureScaling', 
    'PlattScaling',
    'IsotonicCalibration',
    'ConfidenceCalibrator',
    'compute_ece',
    'compute_ece_binary',
    'create_confidence_calibrator'
] 