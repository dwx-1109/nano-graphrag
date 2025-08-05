#!/usr/bin/env python3
"""
阶段1依赖检查和安装脚本
检查并安装阶段1实验所需的所有依赖
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import importlib.util
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyChecker:
    """依赖检查器"""
    
    def __init__(self):
        self.missing_deps = []
        self.gpu_available = torch.cuda.is_available()
        self.model_path = "nano_graphrag/models/modernbert_complexity_classifier"
        
    def check_python_packages(self) -> bool:
        """检查Python包依赖"""
        logger.info("检查Python包依赖...")
        
        required_packages = {
            'transformers': '4.30.0',
            'torch': '2.0.0', 
            'datasets': '2.12.0',
            'sentence_transformers': '2.2.0',
            'peft': '0.4.0',
            'huggingface_hub': '0.15.0',
            'pandas': '1.5.0',
            'numpy': '1.21.0',
            'tqdm': '4.64.0',
            'scikit-learn': '1.2.0'
        }
        
        for package, min_version in required_packages.items():
            try:
                spec = importlib.util.find_spec(package)
                if spec is None:
                    logger.error(f"❌ 缺少包: {package}>={min_version}")
                    self.missing_deps.append(f"{package}>={min_version}")
                else:
                    logger.info(f"✅ 已安装: {package}")
            except Exception as e:
                logger.error(f"❌ 检查包 {package} 时出错: {e}")
                self.missing_deps.append(f"{package}>={min_version}")
        
        return len(self.missing_deps) == 0
    
    def check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        logger.info("检查GPU可用性...")
        
        if self.gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU可用: {gpu_count} 个GPU, 内存: {gpu_memory:.1f}GB")
            
            if gpu_memory < 8:
                logger.warning("⚠️  GPU内存可能不足(<8GB)，建议使用更大内存的GPU")
                return False
            return True
        else:
            logger.warning("⚠️  未检测到GPU，将使用CPU运行（速度较慢）")
            return False
    
    def check_model_availability(self) -> bool:
        """检查模型文件可用性"""
        logger.info("检查复杂度分类器模型...")
        
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            logger.error(f"❌ 模型路径不存在: {model_path}")
            logger.info("请确保已训练或下载了ModernBERT复杂度分类器")
            logger.info("模型训练脚本: train_complexity_classifier.py")
            return False
        
        # 检查必需的模型文件
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]
        
        # 检查是否为PEFT模型
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            logger.info("✅ 检测到PEFT/LoRA模型")
            required_files.extend(["adapter_config.json", "adapter_model.bin"])
        else:
            logger.info("✅ 检测到标准模型")
            required_files.extend(["pytorch_model.bin"])
        
        missing_files = []
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"❌ 缺少模型文件: {missing_files}")
            return False
        
        logger.info("✅ 模型文件完整")
        return True
    
    def check_disk_space(self) -> bool:
        """检查磁盘空间"""
        logger.info("检查磁盘空间...")
        
        import shutil
        
        # 检查当前目录可用空间
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        logger.info(f"可用磁盘空间: {free_gb:.1f}GB")
        
        if free_gb < 50:
            logger.warning("⚠️  磁盘空间不足(<50GB)，可能影响数据集下载和处理")
            return False
        
        logger.info("✅ 磁盘空间充足")
        return True
    
    def install_missing_packages(self) -> bool:
        """安装缺失的包"""
        if not self.missing_deps:
            return True
        
        logger.info("安装缺失的Python包...")
        
        for package in self.missing_deps:
            try:
                logger.info(f"安装: {package}")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ 成功安装: {package}")
                else:
                    logger.error(f"❌ 安装失败: {package}")
                    logger.error(result.stderr)
                    return False
                    
            except Exception as e:
                logger.error(f"❌ 安装 {package} 时出错: {e}")
                return False
        
        return True
    
    def check_huggingface_access(self) -> bool:
        """检查Hugging Face访问"""
        logger.info("检查Hugging Face Hub访问...")
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # 尝试访问一个公开数据集
            dataset_info = api.dataset_info("ms_marco", timeout=10)
            logger.info("✅ Hugging Face Hub访问正常")
            return True
            
        except Exception as e:
            logger.error(f"❌ Hugging Face Hub访问失败: {e}")
            logger.info("请检查网络连接或考虑使用镜像源")
            return False
    
    def run_full_check(self) -> bool:
        """运行完整检查"""
        logger.info("=" * 60)
        logger.info("阶段1实验依赖检查")
        logger.info("=" * 60)
        
        checks = [
            ("Python包依赖", self.check_python_packages),
            ("GPU可用性", self.check_gpu_availability), 
            ("模型文件", self.check_model_availability),
            ("磁盘空间", self.check_disk_space),
            ("Hugging Face访问", self.check_huggingface_access)
        ]
        
        results = {}
        for check_name, check_func in checks:
            logger.info(f"\n检查: {check_name}")
            results[check_name] = check_func()
        
        # 汇总结果
        logger.info("\n" + "=" * 60)
        logger.info("检查结果汇总")
        logger.info("=" * 60)
        
        all_passed = True
        for check_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            logger.info(f"{check_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("\n🎉 所有检查通过！可以开始阶段1实验")
        else:
            logger.warning("\n⚠️  部分检查失败，请解决后再开始实验")
            
            # 提供解决建议
            self._provide_fix_suggestions(results)
        
        return all_passed
    
    def _provide_fix_suggestions(self, results: dict):
        """提供修复建议"""
        logger.info("\n🔧 修复建议:")
        
        if not results.get("Python包依赖", True):
            logger.info("1. 安装缺失的Python包:")
            logger.info("   python stage1/setup_dependencies.py --install")
        
        if not results.get("模型文件", True):
            logger.info("2. 训练或下载复杂度分类器:")
            logger.info("   python train_complexity_classifier.py")
            
        if not results.get("GPU可用性", True):
            logger.info("3. GPU问题:")
            logger.info("   - 检查CUDA安装: nvidia-smi")
            logger.info("   - 重装PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
            
        if not results.get("磁盘空间", True):
            logger.info("4. 释放磁盘空间或使用外部存储")
            
        if not results.get("Hugging Face访问", True):
            logger.info("5. 网络问题:")
            logger.info("   - 检查网络连接")
            logger.info("   - 设置代理或使用镜像源")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="阶段1依赖检查")
    parser.add_argument("--install", action="store_true", help="自动安装缺失的包")
    parser.add_argument("--skip-model-check", action="store_true", help="跳过模型文件检查")
    
    args = parser.parse_args()
    
    checker = DependencyChecker()
    
    if args.install:
        # 先检查包依赖
        checker.check_python_packages()
        if checker.missing_deps:
            success = checker.install_missing_packages()
            if not success:
                sys.exit(1)
        else:
            logger.info("所有包已安装")
    
    # 运行完整检查
    if args.skip_model_check:
        logger.info("跳过模型文件检查")
        checker.check_model_availability = lambda: True
    
    success = checker.run_full_check()
    
    if not success:
        sys.exit(1)
    
    logger.info("\n✅ 依赖检查完成，可以开始阶段1实验！")
    logger.info("下一步: python stage1/scripts/data_preparation.py --download --annotate")

if __name__ == "__main__":
    main()