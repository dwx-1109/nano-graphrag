@echo off
echo ===== ModernBERT复杂度分类器训练 (增强版) =====

REM 设置环境变量
set BASE_DIR=%~dp0
cd %BASE_DIR%\..

echo.
echo 开始训练ModernBERT复杂度分类器 (增强版配置)...
echo.

REM 运行训练脚本 - 使用增强版参数，输出目录会自动添加时间戳
python training_data/train_with_modernbert.py --use_lora --use_fp16 --batch_size 4 --num_epochs 15 --learning_rate 1e-5 --lora_r 32 --lora_alpha 64 --early_stopping_patience 3 --eval_steps 50 --save_steps 100

echo.
echo 训练完成!
echo.
pause 