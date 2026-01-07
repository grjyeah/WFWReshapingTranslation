@echo off
REM 切换到UTF-8编码显示中文
chcp 65001 >nul 2>&1

REM GPU音频处理Web服务启动脚本 (Conda环境版)
REM 使用方法: 修改 CONDA_ENV_NAME 为您的conda环境名

setlocal enabledelayedexpansion

REM ===== 配置区域 =====
REM 请修改为您的conda环境名
set CONDA_ENV_NAME=WFWReshapingTranslation

REM ====================

echo ======================================================================
echo            GPU音频处理工具 - Web服务启动脚本 (Conda版)
echo ======================================================================
echo.
echo 使用Conda环境: %CONDA_ENV_NAME%
echo.

REM 检查conda是否可用
where conda >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Conda，请确保已安装Anaconda或Miniconda
    echo 提示: 需要将conda添加到系统PATH
    pause
    exit /b 1
)

echo [1/4] 激活Conda环境...
call conda activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo [错误] 无法激活环境: %CONDA_ENV_NAME%
    echo.
    echo 可用环境列表:
    conda env list
    pause
    exit /b 1
)

echo [OK] 环境已激活: %CONDA_ENV_NAME%
echo.

echo [2/4] 检查Python环境...
python --version
echo Python路径:
where python
echo.

echo [3/4] 检查PyTorch和CUDA...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"
if errorlevel 1 (
    echo [警告] PyTorch未安装或有问题
    echo.
    echo 安装GPU版PyTorch:
    echo   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
)

echo.
echo [4/4] 启动Web服务器...
echo.
echo ======================================================================
echo 服务将在以下地址启动:
echo   主界面: http://localhost:8000
echo   API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务
echo ======================================================================
echo.

REM 启动服务器
python api_server.py

pause
