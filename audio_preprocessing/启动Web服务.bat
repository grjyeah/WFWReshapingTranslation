@echo off
REM 切换到UTF-8编码显示中文
chcp 65001 >nul 2>&1

REM GPU音频处理Web服务 - WFWReshapingTranslation环境专用
REM 端口: 8001

echo ======================================================================
echo            GPU音频处理工具 - Web服务启动
echo ======================================================================
echo.
echo Conda环境: WFWReshapingTranslation
echo 端口: 8001
echo.

REM 激活conda环境
call conda activate WFWReshapingTranslation
if errorlevel 1 (
    echo [错误] 无法激活环境 WFWReshapingTranslation
    echo.
    echo 请确认:
    echo   1. 已安装conda
    echo   2. 环境名称正确
    echo.
    echo 查看所有环境: conda env list
    pause
    exit /b 1
)

echo [OK] 环境已激活
echo.

REM 进入目录
cd /d "%~dp0"
echo 当前目录: %CD%
echo.

REM 检查Python
python --version
echo.

REM 检查PyTorch GPU
echo 检查PyTorch CUDA支持...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
echo.

echo ======================================================================
echo 启动Web服务...
echo.
echo 访问地址:
echo   主界面: http://localhost:8001
echo   API文档: http://localhost:8001/docs
echo.
echo 按 Ctrl+C 停止服务
echo ======================================================================
echo.

python api_server.py

pause
