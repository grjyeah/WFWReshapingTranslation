@echo off
REM GPU音频处理Web服务启动脚本
echo ======================================================================
echo            GPU音频处理工具 - Web服务启动脚本
echo ======================================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 检查依赖包...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [提示] 检测到缺少依赖包，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
) else (
    echo [OK] 依赖包已安装
)

echo.
echo [2/3] 检查ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到ffmpeg，请安装后才能处理音频
    echo Windows安装方法: choco install ffmpeg
    echo 或访问: https://ffmpeg.org/download.html
    echo.
)

echo [3/3] 启动Web服务器...
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
