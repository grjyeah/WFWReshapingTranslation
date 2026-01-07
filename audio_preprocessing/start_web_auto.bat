@echo off
REM GPU音频处理Web服务启动脚本 (自动检测版)
REM 自动检测可用的conda环境

setlocal enabledelayedexpansion

echo ======================================================================
echo            GPU音频处理工具 - Web服务启动脚本 (自动检测)
echo ======================================================================
echo.

REM 检查是否在conda环境中
if defined CONDA_DEFAULT_ENV (
    echo [检测] 当前已在Conda环境中: %CONDA_DEFAULT_ENV%
    echo.
    goto check_pytorch
)

REM 检查conda是否可用
where conda >nul 2>&1
if errorlevel 1 (
    echo [提示] 未检测到Conda，使用系统Python
    goto check_pytorch
)

echo [检测] 发现Conda，查找包含PyTorch的环境...
echo.

REM 查找包含PyTorch的conda环境
set FOUND_ENV=
for /f "delims=" %%i in ('conda env list 2^>nul ^| findstr /r "^# \|^*" ^| findstr /v "^#"') do (
    set env_line=%%i
    set env_line=!env_line:*:=!
    set env_name=!env_line: =!

    if not "!env_name!"=="" (
        echo   检测环境: !env_name!

       REM 激活环境并检查PyTorch
        call conda activate !env_name! 2>nul
        if not errorlevel 1 (
            python -c "import torch" 2>nul
            if not errorlevel 1 (
                python -c "import torch; cuda=torch.cuda.is_available(); print(f'      CUDA可用: {cuda}', end='')" 2>nul
                if not errorlevel 1 (
                    echo.
                    set FOUND_ENV=!env_name!
                    goto found
                )
            )
        )
    )
)

:found
if defined FOUND_ENV (
    echo.
    echo [找到] PyTorch环境: !FOUND_ENV!
    echo.
    call conda activate !FOUND_ENV!
) else (
    echo.
    echo [提示] 未找到包含PyTorch的conda环境
    echo.
    echo 可用环境列表:
    conda env list
    echo.
    choice /C YN /M "是否使用base环境继续"
    if errorlevel 2 exit /b 1
    call conda activate base
)

:check_pytorch
echo [检查] Python环境:
python --version
where python
echo.

echo [检查] PyTorch状态:
python -c "import torch; print(f'  PyTorch版本: {torch.__version__}'); print(f'  CUDA可用: {torch.cuda.is_available()}'); print(f'  CUDA版本: {torch.version.cuda}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul

if errorlevel 1 (
    echo [错误] PyTorch未安装或导入失败
    echo.
    echo 请先安装PyTorch:
    echo   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)

echo.
echo [检查] 依赖包:
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo [提示] 缺少Web服务依赖，正在安装...
    pip install fastapi uvicorn python-multipart
)

echo.
echo ======================================================================
echo 启动Web服务器...
echo.
echo 服务地址:
echo   主界面: http://localhost:8000
echo   API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务
echo ======================================================================
echo.

python api_server.py

pause
