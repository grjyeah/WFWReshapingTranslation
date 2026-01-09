@echo off
chcp 65001 > nul
echo ========================================
echo 安装MoviePy视频处理库
echo ========================================
echo.

echo 正在激活conda环境: WFWReshapingTranslation
call conda activate WFWReshapingTranslation

if errorlevel 1 (
    echo [错误] 无法激活环境 WFWReshapingTranslation
    echo 请确保该环境已存在
    pause
    exit /b 1
)

echo.
echo 正在安装MoviePy及其依赖...
echo.

pip install moviepy

if errorlevel 1 (
    echo.
    echo [错误] 安装失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 验证安装...
python -c "import moviepy; print(f'MoviePy版本: {moviepy.__version__}')"

if errorlevel 1 (
    echo [警告] 验证失败，但可能已安装
) else (
    echo [成功] MoviePy已成功安装并可以使用
)

echo.
pause
