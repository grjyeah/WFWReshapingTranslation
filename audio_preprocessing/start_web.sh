#!/bin/bash
# GPU音频处理Web服务启动脚本

echo "======================================================================"
echo "           GPU音频处理工具 - Web服务启动脚本"
echo "======================================================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python，请先安装Python 3.8+"
    exit 1
fi

echo "[1/3] 检查依赖包..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "[提示] 检测到缺少依赖包，正在安装..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
else
    echo "[OK] 依赖包已安装"
fi

echo ""
echo "[2/3] 检查ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "[警告] 未检测到ffmpeg，请安装后才能处理音频"
    echo "Ubuntu/Debian: sudo apt install ffmpeg"
    echo "macOS: brew install ffmpeg"
    echo ""
fi

echo "[3/3] 启动Web服务器..."
echo ""
echo "======================================================================"
echo "服务将在以下地址启动:"
echo "  主界面: http://localhost:8000"
echo "  API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo "======================================================================"
echo ""

# 启动服务器
python3 api_server.py
