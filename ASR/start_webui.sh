#!/bin/bash

# FunASR WebUI 启动脚本

echo "========================================="
echo "🎤 FunASR WebUI 启动器"
echo "========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3，请先安装Python 3.8+"
    exit 1
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 工作目录: $SCRIPT_DIR"

# 设置环境变量
echo "🔧 设置环境变量..."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$SCRIPT_DIR/models/huggingface"
export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/models/huggingface"
export MODELSCOPE_CACHE="$SCRIPT_DIR/models/modelscope"

# 创建必要目录
mkdir -p models/huggingface
mkdir -p models/modelscope
mkdir -p cache
mkdir -p temp
mkdir -p outputs

echo "✅ 环境变量设置完成"

# 检查是否需要安装依赖
if [ ! -f "requirements_installed.flag" ]; then
    echo "📦 检测到首次运行，安装依赖包..."

    # 检查是否存在requirements_webui.txt
    if [ -f "requirements_webui.txt" ]; then
        echo "正在安装WebUI依赖..."
        pip install -r requirements_webui.txt

        if [ $? -eq 0 ]; then
            touch "requirements_installed.flag"
            echo "✅ 依赖安装完成"
        else
            echo "❌ 依赖安装失败，请检查网络连接或手动安装"
            echo "手动安装命令: pip install -r requirements_webui.txt"
            exit 1
        fi
    else
        echo "⚠️  未找到requirements_webui.txt，跳过依赖安装"
    fi
else
    echo "✅ 依赖已安装，跳过安装步骤"
fi

# 检查FunASR是否已安装
echo "🔍 检查FunASR环境..."
python3 -c "import funasr; print('FunASR版本:', funasr.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: FunASR未正确安装"
    echo "请先运行: pip install funasr"
    exit 1
fi

# 检查Gradio是否已安装
python3 -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: Gradio未正确安装"
    echo "请先运行: pip install gradio"
    exit 1
fi

echo "✅ 环境检查完成"

# 解析命令行参数
HOST="0.0.0.0"
PORT="6006"
SHARE="false"
DEBUG="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE="true"
            shift
            ;;
        --no-debug)
            DEBUG="false"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --host HOST     绑定地址 (默认: 0.0.0.0)"
            echo "  --port PORT     端口号 (默认: 6006)"
            echo "  --share         启用Gradio公共链接"
            echo "  --no-debug      禁用调试模式"
            echo "  -h, --help      显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查端口占用并清理
echo "🔍 检查端口 $PORT 占用情况..."

check_and_kill_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null)

    if [ ! -z "$pids" ]; then
        echo "⚠️  发现端口 $port 被以下进程占用:"
        echo "$pids" | while read pid; do
            if [ ! -z "$pid" ]; then
                ps -p $pid -o pid,ppid,cmd --no-headers 2>/dev/null | while read line; do
                    echo "   PID: $line"
                done
            fi
        done

        echo "🔥 正在清理占用端口 $port 的进程..."
        echo "$pids" | while read pid; do
            if [ ! -z "$pid" ]; then
                echo "   终止进程 PID: $pid"
                kill -TERM "$pid" 2>/dev/null

                # 等待进程优雅退出
                sleep 2

                # 检查进程是否仍在运行
                if kill -0 "$pid" 2>/dev/null; then
                    echo "   强制终止进程 PID: $pid"
                    kill -KILL "$pid" 2>/dev/null
                fi
            fi
        done

        # 再次检查端口是否已释放
        sleep 1
        local remaining_pids=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$remaining_pids" ]; then
            echo "❌ 端口 $port 仍被占用，尝试强制清理..."
            echo "$remaining_pids" | xargs -r kill -KILL 2>/dev/null
            sleep 1
        fi

        # 最终检查
        local final_check=$(lsof -ti:$port 2>/dev/null)
        if [ -z "$final_check" ]; then
            echo "✅ 端口 $port 已成功释放"
        else
            echo "❌ 无法释放端口 $port，请手动处理或使用其他端口"
            echo "提示: 可以尝试使用 --port 参数指定其他端口"
            exit 1
        fi
    else
        echo "✅ 端口 $PORT 可用"
    fi
}

# 执行端口检查和清理
check_and_kill_port $PORT

echo "🚀 启动参数:"
echo "   地址: $HOST"
echo "   端口: $PORT"
echo "   共享: $SHARE"
echo "   调试: $DEBUG"

# 启动WebUI
echo ""
echo "========================================="
echo "🎤 正在启动FunASR WebUI..."
echo "========================================="
echo ""

# 设置Python参数
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 启动应用
if [ "$DEBUG" = "true" ]; then
    if [ "$SHARE" = "true" ]; then
        python3 app.py --host "$HOST" --port "$PORT" --share --debug
    else
        python3 app.py --host "$HOST" --port "$PORT" --debug
    fi
else
    if [ "$SHARE" = "true" ]; then
        python3 app.py --host "$HOST" --port "$PORT" --share
    else
        python3 app.py --host "$HOST" --port "$PORT"
    fi
fi

echo ""
echo "========================================="
echo "👋 FunASR WebUI 已退出"
echo "========================================="