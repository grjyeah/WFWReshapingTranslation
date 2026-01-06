@echo off
chcp 65001 >nul
cd /d "D:\llama.cpp"

echo.
echo 🚀 正在启动 Qwen3-30B-A3B MoE 模型 (Q4_K_XL) ...
echo    模型路径: D:\GGUF\unsloth\Qwen3-30B-A3B-128K-GGUF\Qwen3-30B-A3B-128K-UD-Q4_K_XL.gguf
echo    端口: 6008 | 上下文: 8192 | GPU Layers: 49
echo    Prompt Cache: 16384 MiB
echo.

.\llama-server.exe ^
  --model "D:\GGUF\unsloth\Qwen3-30B-A3B-128K-GGUF\Qwen3-30B-A3B-128K-UD-Q4_K_XL.gguf" ^
  --ctx-size 8192 ^
  --n-gpu-layers 49 ^
  --cache-ram 16384 ^
  --threads 12 ^
  --port 6008 ^
  --host 0.0.0.0

if %errorlevel% neq 0 (
    echo.
    echo ❌ 模型启动失败，请检查路径或显存。
    pause
)