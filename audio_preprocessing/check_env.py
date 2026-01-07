"""
GPU环境诊断脚本
检查PyTorch和CUDA配置
"""
import sys

print("=" * 70)
print("GPU环境诊断工具")
print("=" * 70)
print()

# 1. Python环境检查
print("[1] Python环境信息:")
print(f"    Python版本: {sys.version}")
print(f"    Python路径: {sys.executable}")
print(f"    当前工作目录: {sys.prefix}")
print()

# 2. PyTorch检查
print("[2] PyTorch状态:")
try:
    import torch
    print(f"    ✓ PyTorch版本: {torch.__version__}")
    print(f"    PyTorch路径: {torch.__file__}")
except ImportError:
    print("    ✗ PyTorch未安装")
    print()
    print("解决方案: pip install torch torchaudio")
    sys.exit(1)

print()

# 3. CUDA检查
print("[3] CUDA状态:")
cuda_available = torch.cuda.is_available()
print(f"    CUDA可用: {cuda_available}")

if cuda_available:
    print(f"    ✓ CUDA版本: {torch.version.cuda}")
    print(f"    ✓ GPU数量: {torch.cuda.device_count()}")
    print(f"    ✓ GPU名称: {torch.cuda.get_device_name(0)}")

    # 显存信息
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1024**3
    print(f"    ✓ 显存大小: {total_memory:.1f} GB")
else:
    print("    ✗ CUDA不可用")
    print()
    print("可能的原因:")
    print("  1. 安装了CPU版本的PyTorch")
    print("  2. CUDA驱动未安装或版本不匹配")
    print("  3. GPU不支持当前CUDA版本")
    print()
    print("解决方案:")
    print("  1. 检查PyTorch版本: python -c \"import torch; print(torch.__version__)\"")
    print("  2. 重新安装GPU版PyTorch:")
    print("     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  3. 验证CUDA: nvidia-smi")

print()

# 4. CUDA运行时检查
if cuda_available:
    print("[4] CUDA运行时测试:")
    try:
        # 创建测试张量
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)

        print("    ✓ GPU计算测试通过")
        print(f"    ✓ 测试张量设备: {z.device}")
    except Exception as e:
        print(f"    ✗ GPU计算测试失败: {e}")

print()

# 5. 关键依赖检查
print("[5] 关键依赖检查:")
dependencies = {
    'numpy': 'NumPy',
    'pydub': 'pydub',
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'torchaudio': 'TorchAudio'
}

for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"    ✓ {name} 已安装")
    except ImportError:
        print(f"    ✗ {name} 未安装")

print()
print("=" * 70)
print("诊断完成")
print("=" * 70)

# 如果CUDA不可用，给出建议
if not cuda_available:
    print()
    print("⚠ CUDA不可用，请检查:")
    print()
    print("1. 确认使用的是正确的conda环境:")
    print("   conda activate your_env_name")
    print()
    print("2. 检查PyTorch是否为GPU版本:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    print()
    print("3. 如果返回False，重新安装GPU版PyTorch:")
    print()
    print("   CUDA 12.1版本:")
    print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("   CUDA 11.8版本:")
    print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("4. 或使用conda安装:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
