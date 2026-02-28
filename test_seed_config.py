"""
测试seed参数配置验证
验证：
1. ChineseFormatter的seed参数是否正确传递
2. EnglishTranslator的seed参数是否正确传递
3. 默认值是否为42
4. 自定义值是否生效
"""

import sys
import os
import io
import importlib.util

# 设置标准输出为UTF-8编码（Windows兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def load_module_from_file(module_name, file_path):
    """从文件路径动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_chinese_formatter():
    """测试ChineseFormatter的seed参数"""
    print("=" * 60)
    print("测试 ChineseFormatter")
    print("=" * 60)

    # 动态加载模块（文件名带连字符）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, "chinese_formatter-ollama.py")
    chinese_module = load_module_from_file("chinese_formatter_ollama", module_path)
    ChineseFormatter = chinese_module.ChineseFormatter

    # 测试1: 默认seed=42
    print("\n[测试1] 默认seed值")
    formatter1 = ChineseFormatter()
    assert hasattr(formatter1, 'seed'), "❌ 缺少seed属性"
    assert formatter1.seed == 42, f"❌ 默认seed应为42，实际为{formatter1.seed}"
    assert 'seed' in formatter1.model_options, "❌ model_options中缺少seed"
    assert formatter1.model_options['seed'] == 42, f"❌ model_options['seed']应为42，实际为{formatter1.model_options['seed']}"
    print("✓ 默认seed = 42")
    print(f"✓ model_options['seed'] = {formatter1.model_options['seed']}")

    # 测试2: 自定义seed=123
    print("\n[测试2] 自定义seed值")
    formatter2 = ChineseFormatter(seed=123)
    assert formatter2.seed == 123, f"❌ 自定义seed应为123，实际为{formatter2.seed}"
    assert formatter2.model_options['seed'] == 123, f"❌ model_options['seed']应为123，实际为{formatter2.model_options['seed']}"
    print("✓ 自定义seed = 123")
    print(f"✓ model_options['seed'] = {formatter2.model_options['seed']}")

    # 测试3: seed=None
    print("\n[测试3] seed=None（完全随机）")
    formatter3 = ChineseFormatter(seed=None)
    assert formatter3.seed is None, f"❌ seed应为None，实际为{formatter3.seed}"
    assert formatter3.model_options['seed'] is None, f"❌ model_options['seed']应为None，实际为{formatter3.model_options['seed']}"
    print("✓ seed = None（完全随机模式）")

    print("\n✅ ChineseFormatter 所有测试通过")
    return True


def test_english_translator():
    """测试EnglishTranslator的seed参数"""
    print("\n" + "=" * 60)
    print("测试 EnglishTranslator")
    print("=" * 60)

    # 动态加载模块（文件名带连字符）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, "english_translator-ollama.py")
    english_module = load_module_from_file("english_translator_ollama", module_path)
    EnglishTranslator = english_module.EnglishTranslator

    # 测试1: 默认seed=42
    print("\n[测试1] 默认seed值")
    translator1 = EnglishTranslator()
    assert hasattr(translator1, 'seed'), "❌ 缺少seed属性"
    assert translator1.seed == 42, f"❌ 默认seed应为42，实际为{translator1.seed}"
    assert 'seed' in translator1.model_options, "❌ model_options中缺少seed"
    assert translator1.model_options['seed'] == 42, f"❌ model_options['seed']应为42，实际为{translator1.model_options['seed']}"
    print("✓ 默认seed = 42")
    print(f"✓ model_options['seed'] = {translator1.model_options['seed']}")

    # 测试2: 自定义seed=456
    print("\n[测试2] 自定义seed值")
    translator2 = EnglishTranslator(seed=456)
    assert translator2.seed == 456, f"❌ 自定义seed应为456，实际为{translator2.seed}"
    assert translator2.model_options['seed'] == 456, f"❌ model_options['seed']应为456，实际为{translator2.model_options['seed']}"
    print("✓ 自定义seed = 456")
    print(f"✓ model_options['seed'] = {translator2.model_options['seed']}")

    # 测试3: seed=None
    print("\n[测试3] seed=None（完全随机）")
    translator3 = EnglishTranslator(seed=None)
    assert translator3.seed is None, f"❌ seed应为None，实际为{translator3.seed}"
    assert translator3.model_options['seed'] is None, f"❌ model_options['seed']应为None，实际为{translator3.model_options['seed']}"
    print("✓ seed = None（完全随机模式）")

    print("\n✅ EnglishTranslator 所有测试通过")
    return True


def test_model_options_integrity():
    """验证model_options其他参数完整性"""
    print("\n" + "=" * 60)
    print("验证 model_options 完整性")
    print("=" * 60)

    # 动态加载模块（文件名带连字符）
    current_dir = os.path.dirname(os.path.abspath(__file__))

    chinese_module_path = os.path.join(current_dir, "chinese_formatter-ollama.py")
    chinese_module = load_module_from_file("chinese_formatter_integrity", chinese_module_path)
    ChineseFormatter = chinese_module.ChineseFormatter

    english_module_path = os.path.join(current_dir, "english_translator-ollama.py")
    english_module = load_module_from_file("english_translator_integrity", english_module_path)
    EnglishTranslator = english_module.EnglishTranslator

    formatter = ChineseFormatter()
    translator = EnglishTranslator()

    # ChineseFormatter 应有的参数
    expected_formatter_keys = [
        'seed', 'num_ctx', 'num_predict', 'num_batch',
        'temperature', 'top_p', 'top_k', 'repeat_penalty',
        'presence_penalty', 'frequency_penalty', 'rope_frequency_base', 'stop'
    ]

    print("\n[ChineseFormatter] model_options:")
    for key in expected_formatter_keys:
        assert key in formatter.model_options, f"❌ 缺少参数: {key}"
        print(f"  ✓ {key}: {formatter.model_options[key]}")

    # EnglishTranslator 应有的参数
    expected_translator_keys = [
        'seed', 'num_ctx', 'num_predict', 'num_batch',
        'temperature', 'top_p', 'top_k', 'repeat_penalty',
        'presence_penalty', 'frequency_penalty', 'rope_frequency_base', 'stop'
    ]

    print("\n[EnglishTranslator] model_options:")
    for key in expected_translator_keys:
        assert key in translator.model_options, f"❌ 缺少参数: {key}"
        print(f"  ✓ {key}: {translator.model_options[key]}")

    print("\n✅ model_options 完整性验证通过")
    return True


if __name__ == "__main__":
    try:
        print("\n" + "🧪 开始seed参数验证测试" + "\n")

        test_chinese_formatter()
        test_english_translator()
        test_model_options_integrity()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("\n✅ seed参数已正确添加到两个类中")
        print("✅ 参数传递机制正常工作")
        print("✅ model_options配置完整")
        print("\n📝 使用示例：")
        print("  formatter = ChineseFormatter(seed=42)  # 固定种子，稳定输出")
        print("  translator = EnglishTranslator(seed=42)  # 固定种子，术语一致")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
