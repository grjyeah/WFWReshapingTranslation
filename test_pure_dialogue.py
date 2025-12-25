#!/usr/bin/env python3
"""
测试纯对话输出功能
验证是否只输出对话内容，不添加任何标题或说明性文字
"""
import sys
sys.path.insert(0, '/home/aipcuser/pyworkspaces/WFWReshapingTranslation')
from meeting_processor import MeetingTranscriptProcessor

def test_pure_dialogue():
    """测试纯对话输出"""

    # 测试文本
    test_text = """[主持人]：大家好，欢迎参加今天的会议。今天我们主要讨论数据治理的相关工作。

[张总]：那个，我们需要建立一个完善的数据治理体系。就是说，数据治理非常重要，对企业的数字化转型至关重要。然后呢，我们要重视这个工作。

[李经理]：对，我同意张总的说法。啊，我们的平台能够提供有效的支持。具体来说，可以提升数据质量，优化管理流程。"""

    print("=" * 80)
    print("🧪 测试纯对话输出功能")
    print("=" * 80)

    print(f"\n📝 原文（{len(test_text)}字）：")
    print("-" * 80)
    print(test_text)
    print("-" * 80)

    # 创建处理器并测试
    processor = MeetingTranscriptProcessor()

    print("\n🔄 正在处理...")
    print("-" * 80)

    result = processor.process_transcript(test_text)

    print("\n✅ 处理结果：")
    print("=" * 80)
    print(result)
    print("=" * 80)

    # 验证结果
    print(f"\n📊 统计信息：")
    print(f"   原文长度：{len(test_text)} 字")
    print(f"   输出长度：{len(result)} 字")
    print(f"   输出比例：{len(result) / len(test_text) * 100:.1f}%")

    # 检查是否包含无关内容
    print(f"\n🔍 纯对话检查：")

    forbidden_patterns = [
        "### 会议",
        "【逐句",
        "【书面化",
        "以下是会议",
        "逐句书面化",
        "书面化改写",
        "改写如下",
        "改写版本",
        "会议记录",
        "正式会议",
        "以下是",
        "如下：",
        "：\n\n#",
        "：\n\n**",
    ]

    has_forbidden = False
    found_patterns = []

    for pattern in forbidden_patterns:
        if pattern in result:
            has_forbidden = True
            found_patterns.append(pattern)

    if not has_forbidden:
        print(f"   ✅ 无无关内容（无标题、说明性文字）")
    else:
        print(f"   ❌ 发现无关内容：")
        for pattern in found_patterns:
            print(f"      - {pattern}")

    # 检查是否以对话开头
    lines = result.strip().split('\n')
    first_line = lines[0].strip() if lines else ""

    starts_with_dialogue = bool(
        first_line.startswith('[') or
        first_line.startswith('【') or
        '主持人' in first_line or
        '张总' in first_line or
        '李经理' in first_line
    )

    if starts_with_dialogue:
        print(f"   ✅ 直接以对话开头")
    else:
        print(f"   ❌ 不是以对话开头（首行：{first_line[:50]}...）")

    print("\n" + "=" * 80)

    # 保存结果
    with open("test_pure_dialogue_result.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("原文\n")
        f.write("=" * 80 + "\n")
        f.write(test_text)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("纯对话输出结果\n")
        f.write("=" * 80 + "\n")
        f.write(result)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"原文: {len(test_text)} 字 | 输出: {len(result)} 字 ({len(result) / len(test_text) * 100:.1f}%)\n")
        f.write("=" * 80 + "\n")

    print("💾 测试结果已保存到: test_pure_dialogue_result.txt")
    print("=" * 80)

    # 判断测试是否通过
    if not has_forbidden and starts_with_dialogue:
        print("\n✅ 测试通过！输出为纯对话内容")
        return True
    else:
        print("\n⚠️ 测试未完全通过，可能需要进一步调整")
        return False

if __name__ == "__main__":
    test_pure_dialogue()
