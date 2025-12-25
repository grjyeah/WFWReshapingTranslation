#!/usr/bin/env python3
"""
测试说话人标识和语义段落换行功能
"""
import sys
sys.path.insert(0, '/home/aipcuser/pyworkspaces/WFWReshapingTranslation')
from meeting_processor import MeetingTranscriptProcessor

def test_speaker_paragraph():
    """测试说话人标识和段落换行"""

    # 测试文本：包含多个说话人，同一说话人多句连续发言
    test_text = """[主持人]：大家好，欢迎参加今天的会议。
今天我们主要讨论数据治理的相关工作。
请大家踊跃发言。

[张总]：那个，我们需要建立一个完善的数据治理体系。
就是说，数据治理非常重要，对企业的数字化转型至关重要。
然后呢，我们要重视这个工作。

[李经理]：对，我同意张总的说法。
我们的平台能够提供有效的支持。
具体来说，可以提升数据质量，优化管理流程。

[主持人]：很好，感谢张总和李经理的发言。
现在请大家自由讨论。"""

    print("=" * 80)
    print("🧪 测试说话人标识和语义段落换行")
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

    # 检查说话人标识
    print(f"\n🔍 说话人标识检查：")

    # 统计说话人数量
    speaker_pattern = "【"
    speaker_count = result.count(speaker_pattern)

    expected_speakers = ["主持人", "张总", "李经理"]
    found_speakers = []

    for speaker in expected_speakers:
        if speaker in result:
            found_speakers.append(speaker)

    print(f"   发现说话人标识：{speaker_count} 个")
    print(f"   找到的说话人：{', '.join(found_speakers)}")

    if len(found_speakers) == len(expected_speakers):
        print(f"   ✅ 所有说话人标识完整")
    else:
        print(f"   ❌ 说话人标识不完整")
        missing = set(expected_speakers) - set(found_speakers)
        print(f"      缺失：{', '.join(missing)}")

    # 检查段落格式
    print(f"\n🔍 段落格式检查：")

    lines = result.strip().split('\n')
    speaker_lines = 0
    content_lines = 0

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line.startswith('【'):
            speaker_lines += 1
            print(f"   第{i}行：说话人标识 ✓")
        elif line:
            content_lines += 1

    print(f"   说话人标识行数：{speaker_lines}")
    print(f"   内容行数：{content_lines}")
    print(f"   总行数：{len(lines)}")

    # 检查是否每句话都换行（不应该）
    content_only_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('【')]
    short_lines = [l for l in content_only_lines if len(l) < 30]

    if len(short_lines) > len(content_only_lines) * 0.5:
        print(f"   ⚠️ 警告：可能有过多短行（每句话换行）")
    else:
        print(f"   ✅ 段落组织良好")

    # 检查是否包含无关内容
    print(f"\n🔍 无关内容检查：")

    forbidden_patterns = [
        "### 会议",
        "【逐句",
        "【书面化",
        "以下是会议",
        "逐句书面化",
        "书面化改写",
        "改写如下",
    ]

    has_forbidden = False
    for pattern in forbidden_patterns:
        if pattern in result:
            has_forbidden = True
            print(f"   ❌ 发现无关内容：{pattern}")

    if not has_forbidden:
        print(f"   ✅ 无无关内容")

    print("\n" + "=" * 80)

    # 保存结果
    with open("test_speaker_paragraph_result.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("原文\n")
        f.write("=" * 80 + "\n")
        f.write(test_text)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("处理结果\n")
        f.write("=" * 80 + "\n")
        f.write(result)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"原文: {len(test_text)} 字 | 输出: {len(result)} 字 ({len(result) / len(test_text) * 100:.1f}%)\n")
        f.write("=" * 80 + "\n")

    print("💾 测试结果已保存到: test_speaker_paragraph_result.txt")
    print("=" * 80)

    # 判断测试是否通过
    if len(found_speakers) == len(expected_speakers) and not has_forbidden:
        print("\n✅ 测试通过！说话人标识完整，无无关内容")
        return True
    else:
        print("\n⚠️ 测试未完全通过，可能需要进一步调整")
        return False

if __name__ == "__main__":
    test_speaker_paragraph()
