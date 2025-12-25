#!/usr/bin/env python3
"""
快速测试逐句转换功能
验证新提示词是否真正做到逐句转换，而不是总结
"""
import sys
sys.path.insert(0, '/home/aipcuser/pyworkspaces/WFWReshapingTranslation')
from meeting_processor import MeetingTranscriptProcessor

def test_sentence_conversion():
    """测试逐句转换功能"""

    # 简单的测试文本（包含口语化和说话人标识）
    test_text = """[主持人]：大家好，欢迎参加今天的会议。今天我们主要讨论数据治理的相关工作。

[张总]：那个，我们需要建立一个完善的数据治理体系。就是说，数据治理非常重要，对企业的数字化转型至关重要。然后呢，我们要重视这个工作。

[李经理]：对，我同意张总的说法。啊，我们的平台能够提供有效的支持。具体来说，可以提升数据质量，优化管理流程。"""

    print("=" * 80)
    print("🧪 测试逐句转换功能")
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
    print("-" * 80)
    print(result)
    print("-" * 80)

    # 验证结果
    print(f"\n📊 统计信息：")
    print(f"   原文长度：{len(test_text)} 字")
    print(f"   输出长度：{len(result)} 字")
    print(f"   输出比例：{len(result) / len(test_text) * 100:.1f}%")

    # 检查是否符合要求
    print(f"\n🔍 格式检查：")

    # 1. 检查是否保留了说话人标识（支持 [] 和 【】 两种格式）
    has_speakers = bool(("主持人" in result or "主持人]" in result or "主持人】" in result) and
                       ("张总" in result or "张总]" in result or "张总】" in result) and
                       ("李经理" in result or "李经理]" in result or "李经理】" in result))
    print(f"   ✅ 保留说话人标识" if has_speakers else "   ❌ 说话人标识丢失")

    # 2. 检查是否是逐句转换（不是总结）
    has_summary_keywords = any(keyword in result for keyword in [
        "主要讨论了", "会议总结", "重点提到", "归纳如下",
        "会议议题", "会议内容", "主要内容包括"
    ])
    print(f"   ✅ 无总结模式" if not has_summary_keywords else "   ❌ 仍然是总结模式")

    # 3. 检查是否删除了口语词
    has_colloquialism = any(word in result for word in [
        "那个，", "然后呢，", "就是说", "啊，", "嗯", "呃"
    ])
    print(f"   ✅ 已删除口语词" if not has_colloquialism else "   ❌ 仍有口语词")

    # 4. 检查是否有总结性标题
    has_summary_titles = any(keyword in result for keyword in [
        "###", "一、", "二、", "会议主题", "议题一"
    ])
    print(f"   ✅ 无总结性标题" if not has_summary_titles else "   ❌ 包含总结性标题")

    print("\n" + "=" * 80)

    # 保存结果
    with open("test_sentence_conversion_result.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("原文\n")
        f.write("=" * 80 + "\n")
        f.write(test_text)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("逐句转换结果\n")
        f.write("=" * 80 + "\n")
        f.write(result)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"原文: {len(test_text)} 字 | 输出: {len(result)} 字 ({len(result) / len(test_text) * 100:.1f}%)\n")
        f.write("=" * 80 + "\n")

    print("💾 测试结果已保存到: test_sentence_conversion_result.txt")
    print("=" * 80)

    # 判断测试是否通过
    if has_speakers and not has_summary_keywords and not has_colloquialism and not has_summary_titles:
        print("\n✅ 测试通过！逐句转换功能正常工作")
        return True
    else:
        print("\n⚠️ 测试未完全通过，可能需要进一步调整")
        return False

if __name__ == "__main__":
    test_sentence_conversion()
