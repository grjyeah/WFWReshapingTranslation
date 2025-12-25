#!/usr/bin/env python3
"""
测试去重功能
"""
from meeting_processor import MeetingTranscriptProcessor


def test_dedup():
    """测试去重功能"""
    # 测试文本（包含重复内容）
    test_text = """[主持人]：
各位领导、同事，大家好。欢迎大家参加今天的数据治理工具交流会议。
今天我们将重点讨论数据治理相关的工作及日常遇到的问题。
数据治理非常重要。数据治理对企业的数字化转型至关重要。
数据治理能够为企业提供有效的支持。我们必须重视数据治理工作。
"""

    print("=" * 70)
    print("🧪 测试去重功能")
    print("=" * 70)

    processor = MeetingTranscriptProcessor()

    print(f"\n📝 原始文本:")
    print(f"   长度: {len(test_text)} 字符")
    print(f"   内容:\n{test_text}")

    # 测试去重
    print(f"\n{'=' * 70}")
    print("🔄 执行去重...")
    print(f"{'=' * 70}")

    dedup_text = processor.remove_duplicates(test_text)

    print(f"\n📊 去重结果:")
    print(f"   原长度: {len(test_text)} 字符")
    print(f"   新长度: {len(dedup_text)} 字符")
    print(f"   删除: {len(test_text) - len(dedup_text)} 字符")
    print(f"   精简率: {(1 - len(dedup_text)/len(test_text)) * 100:.1f}%")

    print(f"\n📄 去重后内容:")
    print(dedup_text)

    # 测试重复检测
    print(f"\n{'=' * 70}")
    print("🔍 测试重复检测...")
    print(f"{'=' * 70}")

    # 正常文本
    normal_text = "这是一段正常的文本，没有重复的内容。"
    has_repetition = processor.detect_repetition(normal_text)
    print(f"\n正常文本: '{normal_text}'")
    print(f"检测结果: {'✅ 无重复' if not has_repetition else '❌ 检测到重复'}")

    # 重复文本
    repeat_text = "这是重复的文本。这是重复的文本。"
    has_repetition = processor.detect_repetition(repeat_text)
    print(f"\n重复文本: '{repeat_text}'")
    print(f"检测结果: {'❌ 检测到重复' if has_repetition else '✅ 无重复'}")

    # 长重复文本
    long_repeat = (
        "数据治理非常重要。我们需要建立完善的体系。" * 3 +
        "数据治理非常重要。我们需要建立完善的体系。"
    )
    has_repetition = processor.detect_repetition(long_repeat)
    print(f"\n长重复文本: '{long_repeat[:50]}...'")
    print(f"检测结果: {'❌ 检测到重复' if has_repetition else '✅ 无重复'}")

    print(f"\n{'=' * 70}")
    print("✅ 测试完成！")
    print(f"{'=' * 70}")

    # 保存结果
    with open("test_dedup_result.txt", "w", encoding="utf-8") as f:
        f.write(f"原文:\n{test_text}\n\n")
        f.write(f"去重后:\n{dedup_text}\n")
    print(f"\n💾 结果已保存到: test_dedup_result.txt")


def test_sentences():
    """测试句子级别去重"""
    processor = MeetingTranscriptProcessor()

    sentences = [
        "我们需要建立一个完整的数据治理体系。",
        "我们需要建立一个完善的数据治理体系。",
        "数据质量非常重要。",
        "数据质量至关重要。",
        "企业需要重视数据治理工作。",
        "公司必须重视数据治理工作。",
        "平台能够提供有效的支持。",
        "平台可以提供有力的支持。",
        "数据治理能够帮助企业提升效率。",
    ]

    print("\n" + "=" * 70)
    print("🧪 测试句子级别去重")
    print("=" * 70)

    print(f"\n原始句子（{len(sentences)}个）:")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")

    # 合并为文本（用换行分隔，方便查看）
    text = '\n'.join(sentences)

    # 去重
    dedup_text = processor.remove_duplicates(text)
    dedup_sentences = [s.strip() for s in dedup_text.split('\n') if s.strip()]

    print(f"\n去重后（{len(dedup_sentences)}个）:")
    for i, s in enumerate(dedup_sentences, 1):
        print(f"  {i}. {s}")

    print(f"\n📊 统计:")
    print(f"   原始: {len(sentences)} 个句子")
    print(f"   去重: {len(dedup_sentences)} 个句子")
    print(f"   删除: {len(sentences) - len(dedup_sentences)} 个重复句子")
    print(f"   精简率: {(1 - len(dedup_sentences)/len(sentences)) * 100:.1f}%")


if __name__ == "__main__":
    test_dedup()
    test_sentences()
