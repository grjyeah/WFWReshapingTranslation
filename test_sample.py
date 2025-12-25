#!/usr/bin/env python3
"""
快速测试脚本 - 验证精简书面化效果
处理前几个segments，快速查看效果
"""
from meeting_processor import MeetingTranscriptProcessor


def test_sample():
    """测试样本"""
    sample_text = """[主持人]：
呃，那个，大家好啊，我看时间已经到了下午四点了，咱们今天这个会议就开始了哈。
非常感谢各位领导和同事们能够参加今天下午的这个数据治理工具交流会议。
那个，今天我们主要就是想跟大家交流一下关于数据治理这一块的工作，
还有就是我们在日常工作中遇到的一些问题，然后呢，也想分享一些具体的案例和经验。

[张总]：
嗯，好的，谢谢主持人的介绍。我来说两句啊。
就是说我这边呢，主要是负责技术方面的工作的，
那么今天我想主要介绍一下我们公司在数据治理方面的一些实践经验。
大概分为三个部分吧，第一个部分就是我们公司的基本情况介绍，
第二个部分就是我们做过的一些成功的案例分享，
第三个部分就是我们的数据治理平台的一个演示吧。
嗯，大概就是这样子。

[李主管]：
好的，谢谢张总。那我这边的话呢，想补充一下，
就是我们在数据质量管控这一块的一些做法。
其实吧，我们在实际工作中发现了很多问题啊，
就是数据质量这块真的很重要，我们必须要重视起来。
所以说呢，我们建立了一套完整的质量管控体系，
这个体系呢，包括了事前预防、事中监控和事后改进三个环节。
"""

    print("=" * 70)
    print("🧪 快速测试 - 精简书面化效果")
    print("=" * 70)

    # 初始化处理器
    processor = MeetingTranscriptProcessor(
        ollama_url="http://localhost:11434",
        model_name="yasserrmd/Qwen2.5-7B-Instruct-1M:latest"
    )

    print(f"\n📝 原始文本:")
    print(f"   总长度: {len(sample_text)} 字符")
    print(f"   段落数: 3个")

    # 分割
    chunks = processor.split_text(sample_text, max_chars=1000)
    print(f"\n📊 分割结果: {len(chunks)} 个segments")

    # 处理每个segment
    processed_chunks = []
    for i, chunk in enumerate(chunks, 1):
        chunk_length = len(chunk)
        target_length = int(chunk_length * 0.8)

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(chunks)}] Segment 处理")
        print(f"{'=' * 70}")
        print(f"📥 输入 ({chunk_length} 字符):")
        print(f"   {chunk[:100]}...")

        # 构建提示词
        prompt = processor.processing_prompt.format(
            text=chunk,
            text_length=chunk_length,
            target_length=target_length
        )

        # 调用模型
        result = processor.call_ollama(prompt)

        if result:
            result_ratio = len(result) / chunk_length * 100

            print(f"\n📤 输出 ({len(result)} 字符, {result_ratio:.1f}%):")
            print(f"   {result[:100]}...")

            # 评价
            if 75 <= result_ratio <= 85:
                print(f"   ✅ 理想比例")
            elif result_ratio > 100:
                print(f"   ⚠️  输出偏多，需要截断")
            elif result_ratio < 60:
                print(f"   ⚠️  输出偏少，可能信息丢失")
            else:
                print(f"   ✓ 可接受")

            processed_chunks.append(result)
        else:
            print(f"   ❌ 处理失败")
            processed_chunks.append(chunk)

    # 合并结果
    print(f"\n{'=' * 70}")
    print("📋 最终结果对比")
    print(f"{'=' * 70}")

    processed_text = "\n\n".join(processed_chunks)

    print(f"\n原文总长: {len(sample_text)} 字符")
    print(f"输出总长: {len(processed_text)} 字符")
    print(f"精简比例: {len(processed_text) / len(sample_text) * 100:.1f}%")

    print(f"\n{'=' * 70}")
    print("📄 完整输出:")
    print(f"{'=' * 70}")
    print(processed_text)

    # 保存结果
    with open("test_output.txt", "w", encoding="utf-8") as f:
        f.write(processed_text)
    print(f"\n💾 测试结果已保存到: test_output.txt")

    print(f"\n{'=' * 70}")
    print("✅ 测试完成！")
    print(f"{'=' * 70}")
    print("\n💡 建议:")
    print("   1. 检查输出是否够正式（去除口语化）")
    print("   2. 检查篇幅是否精简（目标80%）")
    print("   3. 检查信息是否完整")
    print("   4. 如果效果不理想，调整参数后重新测试")


if __name__ == "__main__":
    try:
        test_sample()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
