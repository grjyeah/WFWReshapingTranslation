#!/usr/bin/env python3
"""
从检查点恢复处理任务
当程序在某个segment卡住时，可以跳过已处理的segments，从指定位置继续
"""
import sys
from meeting_processor import MeetingTranscriptProcessor


def resume_processing(
    input_file: str = "meeting_transcript.txt",
    start_segment: int = 31,
    output_prefix: str = "resumed"
):
    """
    从指定segment开始恢复处理

    Args:
        input_file: 输入文件路径
        start_segment: 从第几个segment开始（从1开始）
        output_prefix: 输出文件前缀
    """
    print(f"🚀 启动恢复处理模式")
    print(f"📂 输入文件: {input_file}")
    print(f"📍 起始segment: {start_segment}")
    print(f"💾 输出前缀: {output_prefix}")
    print(f"=" * 60)

    # 读取输入文件
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            transcript = f.read()
        print(f"✓ 成功读取文件: {len(transcript)} 字符")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {input_file}")
        return

    # 初始化处理器
    processor = MeetingTranscriptProcessor(
        ollama_url="http://localhost:11434",
        model_name="yasserrmd/Qwen2.5-7B-Instruct-1M:latest"
    )

    # 分割文本
    chunks = processor.split_text(transcript, max_chars=800)
    total_segments = len(chunks)

    print(f"📊 文本分成了 {total_segments} 个segments")
    print(f"⏭️  跳过前 {start_segment - 1} 个segments")
    print(f"🎯 将处理 {start_segment} 到 {total_segments} (共{total_segments - start_segment + 1}个)")
    print(f"=" * 60)

    # 处理指定的segments
    processed_chunks = []
    total_output_length = 0

    for i in range(start_segment - 1, total_segments):
        chunk = chunks[i]
        chunk_length = len(chunk)
        target_length = int(chunk_length * 0.9)

        print(f"\n[{i + 1}/{total_segments}] 处理中... (输入: {chunk_length} 字符, 目标: {target_length} 字符)", end=" ")

        # 构建提示词
        prompt = processor.processing_prompt.format(
            text=chunk,
            text_length=chunk_length,
            target_length=target_length
        )

        # 调用模型（使用流式输出，防止卡死）
        result = processor.call_ollama(prompt, use_stream=True)

        if result:
            result_ratio = len(result) / chunk_length * 100

            # 处理异常输出
            if result_ratio > 300:
                print(f"\n  ⚠️ 输出异常 ({len(result)} 字符, {result_ratio:.1f}%)，截断...")
                truncated = result[:chunk_length * 2]
                last_period = truncated.rfind('。')
                if last_period > chunk_length:
                    result = truncated[:last_period + 1]
                else:
                    result = truncated
                result_ratio = len(result) / chunk_length * 100

            elif result_ratio < 60:
                print(f"\n  ⚠️ 输出偏少，重新生成...")
                result = processor.call_ollama(prompt)
                result_ratio = len(result) / chunk_length * 100 if result else 0

            if result:
                processed_chunks.append(result)
                total_output_length += len(result)
                print(f"✓ 输出: {len(result)} 字符 ({result_ratio:.1f}%)")
            else:
                print(f"✗ 处理失败，使用原文")
                processed_chunks.append(chunk)
                total_output_length += len(chunk)
        else:
            print(f"✗ 处理失败")
            processed_chunks.append(chunk)
            total_output_length += len(chunk)

    # 合并结果
    processed_text = "\n\n".join(processed_chunks)

    # 保存结果
    output_file = f"{output_prefix}_processed_chinese.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(processed_text)

    # 统计信息
    overall_ratio = total_output_length / len(transcript) * 100
    print(f"\n{'=' * 60}")
    print(f"✅ 处理完成！")
    print(f"📊 统计信息:")
    print(f"  原文总长: {len(transcript)} 字符")
    print(f"  输出总长: {total_output_length} 字符 ({overall_ratio:.1f}%)")
    print(f"  处理segments: {start_segment} - {total_segments}")
    print(f"💾 结果已保存到: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # 从命令行获取参数
    input_file = sys.argv[1] if len(sys.argv) > 1 else "meeting_transcript.txt"
    start_segment = int(sys.argv[2]) if len(sys.argv) > 2 else 31

    resume_processing(input_file, start_segment)
