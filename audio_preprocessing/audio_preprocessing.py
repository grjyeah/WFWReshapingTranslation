import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import numpy as np

from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from pydub.silence import detect_nonsilent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 配置参数 ====================
class Config:
    """音频处理配置"""

    # 音频分段参数
    MIN_SILENCE_LEN = 1000        # 静音最小长度(ms) - 至少1秒静音才分段
    SILENCE_THRESH = -40          # 静音阈值(dBFS) - 低于此值视为静音
    MIN_CHUNK_LENGTH = 30000      # 最小片段长度(ms) - 30秒,避免碎片化
    MAX_CHUNK_LENGTH = 600000     # 最大片段长度(ms) - 10分钟,控制内存

    # 压缩参数
    THRESHOLD = -20.0             # 压缩阈值
    RATIO = 4.0                   # 压缩比
    ATTACK = 5.0                  # 启动时间
    RELEASE = 50.0                # 释放时间

    # 并行处理参数
    MAX_WORKERS = min(32, cpu_count())  # 最大工作进程数,不超过CPU核心数


def detect_silence_chunks(audio: AudioSegment,
                          min_silence_len: int = Config.MIN_SILENCE_LEN,
                          silence_thresh: int = Config.SILENCE_THRESH,
                          min_chunk_length: int = Config.MIN_CHUNK_LENGTH,
                          max_chunk_length: int = Config.MAX_CHUNK_LENGTH) -> List[Tuple[int, int]]:
    """
    智能检测音频中的静音段,在不切断句子的前提下分段

    Args:
        audio: 音频片段
        min_silence_len: 静音最小长度(ms)
        silence_thresh: 静音阈值(dBFS)
        min_chunk_length: 最小片段长度(ms)
        max_chunk_length: 最大片段长度(ms)

    Returns:
        分段列表 [(start_ms, end_ms), ...]
    """
    logger.info("正在分析音频波形,检测静音段落...")

    # 检测非静音段落 (即有人说话的段落)
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if not nonsilent_ranges:
        # 如果没有检测到非静音段,直接返回整段
        logger.warning("未检测到明显的说话段落,将整体处理")
        return [(0, len(audio))]

    # 智能合并和切分段落
    chunks = []
    current_start = nonsilent_ranges[0][0]
    current_end = nonsilent_ranges[0][1]

    for start, end in nonsilent_ranges[1:]:
        # 计算当前段落到下一个非静音段的间隔
        gap = start - current_end

        # 如果间隔小于最小静音长度,或者当前段落还没达到最小长度,则合并
        if gap < min_silence_len or (current_end - current_start) < min_chunk_length:
            # 合并段落
            current_end = end
        else:
            # 检查是否超过最大长度
            if (current_end - current_start) > max_chunk_length:
                # 强制在当前段落末尾分段
                chunks.append((current_start, current_end))
                current_start = start
                current_end = end
            else:
                # 在静音处分段
                chunks.append((current_start, current_end))
                current_start = start
                current_end = end

    # 添加最后一段
    chunks.append((current_start, current_end))

    logger.info(f"✓ 检测完成,音频已智能切分为 {len(chunks)} 个片段")

    # 输出分段信息
    total_duration = len(audio)
    for i, (start, end) in enumerate(chunks, 1):
        duration = (end - start) / 1000
        logger.info(f"  片段 {i}: {start/1000:.1f}s -> {end/1000:.1f}s (时长: {duration:.1f}s)")

    return chunks


def process_audio_chunk(args: Tuple[AudioSegment, int, int, int]) -> Tuple[int, bytes]:
    """
    处理单个音频片段(多进程工作函数)

    Args:
        args: (audio, start_ms, end_ms, chunk_index)

    Returns:
        (chunk_index, processed_audio_data)
    """
    audio, start_ms, end_ms, chunk_index = args

    try:
        # 提取音频片段
        chunk = audio[start_ms:end_ms]

        # 应用动态范围压缩
        processed = compress_dynamic_range(
            chunk,
            threshold=Config.THRESHOLD,
            ratio=Config.RATIO,
            attack=Config.ATTACK,
            release=Config.RELEASE
        )

        # 导出为原始音频数据
        return (chunk_index, processed.raw_data)

    except Exception as e:
        logger.error(f"片段 {chunk_index} 处理失败: {e}")
        raise


def process_audio(input_path: str,
                  output_path: str = None,
                  threshold: float = Config.THRESHOLD,
                  ratio: float = Config.RATIO,
                  attack: float = Config.ATTACK,
                  release: float = Config.RELEASE):
    """
    处理音频文件 - 智能分段 + 并行处理

    Args:
        input_path: 输入音频文件路径
        output_path: 输出文件路径
        threshold: 压缩阈值
        ratio: 压缩比
        attack: 启动时间
        release: 释放时间
    """
    # 更新配置
    Config.THRESHOLD = threshold
    Config.RATIO = ratio
    Config.ATTACK = attack
    Config.RELEASE = release

    input_file = Path(input_path)

    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_path}")
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 设置输出路径
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_output.wav"
    else:
        output_path = Path(output_path)

    logger.info("=" * 70)
    logger.info("音频处理任务开始 (智能分段 + 多进程并行)")
    logger.info("=" * 70)
    logger.info(f"输入文件: {input_path}")
    logger.info(f"文件大小: {input_file.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"处理参数: threshold={threshold}dB, ratio={ratio}, attack={attack}ms, release={release}ms")
    logger.info(f"硬件配置: {Config.MAX_WORKERS} CPU核心将参与并行处理")

    # 步骤1: 加载音频文件
    logger.info("\n[1/5] 正在加载音频文件...")
    try:
        audio = AudioSegment.from_file(input_path)
        duration_seconds = len(audio) / 1000.0
        logger.info(f"✓ 音频加载成功! 时长: {duration_seconds:.2f} 秒 ({duration_seconds/60:.2f} 分钟)")
    except Exception as e:
        logger.error(f"✗ 音频加载失败: {e}")
        raise

    # 步骤2: 智能分段
    logger.info("\n[2/5] 正在进行智能分段分析...")
    try:
        chunks = detect_silence_chunks(audio)
        logger.info(f"✓ 分段完成,共 {len(chunks)} 个片段")
    except Exception as e:
        logger.error(f"✗ 分段失败: {e}")
        raise

    # 步骤3: 准备并行处理任务
    logger.info(f"\n[3/5] 准备并行处理任务 (使用 {Config.MAX_WORKERS} 个CPU核心)...")

    # 创建临时目录存储中间结果
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_processing_"))
    logger.info(f"临时工作目录: {temp_dir}")

    try:
        # 步骤4: 并行处理音频片段
        logger.info(f"\n[4/5] 正在并行处理 {len(chunks)} 个音频片段...")

        # 准备任务参数
        tasks = [(audio, start, end, i) for i, (start, end) in enumerate(chunks)]

        # 使用多进程池并行处理
        with Pool(processes=Config.MAX_WORKERS) as pool:
            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(process_audio_chunk, tasks),
                total=len(tasks),
                desc="处理进度",
                unit="片段",
                ncols=80
            ))

        logger.info("✓ 所有片段处理完成!")

        # 步骤5: 合并处理后的音频
        logger.info("\n[5/5] 正在合并处理后的音频片段...")
        logger.info("处理中,请稍候...")

        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])

        # 创建第一个音频片段作为基础
        first_index, first_data = results[0]
        # 重建第一个片段的AudioSegment对象
        first_chunk_start, first_chunk_end = chunks[first_index]
        first_chunk_original = audio[first_chunk_start:first_chunk_end]
        merged_audio = first_chunk_original._spawn(data=first_data)

        # 逐个合并后续片段
        with tqdm(total=len(results)-1, desc="合并进度", unit="片段", ncols=80) as pbar:
            for index, data in results[1:]:
                chunk_start, chunk_end = chunks[index]
                chunk_original = audio[chunk_start:chunk_end]
                processed_chunk = chunk_original._spawn(data=data)
                merged_audio += processed_chunk
                pbar.update(1)

        logger.info("✓ 合并完成!")

        # 导出最终音频文件
        logger.info("\n正在导出最终音频文件...")
        with tqdm(total=100, desc="导出进度", unit="%", ncols=80) as pbar:
            merged_audio.export(str(output_path), format="wav")
            pbar.update(100)

        output_size = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ 导出成功! 输出文件大小: {output_size:.2f} MB")
        logger.info(f"✓ 文件已保存至: {output_path}")

    except Exception as e:
        logger.error(f"✗ 处理失败: {e}")
        raise
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.debug(f"✓ 临时文件已清理: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("音频处理任务完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    # 配置输入输出路径
    INPUT_FILE = r"D:\FunASR_installer\真实会议测试音频\2025-11-10 数据治理产品分享恩核供应商.m4a"
    OUTPUT_FILE = r"D:\FunASR_installer\真实会议测试音频\2025-11-10 数据治理产品分享恩核供应商_output.wav"

    # 执行音频处理
    process_audio(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0
    )
