"""
GPU加速音频处理 - RTX 5090优化版
使用PyTorch实现GPU并行处理，兼容最新版本
"""
import logging
import time
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GPUAudioProcessor:
    """GPU加速音频处理器"""

    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        if self.device == 'cuda':
            logger.info(f"✓ GPU加速已启用: {torch.cuda.get_device_name(0)}")
            logger.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("⚠ GPU不可用，使用CPU模式")

    def dynamic_range_compression_gpu(self,
                                     waveform: torch.Tensor,
                                     threshold_db: float = -20.0,
                                     ratio: float = 4.0,
                                     attack_ms: float = 5.0,
                                     release_ms: float = 50.0,
                                     sample_rate: int = 44100) -> torch.Tensor:
        """
        GPU加速的动态范围压缩

        Args:
            waveform: 音频波形 [channels, samples]
            threshold_db: 压缩阈值
            ratio: 压缩比
            attack_ms: 启动时间
            release_ms: 释放时间
            sample_rate: 采样率
        """
        # 转换到GPU
        waveform = waveform.to(self.device)

        # 转换阈值
        threshold_linear = 10 ** (threshold_db / 20.0)

        # 计算增益曲线（基于振幅）
        abs_wave = torch.abs(waveform)

        # 平滑处理（模拟attack/release）
        kernel_size = max(3, int(sample_rate * attack_ms / 1000))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 使用平均池化平滑
        smoothed = torch.nn.functional.avg_pool1d(
            abs_wave.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        ).squeeze(0)

        # 计算压缩增益
        gain = torch.ones_like(smoothed)

        # 超过阈值的部分进行压缩
        mask = smoothed > threshold_linear
        if mask.any():
            over_threshold = smoothed[mask] / threshold_linear
            compressed_gain = over_threshold ** (1.0 / ratio - 1.0)
            gain[mask] = compressed_gain

        # 应用增益
        compressed = waveform * gain

        # 归一化防止削波
        max_val = torch.max(torch.abs(compressed))
        if max_val > 0.95:
            compressed = compressed * (0.95 / max_val)

        return compressed

    def process_audio_file(self,
                          input_path: str,
                          output_path: str,
                          threshold: float = -20.0,
                          ratio: float = 4.0,
                          attack: float = 5.0,
                          release: float = 50.0,
                          chunk_duration: float = 60.0,
                          output_format: str = "mp3"):
        """
        处理完整音频文件（分块处理避免显存不足）

        Args:
            output_format: 输出格式，支持 'mp3', 'wav', 'm4a', 'flac'
        """
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")

        total_start = time.time()

        logger.info("=" * 70)
        logger.info("GPU音频处理任务 - RTX 5090加速")
        logger.info("=" * 70)
        logger.info(f"输入: {input_path}")
        logger.info(f"文件大小: {input_file.stat().st_size / 1024 / 1024:.1f} MB")
        logger.info(f"输出: {output_path}")
        logger.info(f"参数: threshold={threshold}dB, ratio={ratio}, attack={attack}ms, release={release}ms")

        # ===== 步骤1: 使用pydub加载音频 =====
        logger.info("\n[1/3] 加载音频...")
        load_start = time.time()

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)

            sample_rate = audio.frame_rate
            num_channels = audio.channels
            duration_sec = len(audio) / 1000.0

            logger.info(f"✓ 采样率: {sample_rate} Hz")
            logger.info(f"  时长: {duration_sec:.1f}秒 ({duration_sec/60:.1f}分钟)")
            logger.info(f"  声道: {num_channels}")

            # 转换为numpy数组
            samples = np.array(audio.get_array_of_samples())
            if num_channels == 2:
                samples = samples.reshape((-1, 2))
                # 转换为 [channels, samples]
                samples = samples.T
            else:
                samples = samples.reshape((1, -1))

            # 归一化到 [-1, 1]
            if audio.sample_width == 1:  # 8-bit
                samples = samples.astype(np.float32) / 128.0 - 1.0
            elif audio.sample_width == 2:  # 16-bit
                samples = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples.astype(np.float32) / 2147483648.0

            # 转换为torch tensor
            waveform = torch.from_numpy(samples).float()

            logger.info(f"  波形形状: {waveform.shape}")

        except ImportError:
            logger.error("需要安装pydub: pip install pydub")
            raise

        # ===== 步骤2: 分块GPU处理 =====
        logger.info(f"\n[2/3] GPU并行处理 (分块大小: {chunk_duration}秒)...")

        chunk_samples = int(chunk_duration * sample_rate)
        num_samples = waveform.shape[1]
        num_chunks = (num_samples + chunk_samples - 1) // chunk_samples

        logger.info(f"  分块数量: {num_chunks}")

        process_start = time.time()

        # 存储处理后的音频块
        processed_chunks = []

        # 逐块处理
        with tqdm(total=num_chunks, desc="GPU处理", unit="块", ncols=100) as pbar:
            for i in range(num_chunks):
                start_idx = i * chunk_samples
                end_idx = min(start_idx + chunk_samples, num_samples)

                # 提取块
                chunk = waveform[:, start_idx:end_idx]

                # GPU处理
                with torch.no_grad():
                    processed = self.dynamic_range_compression_gpu(
                        chunk,
                        threshold_db=threshold,
                        ratio=ratio,
                        attack_ms=attack,
                        release_ms=release,
                        sample_rate=sample_rate
                    )

                # 移回CPU
                processed_chunks.append(processed.cpu())

                # 清理GPU缓存
                del chunk, processed
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

                pbar.update(1)

        process_time = time.time() - process_start
        logger.info(f"✓ GPU处理完成: {process_time:.1f}秒")
        logger.info(f"  处理速度: {duration_sec/process_time:.1f}x 实时速度")

        # ===== 步骤3: 合并并导出 =====
        logger.info(f"\n[3/3] 合并并导出 ({output_format.upper()}格式)...")
        merge_start = time.time()

        # 合并所有块
        final_waveform = torch.cat(processed_chunks, dim=1)

        # 转换回整数格式
        final_samples = (final_waveform.numpy() * 32767.0).astype(np.int16)

        # 如果是立体声，需要转换回交错格式
        if num_channels == 2:
            final_samples = final_samples.T.flatten()
        else:
            final_samples = final_samples.flatten()

        # 使用pydub导出
        final_audio = AudioSegment(
            final_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=num_channels
        )

        # 根据格式设置导出参数
        export_params = {"format": output_format}

        if output_format == "mp3":
            # MP3参数：比特率192kbps，质量好且文件小
            export_params["bitrate"] = "192k"
            export_params["parameters"] = ["-q:a", "2"]  # 高质量
        elif output_format == "wav":
            # WAV是无损格式，文件会较大
            pass
        elif output_format == "m4a":
            # AAC编码，比特率192kbps
            export_params["bitrate"] = "192k"
            export_params["codec"] = "aac"
        elif output_format == "flac":
            # FLAC无损压缩，比WAV小但仍是无损
            pass

        final_audio.export(output_path, **export_params)

        merge_time = time.time() - merge_start
        logger.info(f"✓ 导出完成: {merge_time:.1f}秒")

        output_size = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"  输出大小: {output_size:.1f} MB")

        # ===== 总结 =====
        total_time = time.time() - total_start
        logger.info("\n" + "=" * 70)
        logger.info("✓ 任务完成!")
        logger.info(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        logger.info(f"音频时长: {duration_sec:.1f}秒 ({duration_sec/60:.1f}分钟)")
        logger.info(f"加速比: {duration_sec/total_time:.2f}x")
        logger.info("=" * 70)


def process_audio_gpu(input_path: str,
                     output_path: str = None,
                     threshold: float = -20.0,
                     ratio: float = 4.0,
                     attack: float = 5.0,
                     release: float = 50.0,
                     output_format: str = "mp3"):
    """
    GPU加速音频处理入口函数

    Args:
        output_format: 输出格式 (mp3/wav/m4a/flac)，默认mp3以减小文件体积
    """
    if output_path is None:
        input_file = Path(input_path)
        # 根据格式设置扩展名
        output_path = str(input_file.parent / f"{input_file.stem}_output.{output_format}")

    processor = GPUAudioProcessor(device='cuda')
    processor.process_audio_file(
        input_path=input_path,
        output_path=output_path,
        threshold=threshold,
        ratio=ratio,
        attack=attack,
        release=release,
        chunk_duration=60.0,  # 60秒一块
        output_format=output_format
    )


if __name__ == "__main__":
    # 检查CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    else:
        logger.warning("CUDA不可用，将使用CPU模式")
        logger.warning("如需GPU加速，请安装: pip install torch --index-url https://download.pytorch.org/whl/cu130\n")

    # 配置路径
    INPUT_FILE = r"D:\FunASR_installer\真实会议测试音频\2025-11-10 数据治理产品分享恩核供应商.m4a"
    OUTPUT_FILE = r"D:\FunASR_installer\真实会议测试音频\2025-11-10 数据治理产品分享恩核供应商_output.wav"

    # 执行GPU加速处理
    process_audio_gpu(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0
    )