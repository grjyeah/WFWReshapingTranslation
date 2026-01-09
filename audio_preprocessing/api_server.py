"""
音频处理Web API服务器
提供RESTful API接口供前端调用
"""
import logging
import asyncio
from pathlib import Path
from typing import Optional
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audio_preprocessing_gpu import GPUAudioProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="GPU音频处理API",
    description="基于RTX 5090的GPU加速音频动态范围压缩服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件目录
TEMP_DIR = Path("./temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# 静态文件目录
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 模板目录
templates_dir = Path(__file__).parent / "templates"

# 处理任务状态存储
processing_tasks = {}


class ProcessRequest(BaseModel):
    """音频处理请求"""
    threshold: float = -20.0
    ratio: float = 4.0
    attack: float = 5.0
    release: float = 50.0
    output_format: str = "mp3"  # 输出格式：mp3/wav/m4a/flac


@app.get("/")
async def root():
    """返回Web界面"""
    index_file = templates_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "GPU音频处理API服务",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "process": "/api/process/{filename}",
            "download": "/api/download/{filename}",
            "status": "/api/status/{task_id}",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    import torch
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "temp_dir": str(TEMP_DIR.absolute())
    }


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    上传音频/视频文件

    Args:
        file: 音频或视频文件

    Returns:
        JSON响应，包含文件名和存储路径
    """
    # 验证文件类型
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    allowed_extensions = audio_extensions | video_extensions

    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file_ext}"
        )

    # 保存文件
    file_path = TEMP_DIR / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_type = "视频" if file_ext in video_extensions else "音频"
        logger.info(f"文件上传成功: {file.filename} -> {file_path}")

        # 如果是视频文件，提取音频
        actual_filename = file.filename
        actual_file_path = str(file_path)

        if file_ext in video_extensions:
            logger.info(f"检测到视频文件，正在提取音频...")
            try:
                # 使用moviepy提取音频
                from moviepy.editor import VideoFileClip

                # 生成音频文件名
                audio_filename = f"{Path(file.filename).stem}.wav"
                audio_file_path = TEMP_DIR / audio_filename

                # 提取音频
                logger.info(f"开始提取音频: {file.filename} -> {audio_filename}")
                video_clip = VideoFileClip(str(file_path))
                video_clip.audio.write_audiofile(
                    str(audio_file_path),
                    codec='pcm_s16le',
                    fps=44100
                )
                video_clip.close()

                logger.info(f"音频提取完成: {audio_filename}")

                # 删除原视频文件以节省空间
                file_path.unlink()

                # 更新文件信息
                actual_filename = audio_filename
                actual_file_path = str(audio_file_path)

                return {
                    "success": True,
                    "filename": actual_filename,
                    "path": actual_file_path,
                    "size": audio_file_path.stat().st_size,
                    "message": f"视频文件，音频已提取",
                    "file_type": "video",
                    "original_file": file.filename
                }

            except ImportError:
                logger.error("需要安装moviepy: pip install moviepy")
                raise HTTPException(
                    status_code=500,
                    detail="视频处理功能需要安装moviepy库"
                )
            except Exception as e:
                logger.error(f"音频提取失败: {e}")
                # 提取失败，删除视频文件
                file_path.unlink()
                raise HTTPException(
                    status_code=500,
                    detail=f"音频提取失败: {str(e)}"
                )

        return {
            "success": True,
            "filename": actual_filename,
            "path": actual_file_path,
            "size": file_path.stat().st_size,
            "message": f"{file_type}文件上传成功",
            "file_type": "audio"
        }

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@app.post("/api/process/{filename}")
async def process_audio(filename: str, params: ProcessRequest, background_tasks: BackgroundTasks):
    """
    处理音频文件（后台任务）

    Args:
        filename: 音频文件名
        params: 处理参数
        background_tasks: 后台任务

    Returns:
        JSON响应，包含任务ID
    """
    input_path = TEMP_DIR / filename
    output_filename = f"{Path(filename).stem}_output.{params.output_format}"
    output_path = TEMP_DIR / output_filename

    # 验证文件存在
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    # 生成任务ID
    import uuid
    task_id = str(uuid.uuid4())

    # 初始化任务状态
    processing_tasks[task_id] = {
        "status": "pending",
        "input_file": filename,
        "output_file": output_filename,
        "progress": 0,
        "message": "任务已创建，等待处理..."
    }

    # 添加后台任务
    background_tasks.add_task(
        process_audio_task,
        task_id,
        str(input_path),
        str(output_path),
        params
    )

    return {
        "success": True,
        "task_id": task_id,
        "message": "音频处理任务已创建",
        "output_file": output_filename
    }


async def process_audio_task(
    task_id: str,
    input_path: str,
    output_path: str,
    params: ProcessRequest
):
    """
    执行音频处理任务

    Args:
        task_id: 任务ID
        input_path: 输入文件路径
        output_path: 输出文件路径
        params: 处理参数
    """
    try:
        # 更新任务状态
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "正在处理音频..."

        # 创建处理器
        processor = GPUAudioProcessor(device='cuda')

        # 处理音频
        processor.process_audio_file(
            input_path=input_path,
            output_path=output_path,
            threshold=params.threshold,
            ratio=params.ratio,
            attack=params.attack,
            release=params.release,
            chunk_duration=60.0,
            output_format=params.output_format
        )

        # 更新任务状态
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["message"] = "处理完成！"

    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {e}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["message"] = f"处理失败: {str(e)}"


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态

    Args:
        task_id: 任务ID

    Returns:
        JSON响应，包含任务状态
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    return processing_tasks[task_id]


@app.get("/api/download/{filename}")
async def download_audio(filename: str):
    """
    下载处理后的音频文件

    Args:
        filename: 文件名

    Returns:
        文件响应
    """
    file_path = TEMP_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    return FileResponse(
        path=file_path,
        media_type='audio/wav',
        filename=filename
    )


@app.delete("/api/cleanup")
async def cleanup_files():
    """
    清理临时文件
    """
    try:
        file_count = 0
        for file in TEMP_DIR.glob("*"):
            if file.is_file():
                file.unlink()
                file_count += 1

        logger.info(f"已清理 {file_count} 个临时文件")
        return {"success": True, "cleaned_files": file_count}

    except Exception as e:
        logger.error(f"清理失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import sys
    import torch

    logger.info("=" * 70)
    logger.info("启动GPU音频处理Web服务")
    logger.info("=" * 70)
    logger.info(f"Python路径: {sys.executable}")
    logger.info(f"Python版本: {sys.version.split()[0]}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("⚠ CUDA不可用，将使用CPU模式")
        logger.warning("如需GPU加速，请确保：")
        logger.warning("  1. 在正确的conda环境中启动")
        logger.warning("  2. 安装了GPU版本的PyTorch")
        logger.warning("  3. 使用命令: conda activate your_env && python api_server.py")
    logger.info("=" * 70)
    logger.info(f"API地址: http://localhost:8001")
    logger.info(f"API文档: http://localhost:8001/docs")
    logger.info(f"临时目录: {TEMP_DIR.absolute()}")
    logger.info("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
