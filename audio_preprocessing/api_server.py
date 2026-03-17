"""
音频处理Web API服务器
提供RESTful API接口供前端调用
"""
import logging
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional
import shutil
import xml.etree.ElementTree as ET

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import Send

from audio_preprocessing_gpu import GPUAudioProcessor

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

# 项目路径配置
PROMPT_TEMPLATES_DIR = project_root / "formatted_prompt_templates"
HOTWORD_DIR = project_root / "hotword"


class ProcessRequest(BaseModel):
    """音频处理请求"""
    threshold: float = -20.0
    ratio: float = 4.0
    attack: float = 5.0
    release: float = 50.0
    output_format: str = "mp3"  # 输出格式：mp3/wav/m4a/flac


class TextProcessRequest(BaseModel):
    """文本处理请求"""
    text: str
    config: dict


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


# ============= 文本处理API端点 =============

@app.get("/text")
async def text_processor_page():
    """返回文本处理页面"""
    text_processor_file = templates_dir / "text_processor.html"
    if text_processor_file.exists():
        return FileResponse(str(text_processor_file))
    return {"message": "文本处理页面未找到"}


@app.get("/api/text/prompt/{prompt_type}")
async def get_prompt_template(prompt_type: str):
    """
    获取提示词模板

    Args:
        prompt_type: 提示词类型 (formatter 或 translator)

    Returns:
        JSON响应，包含提示词内容
    """
    try:
        if prompt_type == "formatter":
            prompt_file = PROMPT_TEMPLATES_DIR / "chinese_formatter_prompt.xml"
        elif prompt_type == "translator":
            prompt_file = PROMPT_TEMPLATES_DIR / "english_translator_prompt.xml"
        else:
            raise HTTPException(status_code=400, detail=f"不支持的提示词类型: {prompt_type}")

        if not prompt_file.exists():
            raise HTTPException(status_code=404, detail=f"提示词文件不存在: {prompt_file}")

        # 读取XML文件并提取instructions部分
        tree = ET.parse(prompt_file)
        root = tree.getroot()

        instructions_elem = root.find('.//instructions')
        if instructions_elem is not None:
            # 将XML元素转换为字符串
            prompt_str = ET.tostring(instructions_elem, encoding='unicode', method='xml')
            # 去除XML声明
            prompt_str = prompt_str.replace('<?xml version="1.0" encoding="UTF-8"?>', '').strip()
            return {"prompt": prompt_str}
        else:
            raise HTTPException(status_code=500, detail="XML文件中未找到instructions元素")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载提示词失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载提示词失败: {str(e)}")


@app.get("/api/text/hotwords")
async def get_hotwords():
    """
    获取热词表

    Returns:
        JSON响应，包含热词表内容
    """
    try:
        hotword_file = HOTWORD_DIR / "中译英对照词库.txt"

        if not hotword_file.exists():
            return {"hotwords": "", "count": 0}

        with open(hotword_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 统计热词数量（非空行）
        lines = content.strip().split('\n')
        count = len([line for line in lines if line.strip()])

        return {"hotwords": content, "count": count}

    except Exception as e:
        logger.error(f"加载热词表失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载热词表失败: {str(e)}")


async def stream_text_process(process_type: str, request: TextProcessRequest):
    """
    流式处理文本的异步生成器

    Args:
        process_type: 处理类型 (formatter 或 translator)
        request: 处理请求

    Yields:
        JSON格式的SSE数据
    """
    import time
    from datetime import datetime
    import importlib.util
    import re

    start_time = time.time()

    def safe_json_dumps(obj):
        """安全的JSON序列化，转义特殊字符"""
        def escape_special_chars(text):
            if not isinstance(text, str):
                return text
            # 转义换行符、回车符、制表符等
            text = text.replace('\\', '\\\\')
            text = text.replace('\n', '\\n')
            text = text.replace('\r', '\\r')
            text = text.replace('\t', '\\t')
            text = text.replace('"', '\\"')
            return text

        if isinstance(obj, dict):
            return '{' + ', '.join(f'"{k}": "{escape_special_chars(v)}"' if isinstance(v, str) else f'"{k}": {v}' for k, v in obj.items()) + '}'
        return json.dumps(obj, ensure_ascii=False)

    async def send_log(level: str, message: str):
        """发送日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        logger.info(f"[{level.upper()}] {message}")

        # 转义消息中的特殊字符
        safe_message = message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

        data = {
            "type": "log",
            "level": level,
            "message": safe_message,
            "timestamp": timestamp
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        # 发送开始日志
        async for chunk in send_log('info', f'开始执行{process_type}处理'):
            yield chunk

        # 动态导入相应的类（处理带连字符的文件名）
        if process_type == "formatter":
            module_path = project_root / "chinese_formatter-ollama.py"
            spec = importlib.util.spec_from_file_location("chinese_formatter_ollama", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["chinese_formatter_ollama"] = module
            spec.loader.exec_module(module)
            processor_class = module.ChineseFormatter
            process_method = "process_transcript"
        elif process_type == "translator":
            module_path = project_root / "english_translator-ollama.py"
            spec = importlib.util.spec_from_file_location("english_translator_ollama", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["english_translator_ollama"] = module
            spec.loader.exec_module(module)
            processor_class = module.EnglishTranslator
            process_method = "translate_to_english"
        else:
            raise ValueError(f"不支持的处理类型: {process_type}")

        # 获取配置
        config = request.config
        model_name = config.get('model_name', '')
        lm_studio_url = config.get('lm_studio_url', 'http://127.0.0.1:1234')
        model_options = config.get('model_options', {})
        seed = model_options.get('seed', 42)

        async for chunk in send_log('info', f'使用模型: {model_name}'):
            yield chunk
        async for chunk in send_log('info', f'API地址: {lm_studio_url}'):
            yield chunk

        # 创建处理器实例
        async for chunk in send_log('info', '初始化处理器...'):
            yield chunk

        if process_type == "formatter":
            processor = processor_class(
                lm_studio_url=lm_studio_url,
                model_name=model_name,
                seed=seed
            )
        else:
            processor = processor_class(
                lm_studio_url=lm_studio_url,
                model_name=model_name,
                seed=seed
            )

        # 更新模型参数
        processor.model_options.update(model_options)

        # 如果提供了自定义提示词，更新提示词
        custom_prompt = config.get('prompt', '')
        if custom_prompt:
            if process_type == "formatter":
                processor.processing_prompt = custom_prompt
            else:
                processor.translation_prompt = custom_prompt
            async for chunk in send_log('info', '使用自定义提示词'):
                yield chunk

        # 如果提供了热词（仅translator）
        if process_type == "translator" and config.get('hotwords'):
            hotwords = config['hotwords']
            hotword_dict = {}
            for line in hotwords.strip().split('\n'):
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        chinese = parts[0].strip()
                        english = parts[1].strip()
                        if chinese and english:
                            hotword_dict[chinese] = english

            processor.hotword_dict = hotword_dict
            async for chunk in send_log('info', f'已加载 {len(hotword_dict)} 个自定义热词'):
                yield chunk

        # 创建自定义的print函数来实时发送日志
        log_queue = []
        processing_done = False

        def custom_print(*args, **kwargs):
            """捕获print输出并立即发送"""
            message = ' '.join(str(arg) for arg in args)
            log_queue.append(message)

        # 临时替换print函数
        import builtins
        original_print = builtins.print
        builtins.print = custom_print

        try:
            # 执行处理
            async for chunk in send_log('process', '开始处理文本...'):
                yield chunk

            # 在后台线程中执行处理，同时定期发送队列中的日志
            import threading
            import queue

            result_container = []
            exception_container = []

            def process_in_thread():
                try:
                    if process_type == "formatter":
                        result = getattr(processor, process_method)(request.text)
                    else:
                        result = getattr(processor, process_method)(request.text)
                    result_container.append(result)
                except Exception as e:
                    exception_container.append(e)

            # 启动处理线程
            process_thread = threading.Thread(target=process_in_thread)
            process_thread.start()

            # 定期发送日志
            last_log_count = 0
            while process_thread.is_alive():
                # 发送新增的日志
                while len(log_queue) > last_log_count:
                    log_message = log_queue[last_log_count]
                    last_log_count += 1

                    # 过滤无意义的日志：空行、只有点号、只有空格等
                    is_meaningful = (
                        log_message.strip() and  # 非空
                        log_message.strip() != '.' and  # 不是单个点号
                        log_message.strip() != '..' and  # 不是两个点号
                        log_message.strip() != '...' and  # 不是三个点号
                        not log_message.strip().startswith('[生成中') and  # 不是生成中标记
                        not log_message.strip() == ']' and  # 不是结束标记
                        not log_message.strip().startswith('[无输出') and  # 不是无输出警告
                        not log_message.replace('.', '').replace(' ', '').strip() == ''  # 不全是点号和空格
                    )

                    if not is_meaningful:
                        continue

                    # 确定日志级别
                    if any(marker in log_message for marker in ['✓', '完成', '成功']):
                        log_level = 'success'
                    elif any(marker in log_message for marker in ['⚠', '注意', '警告']):
                        log_level = 'warning'
                    elif any(marker in log_message for marker in ['✗', '❌', '错误', '失败']):
                        log_level = 'error'
                    else:
                        log_level = 'process'

                    async for chunk in send_log(log_level, log_message):
                        yield chunk

                # 等待一小段时间再检查
                await asyncio.sleep(0.1)

            # 发送剩余的日志
            while len(log_queue) > last_log_count:
                log_message = log_queue[last_log_count]
                last_log_count += 1

                # 过滤无意义的日志
                is_meaningful = (
                    log_message.strip() and
                    log_message.strip() != '.' and
                    log_message.strip() != '..' and
                    log_message.strip() != '...' and
                    not log_message.strip().startswith('[生成中') and
                    not log_message.strip() == ']' and
                    not log_message.strip().startswith('[无输出') and
                    not log_message.replace('.', '').replace(' ', '').strip() == ''
                )

                if not is_meaningful:
                    continue

                if any(marker in log_message for marker in ['✓', '完成', '成功']):
                    log_level = 'success'
                elif any(marker in log_message for marker in ['⚠', '注意', '警告']):
                    log_level = 'warning'
                elif any(marker in log_message for marker in ['✗', '❌', '错误', '失败']):
                    log_level = 'error'
                else:
                    log_level = 'process'

                async for chunk in send_log(log_level, log_message):
                    yield chunk

            # 检查是否有异常
            if exception_container:
                raise exception_container[0]

            # 获取结果
            result = result_container[0] if result_container else ""

        finally:
            # 恢复原始print函数
            builtins.print = original_print

        # 发送完成日志
        elapsed_time = time.time() - start_time
        async for chunk in send_log('info', f'处理完成，耗时 {elapsed_time:.1f} 秒'):
            yield chunk

        # 流式发送结果
        if result:
            result_lines = result.split('\n')
            total_lines = len(result_lines)

            for i, line in enumerate(result_lines):
                # 转义特殊字符
                safe_line = line.replace('\\', '\\\\').replace('"', '\\"')

                data = {
                    "type": "result",
                    "content": safe_line + ('\n' if i < total_lines - 1 else '')
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # 更新进度
                if total_lines > 0:
                    progress = min(int((i + 1) / total_lines * 100), 100)
                    progress_data = {
                        "type": "progress",
                        "progress": progress
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"

            # 发送完成消息
            complete_data = {
                "type": "complete",
                "stats": {
                    "input_length": len(request.text),
                    "output_length": len(result),
                    "duration": f"{elapsed_time:.1f}"
                }
            }
            yield f"data: {json.dumps(complete_data)}\n\n"

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        # 发送错误日志
        async for chunk in send_log('error', str(e)):
            yield chunk
        # 发送错误事件
        error_data = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/api/text/run/{process_type}")
async def run_text_process(process_type: str, request: TextProcessRequest):
    """
    执行文本处理（流式响应）

    Args:
        process_type: 处理类型 (formatter 或 translator)
        request: 处理请求

    Returns:
        流式响应
    """
    if process_type not in ["formatter", "translator"]:
        raise HTTPException(status_code=400, detail=f"不支持的处理类型: {process_type}")

    return StreamingResponse(
        stream_text_process(process_type, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
