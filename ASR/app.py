#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR WebUI - 基于Gradio的语音识别Web界面
功能：语音识别、VAD、标点恢复、说话人分离、情感识别等
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import gradio as gr
import numpy as np
import pandas as pd

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError as e:
    print(f"Error importing FunASR: {e}")
    print("Please install FunASR first: pip install funasr")
    sys.exit(1)

# 全局变量存储已加载的模型
LOADED_MODELS = {}
MODEL_CONFIG = {
    "asr_models": {
        "SenseVoiceSmall": "/root/FunASR/models/modelscope/iic/SenseVoiceSmall",
        "Paraformer-zh": "/root/FunASR/models/modelscope/iic/paraformer-zh",
        "Paraformer-zh-streaming": "/root/FunASR/models/modelscope/iic/paraformer-zh-streaming",
        "SeACoParaformer": "/root/FunASR/models/modelscope/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "EngParaformer": "/root/FunASR/models/modelscope/iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        "Paraformer-en": "paraformer-en",
        "Conformer-en": "/root/FunASR/models/modelscope/iic/conformer-en",
        "Whisper-large-v3": "/root/FunASR/models/modelscope/iic/Whisper-large-v3",
        "Whisper-large-v3-turbo": "/root/FunASR/models/modelscope/iic/Whisper-large-v3-turbo"
    },
    "vad_models": {
        "None": None,
        "FSMN-VAD": "/root/FunASR/models/modelscope/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    },
    "punc_models": {
        "None": None,
        "CT-Transformer": "/root/FunASR/models/modelscope/iic/punc_ct-transformer_cn-en-common-vocab471067-large"
    },
    "spk_models": {
        "None": None,
        "CAM++": "/root/FunASR/models/modelscope/iic/speech_campplus_sv_zh-cn_16k-common"
    },
    "emotion_models": {
        "None": None,
        "Emotion2Vec-Large": "emotion2vec_plus_large",
        "Emotion2Vec-Base": "emotion2vec_plus_base"
    }
}

LANGUAGE_OPTIONS = {
    "auto": "自动检测",
    "zh": "中文",
    "en": "英文",
    "yue": "粤语",
    "ja": "日语",
    "ko": "韩语"
}

# 语言标签映射表
LANGUAGE_TAG_MAPPING = {
    "<|zh|>": "中文：",
    "<|en|>": "英文：",
    "<|yue|>": "粤语：",
    "<|ja|>": "日语：",
    "<|ko|>": "韩语：",
    "<|auto|>": "自动检测："
}

def postprocess_sensevoice_result(text: str) -> str:
    """后处理SenseVoiceSmall结果，替换语言标签为可读前缀

    Args:
        text: 原始识别文本，可能包含语言标签

    Returns:
        处理后的文本，语言标签被替换为可读前缀并添加换行
    """
    if not text:
        return text

    import re

    # 使用正则表达式匹配语言标签模式
    # 匹配 <|语言|> 格式的标签
    pattern = r'<\|([^|]+)\|>'

    def replace_tag(match):
        language = match.group(1)
        tag = f"<|{language}|>"
        prefix = LANGUAGE_TAG_MAPPING.get(tag, f"{language}：")
        # 为第一个标签前不添加换行，后续标签前添加换行
        if match.start() == 0:
            return f"{prefix} "
        else:
            return f"\n{prefix} "

    # 执行替换
    result = re.sub(pattern, replace_tag, text)

    # 清理多余的空格，但保留换行符
    # 首先处理换行符周围的空格
    result = re.sub(r'\s*\n\s*', '\n', result)  # 清理换行前后的空格
    # 然后处理不跨换行符的连续空格
    result = re.sub(r'[^\S\r\n]+', ' ', result)  # 合并非换行符的空白字符
    # 清理每行末尾的空格
    result = re.sub(r' +\n', '\n', result)  # 清理换行前的空格
    result = result.strip()  # 去除首尾空白

    return result


def format_timestamp(milliseconds: float) -> str:
    """将毫秒转换为"X时X分X秒"格式

    Args:
        milliseconds: 毫秒数

    Returns:
        格式化后的时间字符串，如"0时22分44秒"
    """
    if milliseconds == 0:
        return "0时0分0秒"

    total_seconds = milliseconds / 1000

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours}时{minutes}分{seconds}秒"


class FunASRWebUI:
    def __init__(self):
        self.setup_environment()

    def get_model_capabilities(self, model_name: str) -> Dict[str, bool]:
        """获取模型功能支持情况"""
        # 根据FunASR文档，只有这两个模型支持时间戳预测和说话人分离
        timestamp_models = {
            "/root/FunASR/models/modelscope/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "/root/FunASR/models/modelscope/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            # "/root/FunASR/models/modelscope/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        }

        model_id = MODEL_CONFIG["asr_models"].get(model_name, "")

        return {
            "supports_timestamp": model_id in timestamp_models,
            "supports_speaker_diarization": model_id in timestamp_models,
            "supports_hotword": True,  # 大部分模型都支持热词
            "supports_multilingual": model_name in ["SenseVoiceSmall", "Whisper-large-v3", "Whisper-large-v3-turbo"],
            "model_id": model_id,
            "model_name": model_name
        }

    def get_optimal_device(self):
        """获取最优计算设备"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"🎯 检测到 {device_count} 个GPU设备，优先使用GPU")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🎯 检测到MPS设备，使用MPS加速")
                return "mps"
            else:
                print("⚠️  未检测到GPU设备，使用CPU")
                return "cpu"
        except Exception as e:
            print(f"⚠️  设备检测失败: {e}，回退到CPU")
            return "cpu"

    def setup_environment(self):
        """设置环境变量"""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        models_dir = current_dir / "models"
        models_dir.mkdir(exist_ok=True)

        hf_cache = models_dir / "huggingface"
        ms_cache = models_dir / "modelscope"
        hf_cache.mkdir(exist_ok=True)
        ms_cache.mkdir(exist_ok=True)

        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
        os.environ["MODELSCOPE_CACHE"] = str(ms_cache)

    def load_model(self, model_key: str, model_config: Dict) -> Any:
        """加载模型"""
        if model_key in LOADED_MODELS:
            return LOADED_MODELS[model_key]

        try:
            # 添加disable_update=True禁用自动更新检查
            model_config["disable_update"] = True
            model = AutoModel(**model_config)
            LOADED_MODELS[model_key] = model
            return model
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")

    def create_basic_asr_interface(self):
        """创建基础语音识别界面"""
        with gr.Column():
            gr.Markdown("## 🎤 基础语音识别")
            gr.Markdown("上传音频文件或录制语音进行识别")

            with gr.Row():
                with gr.Column(scale=2, min_width=400):
                    # 输入区域
                    audio_input = gr.Audio(
                        label="音频输入",
                        type="filepath",
                        sources=["upload", "microphone"],
                        container=True
                    )

                    # 模型和语言选择（在一行中）
                    with gr.Row():
                        asr_model = gr.Dropdown(
                            choices=list(MODEL_CONFIG["asr_models"].keys()),
                            value="SeACoParaformer",  # 默认使用SeACoParaformer
                            label="语音识别模型",
                            scale=2
                        )

                        language = gr.Dropdown(
                            choices=[
                                ("自动检测", "auto"),
                                ("中文", "zh"),
                                ("英文", "en"),
                                ("粤语", "yue"),
                                ("日语", "ja"),
                                ("韩语", "ko")
                            ],
                            value="auto",
                            label="语言",
                            scale=1
                        )

                    # 功能选项（优化布局）
                    with gr.Row():
                        use_vad = gr.Checkbox(label="使用VAD", value=True)
                        use_punc = gr.Checkbox(label="使用标点", value=True)
                        use_itn = gr.Checkbox(label="使用ITN", value=True)

                    # 热词功能
                    with gr.Row():
                        use_hotword = gr.Checkbox(label="使用热词", value=False)

                    # 热词文件上传（根据开关显示/隐藏）
                    hotword_file = gr.File(
                        label="热词文件（可选）",
                        file_types=[".txt"],
                        type="filepath",
                        visible=False  # 默认隐藏，只在需要时显示
                    )

                    # 热词开关
                    use_hotword = gr.Checkbox(label="使用热词", value=False)

                    # 根据热词开关显示/隐藏文件上传
                    def toggle_hotword_visibility(use_hotword):
                        return gr.File(visible=use_hotword)

                    use_hotword.change(
                        toggle_hotword_visibility,
                        inputs=[use_hotword],
                        outputs=[hotword_file]
                    )

                    # 识别按钮（增大尺寸）
                    recognize_btn = gr.Button(
                        "🚀 开始识别",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=3, min_width=500):
                    # 输出区域（增加间距）
                    with gr.Group():
                        result_text = gr.Textbox(
                            label="识别结果",
                            lines=10,  # 增加行数
                            placeholder="识别结果将在这里显示...",
                            max_lines=20,
                            interactive=False,
                            container=True
                        )

                        status_info = gr.Textbox(
                            label="状态信息",
                            lines=4,  # 增加行数
                            placeholder="准备就绪",
                            interactive=False,
                            container=True
                        )

                    # 结果详情（优化展示）
                    with gr.Accordion("📊 详细信息", open=False):
                        result_json = gr.JSON(
                            label="完整结果",
                            container=True
                        )

            # 识别音频函数
            def recognize_audio(audio_path, model_name, lang, vad, punc, itn, use_hotword, hotword_file=None):
                if not audio_path:
                    return "请先上传或录制音频文件", "错误：没有音频输入", {}

                try:
                    status_msg = f"正在加载模型 {model_name}..."
                    yield "", status_msg, {}

                    # 构建模型配置
                    config = {
                        "model": MODEL_CONFIG["asr_models"][model_name],
                        "device": self.get_optimal_device()
                    }

                    # 如果是SeACoParaformer模型，添加版本号
                    if model_name == "SeACoParaformer":
                        config["model_revision"] = "v2.0.4"

                    if vad:
                        config["vad_model"] = "fsmn-vad"
                        config["vad_kwargs"] = {"max_single_segment_time": 30000}
                        # 如果是SeACoParaformer模型，添加VAD版本号
                        if model_name == "SeACoParaformer":
                            config["vad_model_revision"] = "v2.0.4"

                    if punc:
                        config["punc_model"] = "ct-punc"
                        # 如果是SeACoParaformer模型，添加标点版本号
                        if model_name == "SeACoParaformer":
                            config["punc_model_revision"] = "v2.0.4"

                    # 加载模型
                    model_key = f"{model_name}_vad-{vad}_punc-{punc}"
                    model = self.load_model(model_key, config)

                    status_msg = "正在识别中..."
                    yield "", status_msg, {}

                    # 执行识别
                    start_time = time.time()

                    # 构建生成参数
                    generate_kwargs = {
                        "input": audio_path,
                        "batch_size_s": 300,
                        "merge_vad": vad
                    }

                    # 为不同模型添加特定参数
                    if model_name == "SenseVoiceSmall":
                        generate_kwargs.update({
                            "cache": {},
                            "language": lang,
                            "use_itn": itn,
                            "batch_size_s": 60,
                            "merge_length_s": 15
                        })
                    elif model_name.startswith("Whisper"):
                        # Whisper模型特殊处理
                        generate_kwargs["batch_size_s"] = 0  # 设置为0确保batch_size=1

                    # 添加热词参数（如果启用了热词功能）
                    if use_hotword and hotword_file and os.path.exists(hotword_file):
                        generate_kwargs["hotword"] = hotword_file
                        status_msg = "正在识别中（使用热词）..."
                        yield "", status_msg, {}

                    # 执行识别
                    res = model.generate(**generate_kwargs)

                    if res and len(res) > 0:
                        if model_name == "SenseVoiceSmall":
                            raw_text = rich_transcription_postprocess(res[0]["text"]) if res[0]["text"] else "未识别到内容"
                            text = postprocess_sensevoice_result(raw_text)
                        else:
                            text = res[0].get("text", "未识别到内容")
                        result_detail = res[0]
                    else:
                        text = "未识别到内容"
                        result_detail = {}

                    end_time = time.time()
                    duration = end_time - start_time

                    status_msg = f"识别完成！耗时: {duration:.2f}秒"
                    if use_hotword and hotword_file:
                        status_msg += "（使用热词）"

                    yield text, status_msg, result_detail

                except Exception as e:
                    error_msg = f"识别失败: {str(e)}"

                    # 检查是否是模型相关的错误
                    if "timestamp" in str(e).lower() or "speaker" in str(e).lower():
                        if model_name == "SenseVoiceSmall":
                            error_msg = f"识别失败: SenseVoiceSmall模型遇到了配置问题。\n建议：检查高级功能设置或尝试其他模型。\n详细错误: {str(e)}"
                        else:
                            error_msg = f"识别失败: {model_name}模型遇到了配置问题。\n详细错误: {str(e)}"

                    yield "", error_msg, {"error": str(e)}

            # 按钮点击事件
            recognize_btn.click(
                recognize_audio,
                inputs=[audio_input, asr_model, language, use_vad, use_punc, use_itn, use_hotword, hotword_file],
                outputs=[result_text, status_info, result_json]
            )

        return audio_input, asr_model, language, use_vad, use_punc, use_itn, use_hotword, hotword_file, recognize_btn, result_text, status_info, result_json

    def create_advanced_interface(self):
        """创建高级功能界面"""
        with gr.Column():
            gr.Markdown("## ⚙️ 高级功能")
            gr.Markdown("包含说话人分离、情感识别、时间戳等高级功能")

            with gr.Row():
                with gr.Column(scale=2, min_width=450):
                    # 音频输入
                    audio_input_adv = gr.Audio(
                        label="音频输入",
                        type="filepath",
                        sources=["upload", "microphone"],
                        container=True
                    )

                    # 模型选择组（优化布局）
                    with gr.Group():
                        gr.Markdown("### 🤖 模型选择")

                        # ASR和VAD模型在一行
                        with gr.Row():
                            asr_model_adv = gr.Dropdown(
                                choices=list(MODEL_CONFIG["asr_models"].keys()),
                                value="SeACoParaformer",  # 默认使用SeACoParaformer
                                label="ASR模型",
                                scale=2
                            )
                            vad_model_adv = gr.Dropdown(
                                choices=list(MODEL_CONFIG["vad_models"].keys()),
                                value="FSMN-VAD",
                                label="VAD模型",
                                scale=1
                            )

                        # 标点和说话人模型在一行
                        with gr.Row():
                            punc_model_adv = gr.Dropdown(
                                choices=list(MODEL_CONFIG["punc_models"].keys()),
                                value="CT-Transformer",
                                label="标点模型",
                                scale=1
                            )
                            spk_model_adv = gr.Dropdown(
                                choices=list(MODEL_CONFIG["spk_models"].keys()),
                                value="CAM++",
                                label="说话人模型",
                                scale=1
                            )

                    # 热词功能（高级界面）
                    with gr.Group():
                        gr.Markdown("### 🔥🔥 热词功能")

                        with gr.Row():
                            use_hotword_adv = gr.Checkbox(label="使用热词", value=False)

                        hotword_file_adv = gr.File(
                            label="热词文件（可选）",
                            file_types=[".txt"],
                            type="filepath",
                            visible=False
                        )

                        # 根据热词开关显示/隐藏文件上传
                        def toggle_hotword_visibility_adv(use_hotword):
                            return gr.File(visible=use_hotword)

                        use_hotword_adv.change(
                            toggle_hotword_visibility_adv,
                            inputs=[use_hotword_adv],
                            outputs=[hotword_file_adv]
                        )

                    # 高级参数（优化布局）
                    with gr.Group():
                        gr.Markdown("### ⚙️ 高级参数")

                        # 数值参数在一行
                        with gr.Row():
                            batch_size = gr.Slider(
                                60, 600, value=300, step=60,
                                label="批处理大小(秒)",
                                interactive=True
                            )
                            merge_length = gr.Slider(
                                5, 30, value=15, step=5,
                                label="VAD合并长度(秒)",
                                interactive=True
                            )

                        # 功能开关在一行
                        with gr.Row():
                            return_timestamps = gr.Checkbox(
                                label="返回时间戳", value=True
                            )
                            sentence_timestamp = gr.Checkbox(
                                label="句子级时间戳", value=True
                            )
                            return_spk_res = gr.Checkbox(
                                label="返回说话人结果", value=True
                            )

                        # 模型功能提示
                        model_capability_info = gr.Textbox(
                            label="模型功能提示",
                            lines=2,
                            interactive=False,
                            container=True
                        )

                    # 处理按钮（增大尺寸）
                    process_adv_btn = gr.Button(
                        "🔧 开始高级处理",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=3, min_width=600):
                    # 结果显示区域（只保留说话人分离标签页）
                    with gr.Group():
                        gr.Markdown("### 👥 说话人分离结果")

                        speaker_result = gr.Dataframe(
                            headers=["说话人", "开始时间", "结束时间", "文本"],
                            label="说话人分离结果",
                            interactive=False
                        )

                        # 一键复制功能
                        with gr.Row():
                            copy_btn = gr.Button(" 一键复制表格", variant="primary")

                        export_output = gr.Textbox(
                            label="复制结果",
                            lines=5,
                            show_copy_button=True  # 这个属性会显示复制按钮
                        )

                        def copy_speaker_results(speaker_data):
                            if speaker_data is None or speaker_data.empty:
                                return "没有数据可复制"

                            # 使用制表符分隔的格式，方便Excel自动分列
                            text = "说话人\t开始时间\t结束时间\t文本内容\n"

                            for index, row in speaker_data.iterrows():
                                speaker = str(row['说话人']).replace('\t', ' ')
                                start_time = str(row['开始时间']).replace('\t', ' ')
                                end_time = str(row['结束时间']).replace('\t', ' ')
                                # 文本列已经包含【说话人X】标签，直接使用即可
                                content = str(row['文本']).replace('\t', ' ').replace('\n', ' ')

                                text += f"{speaker}\t{start_time}\t{end_time}\t{content}\n"

                            return text

                        copy_btn.click(copy_speaker_results, inputs=[speaker_result], outputs=[export_output])

                    # 隐藏的输出组件（用于保持API兼容性）
                    text_result_adv = gr.Textbox(visible=False)
                    timestamp_result = gr.Dataframe(visible=False)
                    full_result_adv = gr.JSON(visible=False)

                    # 状态信息（放在标签页下方）
                    status_adv = gr.Textbox(
                        label="处理状态",
                        lines=4,
                        placeholder="准备就绪",
                        interactive=False,
                        container=True
                    )

            # 更新模型功能提示
            def update_model_capability_info(model_name):
                capabilities = self.get_model_capabilities(model_name)
                info_parts = []

                if not capabilities["supports_timestamp"]:
                    info_parts.append("⚠️ 不支持时间戳预测")
                if not capabilities["supports_speaker_diarization"]:
                    info_parts.append("⚠️ 不支持说话人分离")
                if capabilities["supports_multilingual"]:
                    info_parts.append("✅ 支持多语言识别")
                if capabilities["supports_hotword"]:
                    info_parts.append("✅ 支持热词功能")

                if info_parts:
                    return " | ".join(info_parts)
                else:
                    return "✅ 支持所有高级功能"

            # 高级处理函数
            def process_advanced(audio_path, asr_model, vad_model, punc_model, spk_model,
                                 batch_size, merge_length, timestamps, sent_timestamps, spk_res,
                                 use_hotword, hotword_file):
                if not audio_path:
                    return "请上传音频文件", pd.DataFrame(), pd.DataFrame(), {}, "错误：没有音频输入"

                try:
                    yield "", pd.DataFrame(), pd.DataFrame(), {}, "正在加载模型..."

                    # 构建模型配置
                    config = {
                        "model": MODEL_CONFIG["asr_models"][asr_model],
                        "device": self.get_optimal_device()
                    }

                    # 如果是SeACoParaformer模型，添加版本号
                    if asr_model == "SeACoParaformer":
                        config["model_revision"] = "v2.0.4"

                    if vad_model != "None":
                        config["vad_model"] = MODEL_CONFIG["vad_models"][vad_model]
                        config["vad_kwargs"] = {"max_single_segment_time": 30000}
                        if asr_model == "SeACoParaformer":
                            config["vad_model_revision"] = "v2.0.4"

                    if punc_model != "None":
                        config["punc_model"] = MODEL_CONFIG["punc_models"][punc_model]
                        if asr_model == "SeACoParaformer":
                            config["punc_model_revision"] = "v2.0.4"

                    if spk_model != "None":
                        config["spk_model"] = MODEL_CONFIG["spk_models"][spk_model]

                    # 加载模型
                    model_key = f"adv_{asr_model}_{vad_model}_{punc_model}_{spk_model}"
                    model = self.load_model(model_key, config)

                    yield "", pd.DataFrame(), pd.DataFrame(), {}, "正在处理音频..."

                    # 构建生成参数
                    generate_kwargs = {
                        "input": audio_path,
                        "batch_size_s": batch_size,
                        "merge_vad": True,
                        "merge_length_s": merge_length
                    }

                    # Whisper模型特殊处理
                    if asr_model.startswith("Whisper"):
                        generate_kwargs["batch_size_s"] = 0

                    # 检查模型能力
                    capabilities = self.get_model_capabilities(asr_model)

                    # 检查是否尝试对不支持时间戳的模型使用时间戳功能
                    if not capabilities["supports_timestamp"] and (sent_timestamps or spk_res):
                        if asr_model == "SenseVoiceSmall":
                            warning_msg = "SenseVoiceSmall模型不支持时间戳预测和说话人分离功能，将使用基础识别模式"
                        else:
                            warning_msg = f"{asr_model}模型不支持时间戳预测和说话人分离功能，将使用基础识别模式"

                        yield warning_msg, pd.DataFrame(), pd.DataFrame(), {}, "正在处理音频..."
                        # 不添加不支持的功能参数
                    elif capabilities["supports_timestamp"]:
                        # 只有支持的模型才添加这些参数
                        generate_kwargs["sentence_timestamp"] = sent_timestamps
                        generate_kwargs["return_spk_res"] = spk_res

                    # 添加热词参数（如果启用了热词功能）
                    if use_hotword and hotword_file and os.path.exists(hotword_file):
                        generate_kwargs["hotword"] = hotword_file
                        yield "", pd.DataFrame(), pd.DataFrame(), {}, "正在处理音频（使用热词）..."

                    # 执行识别
                    res = model.generate(**generate_kwargs)

                    if not res or len(res) == 0:
                        return "未识别到内容", pd.DataFrame(), pd.DataFrame(), {}, "识别完成，但未找到有效内容"

                    result = res[0]
                    raw_text = result.get("text", "")

                    # 对SenseVoiceSmall结果进行后处理
                    if asr_model == "SenseVoiceSmall":
                        text = postprocess_sensevoice_result(raw_text)
                    else:
                        text = raw_text

                    # 处理时间戳 - 只有支持的模型才会包含时间戳信息
                    timestamp_df = pd.DataFrame()
                    capabilities = self.get_model_capabilities(asr_model)

                    if capabilities["supports_timestamp"] and "timestamp" in result and result["timestamp"]:
                        timestamp_data = []
                        for ts in result["timestamp"]:
                            start_time = format_timestamp(ts[0])
                            end_time = format_timestamp(ts[1])
                            word = ts[2] if len(ts) > 2 else ""
                            timestamp_data.append([start_time, end_time, word])
                        timestamp_df = pd.DataFrame(timestamp_data, columns=["开始时间", "结束时间", "文本"])
                    elif not capabilities["supports_timestamp"] and (sent_timestamps or spk_res):
                        # 为不支持的模型创建提示信息
                        timestamp_df = pd.DataFrame([["不支持", "不支持", "当前模型不支持时间戳功能"]],
                                                   columns=["开始时间", "结束时间", "文本"])

                    # 处理说话人分离 - 只有支持的模型才会包含说话人信息
                    speaker_df = pd.DataFrame()
                    if capabilities["supports_speaker_diarization"] and "sentence_info" in result and result["sentence_info"]:
                        speaker_data = []
                        for sent in result["sentence_info"]:
                            speaker = sent.get("spk", "未知")
                            start_time = format_timestamp(sent.get('start', 0))
                            end_time = format_timestamp(sent.get('end', 0))
                            sentence = sent.get("sentence", "")
                            if not sentence:
                                sentence = sent.get("text", "")
                            if not sentence:
                                timestamp_text = sent.get("timestamp", [])
                                if timestamp_text and len(timestamp_text) > 0:
                                    sentence = " ".join([ts[2] for ts in timestamp_text if len(ts) > 2])

                            # 在文本前面添加说话人标签
                            sentence_with_label = f"【说话人:{speaker}】{sentence}"
                            speaker_data.append([speaker, start_time, end_time, sentence_with_label])
                        speaker_df = pd.DataFrame(speaker_data, columns=["说话人", "开始时间", "结束时间", "文本"])
                    elif not capabilities["supports_speaker_diarization"] and spk_res:
                        # 为不支持的模型创建提示信息
                        speaker_df = pd.DataFrame([["不支持", "不支持", "不支持", "当前模型不支持说话人分离功能"]],
                                                  columns=["说话人", "开始时间", "结束时间", "文本"])

                    status_msg = "处理完成！"
                    if use_hotword and hotword_file:
                        status_msg += "（使用热词）"

                    yield text, timestamp_df, speaker_df, result, status_msg

                except Exception as e:
                    error_msg = f"处理失败: {str(e)}"

                    # 检查是否是SenseVoiceSmall相关的错误
                    if "timestamp" in str(e).lower() and "speaker" in str(e).lower():
                        if asr_model == "SenseVoiceSmall":
                            error_msg = f"处理失败: SenseVoiceSmall模型不支持时间戳和说话人分离功能。请在高级参数中禁用这些功能后重试。\n详细错误: {str(e)}"
                        else:
                            error_msg = f"处理失败: {asr_model}模型不支持时间戳和说话人分离功能。请在高级参数中禁用这些功能后重试。\n详细错误: {str(e)}"

                    yield "", pd.DataFrame(), pd.DataFrame(), {"error": str(e)}, error_msg

            # 高级处理按钮点击事件
            process_adv_btn.click(
                process_advanced,
                inputs=[audio_input_adv, asr_model_adv, vad_model_adv, punc_model_adv, spk_model_adv,
                        batch_size, merge_length, return_timestamps, sentence_timestamp, return_spk_res,
                        use_hotword_adv, hotword_file_adv],
                outputs=[text_result_adv, timestamp_result, speaker_result, full_result_adv, status_adv]
            )

            # 模型选择变化时更新功能提示
            asr_model_adv.change(
                update_model_capability_info,
                inputs=[asr_model_adv],
                outputs=[model_capability_info]
            )

            # 初始化模型功能提示
            model_capability_info.value = update_model_capability_info("SeACoParaformer")

    def create_batch_interface(self):
        """创建批量处理界面"""
        with gr.Column():
            gr.Markdown("## 📁 批量处理")
            gr.Markdown("批量处理多个音频文件")

            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="选择多个音频文件",
                        file_count="multiple",
                        file_types=["audio"]
                    )

                    batch_model = gr.Dropdown(
                        choices=list(MODEL_CONFIG["asr_models"].keys()),
                        value="SenseVoiceSmall",
                        label="批处理模型"
                    )

                    batch_options = gr.CheckboxGroup(
                        choices=["使用VAD", "使用标点", "包含时间戳"],
                        value=["使用VAD", "使用标点"],
                        label="处理选项"
                    )

                    batch_btn = gr.Button("🚀 开始批量处理", variant="primary")

                with gr.Column(scale=2):
                    batch_progress = gr.Textbox(
                        label="处理进度",
                        lines=3,
                        placeholder="等待开始..."
                    )

                    batch_results = gr.Dataframe(
                        headers=["文件名", "状态", "识别结果", "处理时间"],
                        label="批处理结果"
                    )

                    download_btn = gr.DownloadButton(
                        label="📥 下载结果(CSV)",
                        visible=False
                    )

        def batch_process(files, model_name, options):
            if not files:
                return "请选择要处理的文件", pd.DataFrame(), gr.DownloadButton(visible=False)

            try:
                use_vad = "使用VAD" in options
                use_punc = "使用标点" in options
                use_timestamps = "包含时间戳" in options

                # 加载模型
                config = {
                    "model": MODEL_CONFIG["asr_models"][model_name],
                    "device": self.get_optimal_device()
                }

                if use_vad:
                    config["vad_model"] = "fsmn-vad"
                if use_punc:
                    config["punc_model"] = "ct-punc"

                model_key = f"batch_{model_name}_vad-{use_vad}_punc-{use_punc}"
                model = self.load_model(model_key, config)

                results = []
                total_files = len(files)

                for i, file in enumerate(files):
                    yield f"处理中 {i+1}/{total_files}: {file.name}", pd.DataFrame(results), gr.DownloadButton(visible=False)

                    try:
                        start_time = time.time()

                        # 执行识别
                        if model_name.startswith("Whisper"):
                            # Whisper模型不支持batch处理，设置batch_size_s=0
                            res = model.generate(
                                input=file.name,
                                batch_size_s=0,  # 关键修复：设置为0确保batch_size=1
                                sentence_timestamp=use_timestamps
                            )
                        elif model_name == "SenseVoiceSmall":
                            # SenseVoiceSmall模型不支持时间戳功能，检查用户请求
                            capabilities = self.get_model_capabilities(model_name)
                            if use_timestamps and not capabilities["supports_timestamp"]:
                                # 如果用户请求了时间戳，给出警告但继续处理
                                print(f"警告：SenseVoiceSmall模型不支持时间戳功能，将使用基础识别模式处理文件: {file.name}")
                            res = model.generate(
                                input=file.name,
                                batch_size_s=60,  # SenseVoiceSmall推荐的批处理大小
                                merge_length_s=15
                            )
                        else:
                            # 其他模型（包括SeACoParaformer等支持时间戳的模型）
                            capabilities = self.get_model_capabilities(model_name)
                            if capabilities["supports_timestamp"]:
                                # 支持时间戳的模型
                                res = model.generate(
                                    input=file.name,
                                    batch_size_s=300,
                                    sentence_timestamp=use_timestamps
                                )
                            else:
                                # 不支持时间戳的模型
                                if use_timestamps:
                                    print(f"警告：{model_name}模型不支持时间戳功能，将使用基础识别模式处理文件: {file.name}")
                                res = model.generate(
                                    input=file.name,
                                    batch_size_s=300
                                )

                        end_time = time.time()
                        processing_time = f"{end_time - start_time:.2f}s"

                        if res and len(res) > 0:
                            raw_text = res[0].get("text", "未识别到内容")
                            if model_name == "SenseVoiceSmall":
                                text = postprocess_sensevoice_result(raw_text)
                            else:
                                text = raw_text
                            status = "成功"
                        else:
                            text = "未识别到内容"
                            status = "无内容"

                        results.append([
                            file.name.split("/")[-1],
                            status,
                            text[:100] + "..." if len(text) > 100 else text,
                            processing_time
                        ])

                    except Exception as e:
                        error_text = f"错误: {str(e)}"

                        # 检查是否是时间戳或说话人分离相关的错误
                        if "timestamp" in str(e).lower() or "speaker" in str(e).lower():
                            if model_name == "SenseVoiceSmall":
                                error_text = f"错误: SenseVoiceSmall模型不支持时间戳功能。请取消勾选'包含时间戳'选项后重试。"
                            elif not self.get_model_capabilities(model_name)["supports_timestamp"]:
                                error_text = f"错误: {model_name}模型不支持时间戳功能。请取消勾选'包含时间戳'选项后重试。"

                        results.append([
                            file.name.split("/")[-1],
                            "失败",
                            error_text,
                            "0s"
                        ])

                # 保存结果到CSV
                df = pd.DataFrame(results, columns=["文件名", "状态", "识别结果", "处理时间"])
                csv_path = current_dir / "batch_results.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8')

                yield f"批量处理完成！共处理 {total_files} 个文件", df, gr.DownloadButton(
                    label="📥 下载结果(CSV)",
                    value=str(csv_path),
                    visible=True
                )

            except Exception as e:
                error_msg = f"批量处理失败: {str(e)}"
                yield error_msg, pd.DataFrame(), gr.DownloadButton(visible=False)

        batch_btn.click(
            batch_process,
            inputs=[file_upload, batch_model, batch_options],
            outputs=[batch_progress, batch_results, download_btn]
        )

    def create_model_management_interface(self):
        """创建模型管理界面"""
        with gr.Column():
            gr.Markdown("## 🗂️ 模型管理")
            gr.Markdown("查看和管理已下载的模型")

            with gr.Row():
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
                    clear_cache_btn = gr.Button("🗑️ 清理缓存", variant="stop")

                with gr.Column(scale=3):
                    model_info = gr.Dataframe(
                        headers=["模型名称", "类型", "大小", "路径"],
                        label="已下载模型"
                    )

                    cache_info = gr.Textbox(
                        label="缓存信息",
                        lines=4
                    )

        def refresh_models():
            try:
                models_dir = current_dir / "models"
                model_data = []
                total_size = 0

                # 扫描modelscope目录
                ms_dir = models_dir / "modelscope"
                if ms_dir.exists():
                    for org_dir in ms_dir.iterdir():
                        if org_dir.is_dir() and not org_dir.name.startswith('.'):
                            for model_dir in org_dir.iterdir():
                                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                                    size_mb = size / (1024 * 1024)
                                    total_size += size

                                    model_data.append([
                                        f"{org_dir.name}/{model_dir.name}",
                                        "ModelScope",
                                        f"{size_mb:.1f} MB",
                                        str(model_dir)
                                    ])

                # 扫描huggingface目录
                hf_dir = models_dir / "huggingface"
                if hf_dir.exists():
                    hub_dir = hf_dir / "hub"
                    if hub_dir.exists():
                        for model_dir in hub_dir.iterdir():
                            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                                size_mb = size / (1024 * 1024)
                                total_size += size

                                model_data.append([
                                    model_dir.name,
                                    "HuggingFace",
                                    f"{size_mb:.1f} MB",
                                    str(model_dir)
                                ])

                total_size_gb = total_size / (1024 * 1024 * 1024)
                cache_summary = f"""
缓存统计:
- 总模型数量: {len(model_data)}
- 总占用空间: {total_size_gb:.2f} GB
- ModelScope缓存: {ms_dir}
- HuggingFace缓存: {hf_dir}
- 已加载模型: {len(LOADED_MODELS)}
                """.strip()

                return pd.DataFrame(model_data), cache_summary

            except Exception as e:
                return pd.DataFrame(), f"获取模型信息失败: {str(e)}"

        def clear_cache():
            try:
                # 清理内存中的模型
                global LOADED_MODELS
                LOADED_MODELS.clear()

                return pd.DataFrame(), "缓存已清理，请刷新查看最新状态"
            except Exception as e:
                return pd.DataFrame(), f"清理缓存失败: {str(e)}"

        refresh_btn.click(
            refresh_models,
            outputs=[model_info, cache_info]
        )

        clear_cache_btn.click(
            clear_cache,
            outputs=[model_info, cache_info]
        )

        # 页面加载时自动刷新
        refresh_models()

    def create_settings_interface(self):
        """创建设置界面"""
        with gr.Column():
            gr.Markdown("## ⚙️ 系统设置")
            gr.Markdown("配置系统参数和环境设置")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 环境变量")

                    hf_endpoint = gr.Textbox(
                        label="HuggingFace镜像源",
                        value=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                        placeholder="https://hf-mirror.com"
                    )

                    device_setting = gr.Radio(
                        choices=["auto", "cpu", "cuda"],
                        value="auto",
                        label="计算设备"
                    )

                    max_workers = gr.Slider(
                        1, 8, value=4, step=1,
                        label="最大并行数"
                    )

                    apply_settings_btn = gr.Button("✅ 应用设置", variant="primary")

                with gr.Column():
                    gr.Markdown("### 系统信息")

                    system_info = gr.Textbox(
                        label="系统状态",
                        lines=8,
                        value=self.get_system_info()
                    )

                    refresh_info_btn = gr.Button("🔄 刷新信息", variant="secondary")

        def apply_settings(hf_endpoint_val, device_val, workers_val):
            try:
                # 更新环境变量
                os.environ["HF_ENDPOINT"] = hf_endpoint_val

                # 这里可以添加更多设置应用逻辑

                return "设置已应用！"
            except Exception as e:
                return f"应用设置失败: {str(e)}"

        apply_settings_btn.click(
            apply_settings,
            inputs=[hf_endpoint, device_setting, max_workers],
            outputs=[system_info]
        )

        refresh_info_btn.click(
            lambda: self.get_system_info(),
            outputs=[system_info]
        )

    def get_system_info(self) -> str:
        """获取系统信息"""
        try:
            import torch
            import platform

            info = []
            info.append(f"操作系统: {platform.system()} {platform.release()}")
            info.append(f"Python版本: {platform.python_version()}")
            info.append(f"PyTorch版本: {torch.__version__}")

            # GPU信息
            if torch.cuda.is_available():
                info.append(f"CUDA可用: 是 (版本: {torch.version.cuda})")
                info.append(f"GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    info.append(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                info.append("CUDA可用: 否")

            # 环境变量
            info.append("\n环境变量:")
            info.append(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
            info.append(f"HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
            info.append(f"MODELSCOPE_CACHE: {os.environ.get('MODELSCOPE_CACHE', '未设置')}")

            # 模型缓存信息
            models_dir = current_dir / "models"
            if models_dir.exists():
                total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
                info.append(f"\n模型缓存大小: {total_size / 1024**3:.2f} GB")

            return "\n".join(info)

        except Exception as e:
            return f"获取系统信息失败: {str(e)}"

    def create_interface(self):
        """创建主界面"""
        with gr.Blocks(
                title="FunASR WebUI - 智能语音识别平台",
                theme=gr.themes.Soft(),
                css="""
            /* 响应式容器设置 */
            .gradio-container {
                max-width: 95vw !important;
                min-width: 800px !important;
                width: 100% !important;
                margin: 0 auto !important;
                padding: 10px !important;
            }

            /* 主标题样式 */
            .main-header {
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }

            /* 响应式标签页 */
            .gradio-tabs {
                width: 100% !important;
            }

            /* 输入输出区域自适应 */
            .gradio-row {
                width: 100% !important;
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 15px !important;
            }

            .gradio-column {
                flex: 1 !important;
                min-width: 300px !important;
            }

            /* 音频组件优化 */
            .gradio-audio {
                width: 100% !important;
                max-width: none !important;
            }

            /* 按钮样式优化 */
            .gradio-button {
                width: 100% !important;
                min-height: 45px !important;
                font-size: 16px !important;
                margin: 5px 0 !important;
            }

            /* 文本框自适应 */
            .gradio-textbox {
                width: 100% !important;
            }

            /* 下拉框优化 */
            .gradio-dropdown {
                width: 100% !important;
            }

            /* 大屏幕优化 */
            @media (min-width: 1200px) {
                .gradio-container {
                    max-width: 1400px !important;
                }
                .gradio-column {
                    min-width: 400px !important;
                }
            }

            /* 中等屏幕优化 */
            @media (max-width: 1024px) {
                .gradio-container {
                    max-width: 95vw !important;
                    padding: 8px !important;
                }
                .gradio-column {
                    min-width: 280px !important;
                }
                .main-header {
                    padding: 15px;
                }
            }

            /* 小屏幕优化 */
            @media (max-width: 768px) {
                .gradio-container {
                    min-width: 100% !important;
                    padding: 5px !important;
                }
                .gradio-row {
                    flex-direction: column !important;
                }
                .gradio-column {
                    min-width: 100% !important;
                    width: 100% !important;
                }
                .main-header h1 {
                    font-size: 1.8em !important;
                }
                .main-header p {
                    font-size: 0.9em !important;
                }
            }

            /* 超大屏幕优化 */
            @media (min-width: 1600px) {
                .gradio-container {
                    max-width: 1600px !important;
                }
                .gradio-column {
                    min-width: 500px !important;
                }
            }

            /* 高度自适应 */
            .gradio-interface {
                min-height: 100vh !important;
            }

            /* 表格响应式 */
            .gradio-dataframe {
                width: 100% !important;
                overflow-x: auto !important;
            }

            /* JSON显示优化 */
            .gradio-json {
                width: 100% !important;
                max-height: 400px !important;
                overflow-y: auto !important;
            }

            /* 文件上传组件优化 */
            .gradio-file {
                width: 100% !important;
            }

            /* 滑块组件优化 */
            .gradio-slider {
                width: 100% !important;
            }

            /* 复选框组优化 */
            .gradio-checkboxgroup, .gradio-checkbox {
                width: 100% !important;
            }

            /* 标签页内容优化 */
            .gradio-tab-nav {
                flex-wrap: wrap !important;
            }

            /* 手机端标签页优化 */
            @media (max-width: 480px) {
                .gradio-tab-nav button {
                    font-size: 12px !important;
                    padding: 8px 12px !important;
                }
            }
            """
        ) as interface:

            gr.HTML("""
            <div class="main-header">
                <h1>🎤 FunASR 语音识别</h1>
                <p>智能语音识别平台 - 支持多语言、多模型、多功能的语音处理</p>
            </div>
            """)

            # 只显示高级功能标签页，其他标签页隐藏
            with gr.Column():
                self.create_advanced_interface()

            # 隐藏的标签页（用于保持API兼容性，但不显示）
            with gr.Tabs(visible=False) as tabs:
                with gr.TabItem("🎤 基础识别", id="basic"):
                    self.create_basic_asr_interface()

                with gr.TabItem("📁 批量处理", id="batch"):
                    self.create_batch_interface()

                with gr.TabItem("🗂️ 模型管理", id="models"):
                    self.create_model_management_interface()

                with gr.TabItem("⚙️ 系统设置", id="settings"):
                    self.create_settings_interface()

            gr.Markdown("""
            ---
            ### 📖 使用说明
            - **高级功能**: 包含VAD、标点恢复、说话人分离、时间戳等功能

            💡 **提示**: 首次使用需要下载模型，请耐心等待
            """)

        return interface

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="FunASR WebUI")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="启用Gradio公共分享")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--auth", nargs=2, metavar=("USERNAME", "PASSWORD"), help="设置登录认证")

    args = parser.parse_args()

    webui = FunASRWebUI()
    interface = webui.create_interface()

    # 设置认证
    auth = tuple(args.auth) if args.auth else None

    print(f"🎤 FunASR WebUI 启动中...")
    print(f"📍 访问地址: http://{args.host}:{args.port}")
    if args.share:
        print(f"🌐 公共分享已启用")
    if auth:
        print(f"🔐 已启用登录认证")

    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_api=False,
        auth=auth,
        favicon_path=None,
        app_kwargs={"title": "FunASR WebUI"}
    )

if __name__ == "__main__":
    main()
