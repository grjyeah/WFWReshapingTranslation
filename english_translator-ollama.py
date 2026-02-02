import requests
import json
import re
import os
import glob
from typing import List
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET


class EnglishTranslator:
    """中文会议纪要翻译成英文"""

    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234",
                 model_name: str = "qwen2.5-7b-instruct-1m",
                 prompt_xml_path: str = None):
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.api_endpoint = f"{lm_studio_url}/v1/chat/completions"

        # 从XML文件加载提示词
        if prompt_xml_path is None:
            # 默认路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_xml_path = os.path.join(current_dir, "formatted_prompt_templates", "english_translator_prompt.xml")

        self.translation_prompt = self._load_prompt_from_xml(prompt_xml_path)

        # 模型参数配置（优化为精简书面化输出）
        self.model_options = {
            "num_ctx": 8192,  # 上下文窗口大小
            "num_predict": 8192,  # 限制最大输出，防止过度冗长
            "num_batch": 1024,  # 默认即可，或设为 1024 提升吞吐
            "temperature": 0.5,  # 降低温度，使输出更简洁规范
            "top_p": 0.85,  # 降低top-p，减少发散
            "top_k": 30,  # 降低top-k，更聚焦
            "repeat_penalty": 1.15,  # 提高重复惩罚，避免啰嗦
            "presence_penalty": 0.2,  # 提高存在惩罚
            "frequency_penalty": 0.2,  # 提高频率惩罚
            "rope_frequency_base": 1000000,  # Qwen 长文本适配
            "stop": ["============", "End of", "【结束】"]  # 添加停止词
        }

    def _load_prompt_from_xml(self, xml_path: str) -> str:
        """
        从XML文件加载提示词模板

        Args:
            xml_path: XML文件路径

        Returns:
            提示词字符串
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 将XML转换为字符串，保留CDATA和结构
            import xml.dom.minidom as minidom

            # 使用minidom来保持格式
            dom = minidom.parse(xml_path)
            prompt_str = dom.documentElement.toxml()

            # 只保留instructions部分的内容
            instructions_elem = root.find('.//instructions')
            if instructions_elem is not None:
                # 将XML元素转回字符串格式
                prompt_str = ET.tostring(instructions_elem, encoding='unicode', method='xml')
                # 去除XML声明
                prompt_str = prompt_str.replace('<?xml version="1.0" encoding="UTF-8"?>', '').strip()
                return prompt_str
            else:
                raise ValueError("XML文件中未找到instructions元素")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"找不到提示词模板文件: {xml_path}\n"
                f"请确保 formatted_prompt_templates/english_translator_prompt.xml 文件存在"
            )
        except Exception as e:
            raise Exception(f"加载提示词模板出错: {e}")

    def split_text(self, text: str, max_chars: int = 1500) -> List[str]:
        """
        将长文本按段落智能分割，确保在句子边界切分

        Args:
            text: 原始文本
            max_chars: 每段最大字符数

        Returns:
            分割后的文本片段列表
        """
        # 按说话人和句子分割，保持语义完整性
        speaker_pattern = r'\[([^\]]+)\]：'
        sentence_endings = r'[。！？；…\n]+'

        # 先提取所有说话人段落
        speaker_blocks = []
        current_speaker = None
        current_content = []

        lines = text.split('\n')
        for line in lines:
            speaker_match = re.match(speaker_pattern, line)
            if speaker_match:
                # 保存前一个说话人的内容
                if current_speaker and current_content:
                    content = ''.join(current_content).strip()
                    if content:
                        speaker_blocks.append(f"[{current_speaker}]：{content}")
                # 开始新的说话人
                current_speaker = speaker_match.group(1)
                current_content = [line[len(speaker_match.group(0)):]]
            elif current_speaker:
                current_content.append(line)

        # 保存最后一个说话人
        if current_speaker and current_content:
            content = ''.join(current_content).strip()
            if content:
                speaker_blocks.append(f"[{current_speaker}]：{content}")

        # 如果没有检测到说话人格式，按句子分割
        if not speaker_blocks:
            sentences = re.split(f'({sentence_endings})', text)
            chunks = []
            current_chunk = ""

            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                # 添加标点符号
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]

                if len(current_chunk) + len(sentence) <= max_chars:
                    current_chunk += sentence
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            return chunks if chunks else [text]

        # 按说话人块组合成chunks
        chunks = []
        current_chunk = ""

        for block in speaker_blocks:
            if len(current_chunk) + len(block) <= max_chars:
                current_chunk += ("\n\n" if current_chunk else "") + block
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = block

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def detect_repetition(self, text: str, window_size: int = 100) -> bool:
        """
        检测文本是否出现重复循环

        Args:
            text: 待检测的文本
            window_size: 检测窗口大小

        Returns:
            True if repetition detected
        """
        if len(text) < window_size * 2:
            return False

        # 检查最后window_size个字符是否在前面出现过
        tail = text[-window_size:]

        # 在最后500个字符中查找重复
        search_range = text[-500:-window_size] if len(text) > 500 else text[:-window_size]

        if tail in search_range:
            return True

        return False

    def call_ollama(self, prompt: str, max_retries: int = 2, use_stream: bool = True) -> str:
        """
        调用本地Ollama模型，支持流式输出和重试机制

        Args:
            prompt: 输入提示词
            max_retries: 最大重试次数
            use_stream: 是否使用流式输出（默认True，防止卡死）

        Returns:
            模型生成的文本
        """
        # 使用OpenAI兼容格式（LM Studio）
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": use_stream,
            **self.model_options
        }

        for attempt in range(max_retries + 1):
            try:
                if use_stream:
                    # 使用流式输出，实时监控生成进度
                    return self._stream_response(payload, attempt)
                else:
                    # 非流式模式
                    response = requests.post(
                        self.api_endpoint,
                        json=payload,
                        timeout=180  # 3分钟超时
                    )
                    response.raise_for_status()

                    result = response.json()
                    response_text = result.get('response', '').strip()

                    if response_text:
                        return response_text
                    elif attempt < max_retries:
                        print(f"  第{attempt + 1}次尝试返回空结果，重试中...")
                    else:
                        return ""

            except requests.exceptions.Timeout as err:
                if attempt < max_retries:
                    print(f"\n  ⏱️ 第{attempt + 1}次尝试超时（3分钟），重试中...")
                else:
                    print(f"\n  ❌ API调用超时: {err}")
                    return ""

            except requests.exceptions.RequestException as err:
                if attempt < max_retries:
                    print(f"\n  🔄 第{attempt + 1}次尝试出错: {err}，重试中...")
                else:
                    print(f"\n  ❌ API调用错误: {err}")
                    return ""

        return ""

    def _stream_response(self, payload: dict, attempt: int) -> str:
        """
        流式响应处理，智能检测重复并停止

        Args:
            payload: 请求payload
            attempt: 当前尝试次数

        Returns:
            模型生成的文本
        """
        import time

        response_text = ""
        last_output_time = time.time()
        no_output_count = 0
        max_no_output_intervals = 3  # 最多容忍3次30秒无输出
        check_interval = 30  # 每30秒检查一次是否有新输出

        # 重复检测相关
        repetition_check_interval = 500  # 每生成500字符检查一次重复
        last_check_length = 0

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                stream=True,
                timeout=180  # 连接超时3分钟
            )
            response.raise_for_status()

            print(" [生成中", end="", flush=True)

            for line in response.iter_lines():
                if line:
                    # 跳过空行
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue

                    # 移除 "data: " 前缀（OpenAI格式）
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]

                    # 跳过 [DONE] 标记（OpenAI格式）
                    if line_str == '[DONE]':
                        break

                    try:
                        data = json.loads(line_str)

                        # 支持OpenAI格式 (choices[0].delta.content)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                if content:  # content可能为None
                                    response_text += content
                                    last_output_time = time.time()
                                    no_output_count = 0

                                    # 每1000个字符显示一个点
                                    if len(response_text) % 1000 < 50:
                                        print(".", end="", flush=True)

                            # 检查是否完成（OpenAI格式使用finish_reason）
                            finish_reason = data['choices'][0].get('finish_reason')
                            if finish_reason is not None:
                                print("] ", end="", flush=True)
                                break

                        # 支持Ollama格式 (response字段)
                        elif 'response' in data:
                            response_text += data['response']
                            last_output_time = time.time()
                            no_output_count = 0

                            # 每1000个字符显示一个点
                            if len(response_text) % 1000 < 50:
                                print(".", end="", flush=True)

                        # 检查是否完成（Ollama格式使用done字段）
                        if data.get('done', False):
                            print("] ", end="", flush=True)
                            break

                        # 定期检查是否有新输出
                        current_time = time.time()
                        if current_time - last_output_time > check_interval:
                            no_output_count += 1
                            print(f"[{no_output_count}×无输出]", end="", flush=True)

                            if no_output_count >= max_no_output_intervals:
                                print(f"\n  ⚠️ 检测到模型停滞（{max_no_output_intervals * check_interval}秒无输出）")
                                print(f"  📊 已生成 {len(response_text)} 字符，强制停止")
                                break

                            last_output_time = current_time

                        # 智能重复检测：每生成一定字符后检查
                        if len(response_text) - last_check_length >= repetition_check_interval:
                            if self.detect_repetition(response_text):
                                print(f"\n  🔄 检测到内容重复，自动停止")
                                print(f"  📊 已生成 {len(response_text)} 字符")
                                break
                            last_check_length = len(response_text)

                    except json.JSONDecodeError:
                        continue

            # 清理输出
            response_text = response_text.strip()
            if not response_text and attempt < 2:
                print(f"\n  ⚠️ 第{attempt + 1}次流式请求返回空结果，重试中...")
                return ""
            elif not response_text:
                print(f"\n  ❌ 多次重试仍返回空结果")
                return ""

            return response_text

        except requests.exceptions.Timeout:
            print(f"\n  ⏱️ 流式请求超时")
            return ""
        except Exception as e:
            print(f"\n  ❌ 流式处理出错: {e}")
            return ""

    def translate_to_english(self, chinese_text: str) -> str:
        """
        将中文会议纪要翻译成英文

        Args:
            chinese_text: 中文会议纪要

        Returns:
            英文翻译
        """
        print(f"\n{'=' * 60}")
        print("步骤2: 翻译成英文")
        print(f"{'=' * 60}")

        # 分割文本（使用较小的片段以保持翻译质量）
        chunks = self.split_text(chinese_text, max_chars=1500)
        print(f"文本已分割成 {len(chunks)} 个片段 (每段约1500字符)")

        translated_chunks = []
        total_output_length = 0

        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}/{len(chunks)}] 翻译中... (输入: {len(chunk)} 字符)", end=" ")

            # 构建翻译提示词
            prompt = self.translation_prompt.format(text=chunk)

            # 调用模型
            result = self.call_ollama(prompt)

            if result:
                translated_chunks.append(result)
                total_output_length += len(result)
                ratio = len(result) / len(chunk) * 100
                print(f"[OK] 输出: {len(result)} 字符 ({ratio:.1f}%)")
            else:
                print(f"[FAIL] 翻译失败")
                # 翻译失败时保留原文（虽然不理想，但不会丢失内容）
                translated_chunks.append(chunk)
                total_output_length += len(chunk)

        # 合并所有翻译片段
        translated_text = "\n\n".join(translated_chunks)

        # 输出统计
        print(f"\n翻译统计:")
        print(f"  中文输入: {len(chinese_text)} 字符")
        print(f"  英文输出: {total_output_length} 字符")

        return translated_text

    def _generate_timestamped_filename(self, base_name: str) -> str:
        """
        生成带时间戳的文件名

        Args:
            base_name: 基础文件名（如 "english_translation.txt"）

        Returns:
            带时间戳的文件名（如 "english_translation_20251225_143020.txt"）
        """
        from datetime import datetime

        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 分离文件名和扩展名
        if '.' in base_name:
            name, ext = base_name.rsplit('.', 1)
            return f"{name}_{timestamp}.{ext}"
        else:
            return f"{base_name}_{timestamp}"

    def find_latest_processed_chinese(self, input_dir: str = "processed") -> str:
        """
        查找最新的 processed_chinese_<timestamp>.txt 文件

        Args:
            input_dir: 输入目录路径

        Returns:
            最新文件的完整路径

        Raises:
            FileNotFoundError: 如果没有找到匹配的文件
        """
        # 确保目录存在
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        # 查找所有 processed_chinese_*.txt 文件
        pattern = os.path.join(input_dir, "processed_chinese_*.txt")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"在 {input_dir} 目录中未找到 processed_chinese_<timestamp>.txt 文件\n"
                f"请先运行 python chinese_formatter.py 生成中文书面化文件"
            )

        # 按修改时间排序，获取最新的文件
        latest_file = max(files, key=os.path.getmtime)

        return latest_file


# 使用示例
if __name__ == "__main__":
    # 初始化翻译器
    translator = EnglishTranslator()

    try:
        # 自动查找最新的 processed_chinese 文件
        print("查找最新的 processed_chinese 文件...")
        chinese_filepath = translator.find_latest_processed_chinese("processed")

        print(f"找到文件: {chinese_filepath}")

        # 读取中文文本
        with open(chinese_filepath, "r", encoding="utf-8") as f:
            chinese_text = f.read()

        print(f"中文文本长度: {len(chinese_text)} 字符")

        # 翻译成英文
        english_translation = translator.translate_to_english(chinese_text)

        # 生成带时间戳的文件名
        english_filename = translator._generate_timestamped_filename("english_translation.txt")
        english_filepath = f"processed/{english_filename}"

        # 确保processed文件夹存在
        os.makedirs("processed", exist_ok=True)

        # 保存结果
        with open(english_filepath, "w", encoding="utf-8") as f:
            f.write(english_translation)

        # 输出统计信息
        print(f"\n{'=' * 60}")
        print("英文翻译完成！统计信息：")
        print(f"{'=' * 60}")
        print(f"中文输入: {len(chinese_text)} 字符")
        print(f"英文输出: {len(english_translation)} 字符")
        print(f"\n输出文件已保存到:")
        print(f"  {english_filepath}")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请按以下步骤操作:")
        print("1. 确保 input_scripts/meeting_transcript.txt 文件存在")
        print("2. 运行 python chinese_formatter.py 生成中文书面化文件")
        print("3. 再运行 python english_translator.py 进行英文翻译")

    except Exception as e:
        print(f"\n处理出错: {e}")
        import traceback
        traceback.print_exc()
