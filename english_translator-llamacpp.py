import requests
import json
import re
import os
import glob
from typing import List
from difflib import SequenceMatcher


class EnglishTranslator:
    """中文会议纪要翻译成英文"""

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 model_name: str = "yasserrmd/Qwen2.5-7B-Instruct-1M:latest"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.api_endpoint = f"{ollama_url}/api/generate"

        # 模型参数配置（优化为精简书面化输出）
        self.model_options = {
            "num_ctx": 131072,  # 上下文窗口大小
            "num_predict": 4096,  # 限制最大输出，防止过度冗长
            "temperature": 0.5,  # 降低温度，使输出更简洁规范
            "top_p": 0.85,  # 降低top-p，减少发散
            "top_k": 30,  # 降低top-k，更聚焦
            "repeat_penalty": 1.15,  # 提高重复惩罚，避免啰嗦
            "presence_penalty": 0.2,  # 提高存在惩罚
            "frequency_penalty": 0.2,  # 提高频率惩罚
            "stop": ["\n\n\n", "============", "End of", "【结束】"]  # 添加停止词
        }

        # 翻译提示词模板
        self.translation_prompt = """请将以下中文会议纪要翻译成英文。要求：
1. 保持专业的商务/学术语言风格
2. 保留说话人标识格式：[Speaker Name/Role]:
3. 确保翻译准确、流畅、地道
4. 不要添加任何额外说明

需要翻译的内容：

{text}"""

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
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": use_stream,
            "options": self.model_options
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

    def call_llamacpp(self, prompt: str, max_retries: int = 2, use_stream: bool = False) -> str:
        """
        调用本地 llama-server (OpenAI 兼容 API)，替代 Ollama
        Args:
            prompt: 输入提示词
            max_retries: 最大重试次数
            use_stream: 是否流式输出（当前暂不支持，设为 False）
        Returns:
            模型生成的文本
        """
        # llama-server 的 OpenAI 兼容端点
        api_endpoint = "http://localhost:6008/v1/chat/completions"

        # 从原 model_options 提取参数（映射到 OpenAI 参数）
        temperature = self.model_options.get("temperature", 0.5)
        top_p = self.model_options.get("top_p", 0.85)
        max_tokens = min(self.model_options.get("num_predict", 4096), 4096)  # llama-server 限制

        # 停止词（llama-server 支持 stop 参数）
        stop = self.model_options.get("stop", ["\n\n\n", "============", "End of", "【结束】"])

        payload = {
            "model": "qwen3-30b-a3b",  # 模型名可任意，llama-server 忽略
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 30,  # 降低top-k，更聚焦
            "repeat_penalty": 1.15,  # 提高重复惩罚，避免啰嗦
            "presence_penalty": 0.2,  # 提高存在惩罚
            "frequency_penalty": 0.2,  # 提高频率惩罚
            "max_tokens": max_tokens,
            "stop": stop,
            "stream": False  # 暂不启用流式（简化）
        }

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    api_endpoint,
                    json=payload,
                    timeout=300  # 5分钟超时
                )
                response.raise_for_status()
                result = response.json()

                # 提取生成内容
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"].strip()
                    if text:
                        return text

                if attempt < max_retries:
                    print(f" 第{attempt + 1}次尝试返回空结果，重试中...")
                else:
                    print(f" ❌ 所有重试失败，返回空")
                    return ""

            except requests.exceptions.Timeout as err:
                if attempt < max_retries:
                    print(f"\n ⏱️ 第{attempt + 1}次尝试超时，重试中...")
                else:
                    print(f"\n ❌ API调用超时: {err}")
                    return ""
            except Exception as err:
                if attempt < max_retries:
                    print(f"\n 🔄 第{attempt + 1}次尝试出错: {err}，重试中...")
                else:
                    print(f"\n ❌ API调用错误: {err}")
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
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            response_text += data['response']
                            last_output_time = time.time()
                            no_output_count = 0

                            # 每1000个字符显示一个点
                            if len(response_text) % 1000 < 50:
                                print(".", end="", flush=True)

                        # 检查是否完成
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
            result = self.call_llamacpp(prompt)

            if result:
                translated_chunks.append(result)
                total_output_length += len(result)
                ratio = len(result) / len(chunk) * 100
                print(f"✓ 输出: {len(result)} 字符 ({ratio:.1f}%)")
            else:
                print(f"✗ 翻译失败")
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
    translator = EnglishTranslator(
        ollama_url="http://localhost:11434",
        model_name="yasserrmd/Qwen2.5-7B-Instruct-1M:latest"
    )

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
