import requests
import json
import re
import os
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET


class ChineseFormatter:
    """中文会议逐字稿书面化处理"""

    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234",
                 model_name: str = "openai/gpt-oss-20b",
                 prompt_xml_path: str = None,
                 seed: int = 42):
        """
        初始化中文格式化器

        Args:
            lm_studio_url: LM Studio 服务地址
            model_name: 使用的模型名称
            prompt_xml_path: 提示词XML文件路径
            seed: 随机种子，用于提高输出稳定性。
                  相同输入下，固定seed会产生一致输出，适合规范化任务。
                  设为None则完全随机。默认42。
        """
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.seed = seed
        self.api_endpoint = f"{lm_studio_url}/v1/chat/completions"

        # 从XML文件加载提示词
        if prompt_xml_path is None:
            # 默认路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_xml_path = os.path.join(current_dir, "formatted_prompt_templates", "chinese_formatter_prompt.xml")

        self.processing_prompt = self._load_prompt_from_xml(prompt_xml_path)

        # 模型参数配置（优化为精简书面化输出）- LM Studio OpenAI兼容格式
        self.model_options = {
            "seed": seed,  # 随机种子，固定后相同输入产生一致输出，提高稳定性
            "num_ctx": 8192,  # 上下文窗口大小
            "num_predict": 4096,  # 提高最大输出，确保足够篇幅
            "num_batch": 1024,  # 默认即可，或设为 1024 提升吞吐
            "temperature": 0.9,  # 降低温度，使输出更稳定
            "top_p": 0.8,  # 提高top-p，增加内容多样性
            "top_k": 40,  # 降低top-k，更聚焦
            "repeat_penalty": 1.08,  # 降低重复惩罚，避免过度精简
            "presence_penalty": 0,  # 提高存在惩罚
            "frequency_penalty": 0,  # 提高频率惩罚
            "rope_frequency_base": 1000000,  # Qwen 长文本适配
            "stop": ["============", "End of", "【结束】"]  # 移除\n\n\n避免说话人段落间误触发
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
                f"请确保 formatted_prompt_templates/chinese_formatter_prompt.xml 文件存在"
            )
        except Exception as e:
            raise Exception(f"加载提示词模板出错: {e}")

    def is_english_sentence(self, text: str) -> bool:
        """
        判断文本是否主要为英文内容

        Args:
            text: 待检测文本

        Returns:
            True如果主要是英文，False如果包含中文
        """
        # 去除说话人标签
        text = re.sub(r'【说话人:\d+】', '', text).strip()

        if not text:
            return False

        # 检查是否包含中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)

        # 如果中文字符占比大于5%，认为是中英文混合或中文
        if total_chars > 0 and chinese_chars / total_chars > 0.05:
            return False

        # 检查英文字母和常见英文单词占比
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        if total_chars > 0 and english_chars / total_chars > 0.3:
            return True

        return False

    def remove_english_sentences(self, text: str) -> str:
        """
        移除主要为英文的句子行，保留说话人标签和中文内容

        Args:
            text: 原始文本

        Returns:
            移除英文句子后的文本
        """
        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            # 提取说话人标签和内容
            match = re.match(r'(【说话人:\d+】)\s*(.*)', line)
            if match:
                speaker_tag = match.group(1)
                content = match.group(2).strip()

                # 检查内容是否主要为英文
                if self.is_english_sentence(content):
                    continue  # 跳过英文行

                # 保留中文或中英文混合行
                filtered_lines.append(line)
            else:
                # 保留非说话人标签行（如空行、标题等）
                filtered_lines.append(line)

        result = '\n'.join(filtered_lines)
        return result

    def split_text(self, text: str, max_chars: int = 1000) -> List[str]:
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

    def remove_duplicates(self, text: str) -> str:
        """
        使用代码实现去重，删除重复的句子和段落
        防止模型幻觉，确保内容简洁

        Args:
            text: 待去重的文本

        Returns:
            去重后的文本
        """
        # 按句子分割（只在句子结束符号处分割，保留换行）
        sentence_ends = []
        for i, char in enumerate(text):
            if char in '。！？；':
                sentence_ends.append(i)

        # 提取句子（保留换行符）
        sentence_list = []
        last_end = -1
        for end in sentence_ends:
            sentence = text[last_end + 1:end + 1].strip()
            if sentence:
                sentence_list.append(sentence)
            last_end = end

        # 处理剩余文本（如果没有句子结束符）
        if last_end < len(text) - 1:
            remaining = text[last_end + 1:].strip()
            if remaining:
                sentence_list.append(remaining)

        # 去重逻辑：使用相似度检测
        unique_sentences = []
        similarity_threshold = 0.80  # 相似度阈值（降低以检测更多重复）

        for sentence in sentence_list:
            is_duplicate = False

            # 与已保留的句子进行比较
            for kept_sentence in unique_sentences:
                # 移除标点符号和空格进行比较
                s1_clean = re.sub(r'[。！？；，、\s]', '', sentence)
                s2_clean = re.sub(r'[。！？；，、\s]', '', kept_sentence)

                if not s1_clean or not s2_clean:
                    continue

                similarity = SequenceMatcher(None, s1_clean, s2_clean).ratio()

                # 如果相似度超过阈值，认为是重复
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    # 保留较长的句子（通常更完整）
                    if len(sentence) > len(kept_sentence):
                        unique_sentences.remove(kept_sentence)
                        unique_sentences.append(sentence)
                    break

            if not is_duplicate:
                unique_sentences.append(sentence)

        # 合并去重后的句子
        dedup_text = ''.join(unique_sentences)

        # 段落级别去重：删除重复的说话人段落
        if '[' in dedup_text and ']：' in dedup_text:
            paragraphs = dedup_text.split('\n\n')
            unique_paragraphs = []

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # 提取说话人标识
                speaker_match = re.match(r'\[[^\]]+\]：', para)

                is_duplicate_para = False
                for kept_para in unique_paragraphs:
                    kept_match = re.match(r'\[[^\]]+\]：', kept_para)

                    # 同一个说话人，检查内容相似度
                    if speaker_match and kept_match:
                        if speaker_match.group(0) == kept_match.group(0):
                            # 移除说话人标识后比较内容
                            content1 = para[len(speaker_match.group(0)):].strip()
                            content2 = kept_para[len(kept_match.group(0)):].strip()

                            if content1 and content2:
                                # 移除标点符号比较
                                c1_clean = re.sub(r'[。！？；，、\s]', '', content1)
                                c2_clean = re.sub(r'[。！？；，、\s]', '', content2)

                                similarity = SequenceMatcher(None, c1_clean, c2_clean).ratio()
                                if similarity >= similarity_threshold:
                                    is_duplicate_para = True
                                    # 保留较长的段落
                                    if len(para) > len(kept_para):
                                        unique_paragraphs.remove(kept_para)
                                        unique_paragraphs.append(para)
                                    break

                if not is_duplicate_para:
                    unique_paragraphs.append(para)

            dedup_text = '\n\n'.join(unique_paragraphs)

        return dedup_text

    def format_speaker_paragraphs(self, text: str) -> str:
        """
        格式化说话人段落：确保每个说话人的标识后跟完整段落，并换行

        Args:
            text: 待格式化的文本

        Returns:
            格式化后的文本
        """
        import re

        # 在每个【说话人】前换行（除了第一个）
        # 使用正则表达式匹配说话人标识
        pattern = r'【([^】]+)】'

        def replacer(match):
            # 如果匹配的不是在开头，就在前面加换行
            matched_text = match.group(0)
            if match.start() > 0:
                return '\n' + matched_text
            return matched_text

        # 替换所有说话人标识，确保每个都在新行开始
        formatted = re.sub(pattern, replacer, text)

        # 去除开头的多余换行
        formatted = formatted.lstrip('\n')

        return formatted

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
        调用本地LM Studio模型（OpenAI兼容API），支持流式输出和重试机制

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

        # 针对重复的重试计数（独立于网络错误重试）
        repetition_retries = 0
        max_repetition_retries = 2  # 最多重试2次（总共3次机会）

        for attempt in range(max_retries + 1):
            try:
                if use_stream:
                    # 使用流式输出，实时监控生成进度
                    response_text, needs_retry = self._stream_response(payload, attempt)

                    # 如果检测到重复，进行重试
                    if needs_retry and repetition_retries < max_repetition_retries:
                        repetition_retries += 1
                        print(f"  🔄 因内容重复进行第{repetition_retries}次重新生成...")
                        # 稍微调整temperature以增加随机性
                        adjusted_payload = payload.copy()
                        adjusted_payload["temperature"] = self.model_options["temperature"] + (0.1 * repetition_retries)
                        continue  # 重新调用

                    # 如果有结果或已达到最大重试次数，返回结果
                    if response_text:
                        return response_text
                    elif attempt < max_retries or repetition_retries > 0:
                        # 如果还有重试机会（包括重复重试），继续
                        if repetition_retries >= max_repetition_retries:
                            print(f"  ⚠️ 已达到最大重复重试次数，放弃该片段")
                            return ""
                        continue
                    else:
                        return ""
                else:
                    # 非流式模式
                    response = requests.post(
                        self.api_endpoint,
                        json=payload,
                        timeout=180  # 3分钟超时
                    )
                    response.raise_for_status()

                    result = response.json()
                    response_text = result['choices'][0]['message']['content'].strip()

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

    def _stream_response(self, payload: dict, attempt: int) -> tuple[str, bool]:
        """
        流式响应处理（LM Studio OpenAI兼容格式），智能检测重复

        Args:
            payload: 请求payload
            attempt: 当前尝试次数

        Returns:
            (模型生成的文本, 是否因重复而需要重试)
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
                        # OpenAI兼容格式：line以"data: "开头
                        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 移除 "data: " 前缀

                            # 检查是否结束（[DONE]）
                            if data_str.strip() == '[DONE]':
                                print("] ", end="", flush=True)
                                break

                            data = json.loads(data_str)

                            # OpenAI格式：choices[0].delta.content
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    response_text += content
                                    last_output_time = time.time()
                                    no_output_count = 0

                                    # 每1000个字符显示一个点
                                    if len(response_text) % 1000 < 50:
                                        print(".", end="", flush=True)

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
                                print(f"\n  🔄 检测到内容重复，标记需重新生成")
                                print(f"  📊 已生成 {len(response_text)} 字符")
                                # 返回空文本和True表示需要重试
                                return ("", True)
                            last_check_length = len(response_text)

                    except json.JSONDecodeError:
                        continue

            # 清理输出
            response_text = response_text.strip()
            if not response_text and attempt < 2:
                print(f"\n  ⚠️ 第{attempt + 1}次流式请求返回空结果，重试中...")
                return ("", False)
            elif not response_text:
                print(f"\n  ❌ 多次重试仍返回空结果")
                return ("", False)

            return (response_text, False)

        except requests.exceptions.Timeout:
            print(f"\n  ⏱️ 流式请求超时")
            return ("", False)
        except Exception as e:
            print(f"\n  ❌ 流式处理出错: {e}")
            return ("", False)

    def process_transcript(self, transcript: str) -> str:
        """
        处理会议逐字稿：去口语化

        Args:
            transcript: 原始会议逐字稿

        Returns:
            处理后的会议纪要
        """
        # 步骤0: 英文预处理
        print(f"\n{'=' * 60}")
        print("步骤0: 英文预处理")
        print(f"{'=' * 60}")

        original_length = len(transcript)
        preprocessed = self.remove_english_sentences(transcript)
        removed_length = original_length - len(preprocessed)

        if removed_length > 0:
            print(f"✓ 已移除 {removed_length} 字符的英文内容")
            print(f"  原始长度: {original_length} 字符")
            print(f"  预处理后: {len(preprocessed)} 字符")

            # 保存预处理副本用于核验（保存到processed目录）
            processed_dir = "processed/remove_english"
            os.makedirs(processed_dir, exist_ok=True)
            preprocessed_filename = os.path.join(processed_dir, self._generate_timestamped_filename("preprocessed_no_english.txt"))
            try:
                with open(preprocessed_filename, 'w', encoding='utf-8') as f:
                    f.write(preprocessed)
                print(f"  ✓ 预处理副本已保存: {preprocessed_filename}")
            except Exception as e:
                print(f"  ⚠ 保存预处理副本失败: {e}")
        else:
            print("未检测到英文句子，跳过预处理")

        # 使用预处理后的文本进行后续处理
        transcript = preprocessed

        print(f"\n{'=' * 60}")
        print("步骤1: 精简书面化处理")
        print(f"{'=' * 60}")

        # 分割文本（使用1000字符片段，方便精简）
        chunks = self.split_text(transcript, max_chars=1000)
        print(f"文本已分割成 {len(chunks)} 个片段 (每段约1000字符)")
        print(f"预期输出篇幅: {int(len(transcript) * 0.7)}-{int(len(transcript) * 0.8)} 字符 (原文70%-80%)")

        processed_chunks = []
        total_output_length = 0
        dedup_count = 0

        for i, chunk in enumerate(chunks, 1):
            chunk_length = len(chunk)
            target_length_min = int(chunk_length * 0.7)
            target_length_max = int(chunk_length * 0.8)

            # 单独输出分段信息，便于前端解析
            print(f"[{i}/{len(chunks)}] 处理中... (输入: {chunk_length} 字符, 目标: {target_length_min}-{target_length_max} 字符)")

            # 构建提示词，包含长度信息
            prompt = self.processing_prompt.format(
                text=chunk,
                text_length=chunk_length,
                target_length=target_length_min
            )

            # 调用模型
            result = self.call_ollama(prompt)

            if result:
                # 先去重，再检查长度
                before_dedup = result
                result = self.remove_duplicates(result)

                # 记录是否进行了去重
                if len(result) < len(before_dedup):
                    dedup_count += 1
                    removed = len(before_dedup) - len(result)
                    print(f"\n  🔄 去重: 删除 {removed} 字符", end=" ")

                # 检查输出长度
                result_ratio = len(result) / chunk_length * 100

                # 如果输出过短（<60%），可能信息丢失，尝试重新生成
                if result_ratio < 60:
                    print(f"\n  ⚠️ 输出过短 ({len(result)} 字符, {result_ratio:.1f}%)，可能信息丢失，重新生成...")
                    result = self.call_ollama(prompt)
                    if result:
                        result = self.remove_duplicates(result)
                    result_ratio = len(result) / chunk_length * 100 if result else 0

                # 格式化说话人段落（确保每个说话人占一行）
                result = self.format_speaker_paragraphs(result)

                processed_chunks.append(result)
                total_output_length += len(result)

                if result:
                    print(f"✓ 输出: {len(result)} 字符 ({result_ratio:.1f}%)")

                    # 评价比例
                    if 70 <= result_ratio <= 80:
                        print(f"    ✓ 理想比例")
                    elif result_ratio < 60:
                        print(f"    ⚠️ 警告: 仍未达到目标比例 (目标: 70%-80%)")
                    elif result_ratio > 100:
                        print(f"    ⚠️ 警告: 输出偏多 (已去重)")
                    else:
                        print(f"    ✓ 可接受比例")
                else:
                    print(f"✗ 重试失败，使用原文")
                    processed_chunks.pop()
                    processed_chunks.append(chunk)
                    total_output_length += len(chunk)
            else:
                print(f"✗ 处理失败")
                processed_chunks.append(chunk)  # 失败时使用原文
                total_output_length += len(chunk)

        # 合并所有处理后的片段
        processed_text = "\n\n".join(processed_chunks)

        # 输出统计
        overall_ratio = total_output_length / len(transcript) * 100
        print(f"\n{'=' * 60}")
        print(f"精简书面化完成统计:")
        print(f"  原文总长: {len(transcript)} 字符")
        print(f"  输出总长: {total_output_length} 字符 ({overall_ratio:.1f}%)")
        print(f"  目标比例: 70%-80%")
        print(f"  去重次数: {dedup_count} 个segments")

        if overall_ratio < 60:
            print(f"  ⚠️ 注意: 输出偏少，可能信息丢失")
        elif overall_ratio > 100:
            print(f"  ⚠️ 注意: 输出偏多，不够精简")
        elif 70 <= overall_ratio <= 80:
            print(f"  ✓ 达到理想比例")
        else:
            print(f"  ✓ 基本达标")
        print(f"{'=' * 60}")

        return processed_text

    def merge_consecutive_turns(self, text: str) -> str:
        """
        修正后的逻辑：仅合并连续（相邻）的相同说话人。
        保留对话的原始先后顺序。
        """
        lines = text.split('\n')
        if not lines:
            return ""

        merged_turns = []
        current_speaker = None
        current_content = ""

        # 匹配模式：【说话人:XX】内容
        pattern = re.compile(r'^【([^】]+)】(.*)')

        for line in lines:
            line = line.strip()
            if not line: continue

            match = pattern.match(line)
            if match:
                speaker_name, content = match.groups()

                # 如果当前行说话人与上一行相同，则合并内容（不加换行）
                if speaker_name == current_speaker:
                    current_content += content
                else:
                    # 说话人变了，先保存上一轮的完整对话
                    if current_speaker is not None:
                        merged_turns.append(f"【{current_speaker}】{current_content}")
                    # 开启新的一轮
                    current_speaker = speaker_name
                    current_content = content
            else:
                # 如果某行没标签，视为上一行的延续
                current_content += line

        # 别忘了添加最后一段
        if current_speaker is not None:
            merged_turns.append(f"【{current_speaker}】{current_content}")

        return '\n'.join(merged_turns)

    def _generate_timestamped_filename(self, base_name: str) -> str:
        """
        生成带时间戳的文件名

        Args:
            base_name: 基础文件名（如 "processed_chinese.txt"）

        Returns:
            带时间戳的文件名（如 "processed_chinese_20251225_143020.txt"）
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


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    formatter = ChineseFormatter()

    # 读取输入文件
    print("读取会议逐字稿...")
    try:
        with open("input_scripts/meeting_transcript.txt", "r", encoding="utf-8") as f:
            transcript = f.read()

        print(f"原始文本长度: {len(transcript)} 字符")

        # 处理文本
        processed_chinese = formatter.process_transcript(transcript)

        # 后处理合并说话人
        processed_chinese = formatter.merge_consecutive_turns(processed_chinese)

        # 生成带时间戳的文件名
        chinese_filename = formatter._generate_timestamped_filename("processed_chinese.txt")
        chinese_filepath = f"processed/{chinese_filename}"

        # 确保processed文件夹存在
        import os
        os.makedirs("processed", exist_ok=True)

        # 保存结果
        with open(chinese_filepath, "w", encoding="utf-8") as f:
            f.write(processed_chinese)

        # 输出统计信息
        print(f"\n{'=' * 60}")
        print("中文书面化完成！统计信息：")
        print(f"{'=' * 60}")
        print(f"原始文本: {len(transcript)} 字符")
        print(f"处理后中文: {len(processed_chinese)} 字符")
        print(f"\n输出文件已保存到:")
        print(f"  {chinese_filepath}")
        print(f"\n提示: 可以运行 python english_translator.py 来进行英文翻译")

    except FileNotFoundError:
        print("错误: 找不到 input_scripts/meeting_transcript.txt 文件")
        print("\n请确保 input_scripts/meeting_transcript.txt 文件存在")
        print("或者使用以下代码直接处理文本：\n")
        print('transcript = """[说话人1]：那个...我觉得...呃...这个项目..."""')
        print("results = formatter.process_transcript(transcript)")

    except Exception as e:
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()
