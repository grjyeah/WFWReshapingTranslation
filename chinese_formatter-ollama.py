import requests
import json
import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher


class ChineseFormatter:
    """中文会议逐字稿书面化处理"""

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 # model_name: str = "yasserrmd/Qwen2.5-7B-Instruct-1M:latest"):
                 model_name: str = "did100/qwen2.5-32B-Instruct-Q4_K_M:latest"):
                 # model_name: str = "alibayram/Qwen3-30B-A3B-Instruct-2507:latest"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.api_endpoint = f"{ollama_url}/api/generate"

        # 模型参数配置（优化为精简书面化输出）
        self.model_options = {
            # "mirostat": 2,
            # "mirostat_tau": 5.0,  # 中文连贯性最佳区间
            # "mirostat_eta": 0.1,
            "repeat_penalty": 1.15,
            "num_thread": 8,  # GPU offload 后，CPU 只需处理剩余层，8 线程足够
            "num_batch": 512,  # 默认即可，或设为 1024 提升吞吐
            "rope_frequency_base": 1000000,   # Qwen 长文本适配

            "num_ctx": 131072,  # 上下文窗口大小
            "num_predict": 8192,  # 限制最大输出，防止过度冗长
            "temperature": 0.3,  # 降低温度，使输出更简洁规范
            "top_p": 0.85,  # 降低top-p，减少发散
            "top_k": 30,  # 降低top-k，更聚焦
            "repeat_penalty": 1.15,  # 提高重复惩罚，避免啰嗦
            "presence_penalty": 0.2,  # 提高存在惩罚
            "frequency_penalty": 0.2,  # 提高频率惩罚
            "stop": ["\n\n\n", "============", "End of", "【结束】"]  # 添加停止词
        }

        # 提示词模板
        self.processing_prompt = """<instructions>
    <role>
        你是一位专业的语言编辑，擅长将口语化的会议逐字稿转换为正式的书面语文档。
    </role>

    <task>
        <requirement type="core" priority="highest">
            <title>核心要求 - 逐句转换，不做总结</title>
            <item id="1">
                <name>逐句转换</name>
                <rules>
                    <rule>对原文中的每一句话进行书面化改写，不要无中生有</rule>
                    <rule>严禁总结、概括或归纳</rule>
                    <rule>输入文本每句话换一行，但输出需根据说话人标签把同一个人说的话合并到一个段落，不要单句分行</rule>
                    <rule>保留所有说话人的所有发言内容</rule>
                    <rule>每句话开始有对应的说话人标签，例如【说话人:0】、【说话人:1】等</rule>
                    <rule>根据标签区分同一个说话人的内容，如果是同一说话人，把内容合并放到一个段落中，不需要太多换行</rule>
                    <rule condition="empty_content">如果整句话判断为纯口语没有输出，这行不需要输出，不用输出空白的行</rule>
                    <rule condition="empty_content">例如【说话人:X】：(标签后只有换行符)【说话人:X】：(标签后没内容)的，连标签也不需要输出</rule>
                </rules>
            </item>

            <item id="2">
                <name>书面化改写</name>
                <rules>
                    <rule>删除所有口语词："那个"、"然后"、"就是说"、"呃"、"嗯"、"啊"、"呢"、"哇"等</rule>
                    <rule>保留所有实质性内容、数据、观点、讨论细节</rule>
                    <rule>将口语表达改为正式书面语表达</rule>
                    <rule>润色语言，使表达更专业、更规范</rule>
                </rules>
            </item>

            <item id="3" type="prohibition">
                <name>严禁以下行为</name>
                <prohibitions>
                    <prohibition>严禁删除任何发言内容</prohibition>
                    <prohibition>严禁总结概括（如"主要讨论了"、"重点提到"）</prohibition>
                    <prohibition>严禁合并句子或段落</prohibition>
                    <prohibition>严禁提炼要点</prohibition>
                    <prohibition>严禁添加原文中没有的内容</prohibition>
                </prohibitions>
            </item>

            <item id="4">
                <name>输出格式要求</name>
                <requirements>
                    <requirement>保留所有说话人标识</requirement>
                    <requirement>保留原有的对话结构和顺序</requirement>
                    <requirement>每个说话人的发言都要完整保留</requirement>
                    <requirement>语言风格：正式、专业、客观</requirement>
                </requirements>
            </item>

            <item id="5" type="prohibition">
                <name>严禁输出无关内容</name>
                <prohibitions>
                    <prohibition>严禁添加任何标题（如"### 会议记录"、"【逐句书面化改写】"）</prohibition>
                    <prohibition>严禁添加说明性文字（如"以下是..."、"改写如下："）</prohibition>
                    <prohibition>严禁添加前言、后语、总结性文字</prohibition>
                    <prohibition priority="critical">只输出对话本身，从第一个说话人开始，到最后一个说话人结束</prohibition>
                    <prohibition priority="critical">不要有任何其他文字，只要对话</prohibition>
                </prohibitions>
            </item>

            <item id="6">
                <name>格式要求</name>
                <formatting_rules>
                    <rule id="speaker_label">
                        <name>说话人标识格式统一</name>
                        <description>使用【说话人姓名】（书名号），不要用方括号[]</description>
                    </rule>
                    <rule id="paragraph_organization">
                        <name>段落组织</name>
                        <description>同一个说话人的连续发言，合并为一个完整的语义段落</description>
                    </rule>
                    <rule id="line_break">
                        <name>换行规则</name>
                        <description>每个说话人的完整语义段落结束后，必须换行（每个说话人占一行）</description>
                    </rule>
                    <rule id="sentence_continuity">
                        <name>不要多句话换行</name>
                        <description>将同一说话人的相关句子组织成连贯的段落</description>
                    </rule>
                    <rule id="one_speaker_one_paragraph">
                        <name>一个说话人=一个段落</name>
                        <format>【说话人】：完整内容（可包含多个句子）然后换行</format>
                    </rule>
                </formatting_rules>
            </item>

            <item id="7">
                <name>输出示例</name>
                <examples>
                    <example type="correct">
                        <description>正确格式</description>
                        <content>
                            【主持人】：大家好，欢迎参加今天的会议。今天我们主要讨论数据治理的相关工作。
                            
                            【张总】：我们需要建立一个完善的数据治理体系。数据治理非常重要，对企业的数字化转型至关重要。
                            
                            【李经理】：我同意张总的说法。我们的平台能够提供有效的支持。
                        </content>
                    </example>
                    <example type="incorrect">
                        <description>错误格式</description>
                        <errors>
                            <error>【主持人】：大家好，【张总】：我们需要建立...（不要把不同说话人放在一起）</error>
                            <error>【主持人】：大家好。今天我们讨论。相关的工作。（同一说话人的相关句子不要断开）</error>
                        </errors>
                    </example>
                </examples>
                <core_principle>逐句书面化改写，不删不减，保留说话人，一个说话人一个段落！</core_principle>
            </item>
        </requirement>
    </task>

    <input>
        <metadata>
            <original_text_length unit="characters">{text_length}</original_text_length>
        </metadata>
        <content>
            <![CDATA[{text}]]>
        </content>
    </input>

    <output_requirement>
        <target_length unit="characters">
            <value>{target_length}</value>
            <tolerance>±10%</tolerance>
        </target_length>
        <format>严格按照上述示例格式，直接输出逐句书面化改写后的对话</format>
    </output_requirement>
</instructions>"""

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
            "options": self.model_options,
            "num_gpu_layers": 60  # 根据你的GPU显存调整数值
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
                        timeout=180  # 3分钟超时（从10分钟缩短）
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

    def process_transcript(self, transcript: str) -> str:
        """
        处理会议逐字稿：去口语化

        Args:
            transcript: 原始会议逐字稿

        Returns:
            处理后的会议纪要
        """
        print(f"\n{'=' * 60}")
        print("步骤1: 精简书面化处理")
        print(f"{'=' * 60}")

        # 分割文本（使用1000字符片段，方便精简）
        chunks = self.split_text(transcript, max_chars=1000)
        print(f"文本已分割成 {len(chunks)} 个片段 (每段约1000字符)")
        print(f"预期输出篇幅: {int(len(transcript) * 0.8)}-{int(len(transcript) * 0.9)} 字符 (原文80%-90%)")

        processed_chunks = []
        total_output_length = 0
        dedup_count = 0

        for i, chunk in enumerate(chunks, 1):
            chunk_length = len(chunk)
            target_length_min = int(chunk_length * 0.8)
            target_length_max = int(chunk_length * 0.9)

            print(f"\n[{i}/{len(chunks)}] 处理中... (输入: {chunk_length} 字符, 目标: {target_length_min}-{target_length_max} 字符)", end=" ")

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
                    if 80 <= result_ratio <= 90:
                        print(f"    ✓ 理想比例")
                    elif result_ratio < 70:
                        print(f"    ⚠️ 警告: 仍未达到目标比例 (目标: 80%-90%)")
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
        print(f"  目标比例: 80%-90%")
        print(f"  去重次数: {dedup_count} 个segments")

        if overall_ratio < 70:
            print(f"  ⚠️ 注意: 输出偏少，可能信息丢失")
        elif overall_ratio > 100:
            print(f"  ⚠️ 注意: 输出偏多，不够精简")
        elif 80 <= overall_ratio <= 90:
            print(f"  ✓ 达到理想比例")
        else:
            print(f"  ✓ 基本达标")
        print(f"{'=' * 60}")

        return processed_text

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
    formatter = ChineseFormatter(
        ollama_url="http://localhost:11434",
        model_name="yasserrmd/Qwen2.5-7B-Instruct-1M:latest"
    )

    # 读取输入文件
    print("读取会议逐字稿...")
    try:
        with open("input_scripts/meeting_transcript.txt", "r", encoding="utf-8") as f:
            transcript = f.read()

        print(f"原始文本长度: {len(transcript)} 字符")

        # 处理文本
        processed_chinese = formatter.process_transcript(transcript)

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
