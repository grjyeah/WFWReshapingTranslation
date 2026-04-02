"""
LLM 驱动的文本对比分析器
用于分析人工校对与大模型书面化的差异
"""
import requests
import json
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TextComparisonAnalyzer:
    """文本对比分析器（基于本地 LLM）"""

    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234",
                 model_name: str = "qwen2.5-7b-instruct"):
        """
        初始化分析器

        Args:
            lm_studio_url: LM Studio API 地址
            model_name: 使用的模型名称
        """
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.api_endpoint = f"{lm_studio_url}/v1/chat/completions"

    def analyze_corrections(self, formalized_text: str, human_text: str) -> List[Dict]:
        """
        分析校对差异

        Args:
            formalized_text: 大模型书面化文本
            human_text: 人工校对文本

        Returns:
            校对行为列表，每个行为包含：
            - type: 校对类型（删除/添加/修改/合并/分割）
            - original: 原始文本
            - corrected: 修改后文本
            - reason: 修改原因
            - pattern: 识别的模式
        """
        # 构建分析提示词
        prompt = self._build_analysis_prompt(formalized_text, human_text)

        try:
            # 调用 LLM
            response = self._call_llm(prompt)

            # 解析响应
            corrections = self._parse_corrections(response)
            logger.info(f"分析完成，识别到 {len(corrections)} 个校对行为")
            return corrections

        except Exception as e:
            logger.error(f"分析失败: {e}")
            return []

    def _build_analysis_prompt(self, formalized_text: str, human_text: str) -> str:
        """构建分析提示词"""
        prompt = f"""你是一个专业的文本分析专家，专门分析会议纪要校对行为。

请仔细对比以下两段文本，识别出所有的修改行为：

【大模型书面化文本】
{formalized_text}

【人工校对文本】
{human_text}

请按照以下 JSON 格式输出分析结果：

[
  {{
    "type": "修改类型",
    "original": "原始文本片段",
    "corrected": "修改后文本片段",
    "reason": "修改原因",
    "pattern": "识别的模式名称"
  }}
]

修改类型包括：
- 删除：删除了冗余、错误或不必要的内容
- 添加：补充了遗漏的信息
- 修改：修正了错误或优化表达
- 合并：合并了零散的句子
- 分割：将长句拆分为多个短句
- 术语统一：统一了专业术语或人名
- 口语化调整：将书面语调整为更自然的口语
- 语气调整：调整了语气或情感色彩

模式名称示例：
- 去除冗余词
- 统一专业术语
- 修正错别字
- 优化句子结构
- 补充缺失信息
- 简化表达
- 正式化口语
- 时间表达修正
- 数字格式统一
等等

请只输出 JSON 数组，不要包含其他内容。"""
        return prompt

    def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """调用本地 LLM"""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个专业的文本分析专家，擅长识别和分析文本修改模式。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # 降低温度以获得更稳定的输出
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=120,  # 2分钟超时
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            return content

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def _parse_corrections(self, llm_output: str) -> List[Dict]:
        """解析 LLM 输出为结构化数据"""
        try:
            # 尝试直接解析 JSON
            if llm_output.startswith('['):
                corrections = json.loads(llm_output)
                return corrections

            # 如果包含 markdown 代码块，提取 JSON
            if '```json' in llm_output:
                match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_output)
                if match:
                    corrections = json.loads(match.group(1))
                    return corrections

            # 如果包含 ``` 但不是 json，尝试提取
            if '```' in llm_output:
                match = re.search(r'```\s*([\s\S]*?)\s*```', llm_output)
                if match:
                    corrections = json.loads(match.group(1))
                    return corrections

            # 尝试查找 JSON 数组
            match = re.search(r'\[[\s\S]*\]', llm_output)
            if match:
                corrections = json.loads(match.group(0))
                return corrections

            # 解析失败，返回空列表
            logger.warning(f"无法解析 LLM 输出: {llm_output[:200]}")
            return []

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            logger.debug(f"LLM 输出: {llm_output}")
            return []

    def extract_patterns(self, corrections: List[Dict]) -> Dict[str, int]:
        """
        从校对行为中提取模式

        Args:
            corrections: 校对行为列表

        Returns:
            模式及其频率
        """
        patterns = {}

        for correction in corrections:
            pattern_name = correction.get('pattern', '未分类')
            patterns[pattern_name] = patterns.get(pattern_name, 0) + 1

        # 按频率排序
        sorted_patterns = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
        return sorted_patterns

    def generate_correction_summary(self, corrections: List[Dict]) -> str:
        """生成校对摘要"""
        if not corrections:
            return "未识别到校对行为"

        # 统计类型分布
        type_counts = {}
        for c in corrections:
            ct = c.get('type', '未知')
            type_counts[ct] = type_counts.get(ct, 0) + 1

        # 提取模式
        patterns = self.extract_patterns(corrections)

        # 构建摘要
        summary = f"""校对分析摘要

总计修改: {len(corrections)} 处

类型分布:
"""
        for ct, count in type_counts.items():
            summary += f"  - {ct}: {count} 处\n"

        summary += "\n识别到的模式:\n"
        for pattern, count in list(patterns.items())[:10]:  # 显示前 10 个
            summary += f"  - {pattern}: {count} 次\n"

        return summary


class BatchDocumentProcessor:
    """批量文档处理器"""

    def __init__(self, graph_manager, analyzer: TextComparisonAnalyzer):
        """
        初始化批量处理器

        Args:
            graph_manager: JanusGraph 管理器实例
            analyzer: 文本分析器实例
        """
        self.graph_manager = graph_manager
        self.analyzer = analyzer

    def process_document_batch(self, documents: List[Dict]) -> Dict[str, any]:
        """
        批量处理文档

        Args:
            documents: 文档列表，每个文档包含：
                - audio_filename: 音频文件名
                - human_text_filename: 人工校对文本文件名
                - audio_path: 音频文件路径
                - human_text_path: 人工校对文本路径

        Returns:
            处理结果统计
        """
        results = {
            'total': len(documents),
            'success': 0,
            'failed': 0,
            'errors': []
        }

        for doc in documents:
            try:
                self._process_single_document(doc)
                results['success'] += 1
                logger.info(f"✓ 成功处理: {doc['audio_filename']}")

            except Exception as e:
                results['failed'] += 1
                error_msg = f"{doc['audio_filename']}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(f"✗ 处理失败: {error_msg}")

        return results

    def _process_single_document(self, doc: Dict):
        """处理单个文档"""
        # 1. 创建文档节点
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")

        document_id = self.graph_manager.create_document(
            filename=doc['audio_filename'],
            date=date_str,
            audio_path=doc['audio_path']
        )

        # 2. 读取人工校对文本
        with open(doc['human_text_path'], 'r', encoding='utf-8') as f:
            human_text = f.read()

        # 3. 对音频进行书面化（调用现有的处理器）
        # 这里需要集成 chinese_formatter-ollama.py
        from chinese_formatter_ollama import ChineseFormatter

        formatter = ChineseFormatter()
        formalized_text = formatter.process_transcript("")  # 需要传入音频转文字结果

        # 4. 对比分析
        corrections = self.analyzer.analyze_corrections(formalized_text, human_text)

        # 5. 保存到图谱
        # （这里需要解析句子并创建节点，简化示例）
        # ...

        logger.info(f"文档处理完成: {doc['audio_filename']}")


if __name__ == "__main__":
    # 测试分析器
    analyzer = TextComparisonAnalyzer()

    formalized = "【说话人:0】我觉得这个项目应该尽快推进。"
    human = "【张总】我认为这个项目需要加快进度，确保按时完成。"

    corrections = analyzer.analyze_corrections(formalized, human)

    print("\n识别到的校对行为:")
    for i, c in enumerate(corrections, 1):
        print(f"{i}. {c['type']}: '{c['original']}' -> '{c['corrected']}'")
        print(f"   原因: {c['reason']}")
        print(f"   模式: {c['pattern']}")

    print("\n" + analyzer.generate_correction_summary(corrections))
