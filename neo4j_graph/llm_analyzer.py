"""
LLM 驱动的文本对比分析器（Neo4j 版本）
用于分析人工校对与大模型书面化的差异
"""
import requests
import json
import re
from typing import List, Dict
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
            校对行为列表
        """
        prompt = self._build_analysis_prompt(formalized_text, human_text)

        try:
            response = self._call_llm(prompt)
            corrections = self._parse_corrections(response)
            logger.info(f"分析完成，识别到 {len(corrections)} 个校对行为")
            return corrections
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return []

    def _build_analysis_prompt(self, formalized_text: str, human_text: str) -> str:
        """构建分析提示词"""
        return f"""你是一个专业的文本分析专家，专门分析会议纪要校对行为。

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

修改类型包括：删除、添加、修改、合并、分割、术语统一、口语化调整、语气调整等。

请只输出 JSON 数组，不要包含其他内容。"""

    def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """调用本地 LLM"""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个专业的文本分析专家，擅长识别和分析文本修改模式。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=120,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def _parse_corrections(self, llm_output: str) -> List[Dict]:
        """解析 LLM 输出为结构化数据"""
        try:
            # 尝试直接解析 JSON
            if llm_output.startswith('['):
                return json.loads(llm_output)

            # 如果包含 markdown 代码块，提取 JSON
            if '```json' in llm_output:
                match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_output)
                if match:
                    return json.loads(match.group(1))

            # 如果包含 ``` 但不是 json
            if '```' in llm_output:
                match = re.search(r'```\s*([\s\S]*?)\s*```', llm_output)
                if match:
                    return json.loads(match.group(1))

            # 尝试查找 JSON 数组
            match = re.search(r'\[[\s\S]*\]', llm_output)
            if match:
                return json.loads(match.group(0))

            logger.warning(f"无法解析 LLM 输出: {llm_output[:200]}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            return []

    def extract_patterns(self, corrections: List[Dict]) -> Dict[str, int]:
        """从校对行为中提取模式"""
        patterns = {}
        for correction in corrections:
            pattern_name = correction.get('pattern', '未分类')
            patterns[pattern_name] = patterns.get(pattern_name, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
