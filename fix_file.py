# -*- coding: utf-8 -*-
import re

with open('english_translator-ollama.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义需要替换的错误模式
old_pattern = r'''    def _clean_thinking_tags\(self, text: str\) -> str:
        """
        清理输出中的思考标签内容

        Args:
            text: 原始输出文本

        Returns:
            清理后的文本
        """
        import re

        # 移除 <thinking>\.\.\.</thinking> 标签及其内容
        text = re\.sub\(r'<thinking>\.\*\?</thinking>', '', text, flags=re\.DOTALL \| re\.IGNORECASE\)

        # 清理多余空行
        text = re\.sub\(r'
\s\*
\s\*
', '
\n', text\)

        return text\.strip\(\)'''

new_method = '''    def _clean_thinking_tags(self, text: str) -> str:
        """
        清理输出中的思考标签内容

        Args:
            text: 原始输出文本

        Returns:
            清理后的文本
        """
        import re

        # 移除 <thinking>...</thinking> 标签及其内容
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # 清理多余空行
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

        return text.strip()'''

# 使用正则表达式替换
content = re.sub(old_pattern, new_method, content, flags=re.DOTALL)

with open('english_translator-ollama.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('修复完成')
