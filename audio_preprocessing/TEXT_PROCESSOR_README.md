# 文本处理工作台使用说明

## 功能概述

文本处理工作台是一个基于Web的可视化界面，用于执行中文书面化和英文翻译任务。

### 主要功能

1. **中文书面化处理** - 将口语化的会议逐字稿转换为正式书面语文档
2. **英文翻译** - 将中文文本翻译为专业英文
3. **可视化配置** - 直观配置模型参数、提示词模板
4. **实时日志** - 流式显示处理进度和日志信息
5. **人工校对** - 支持在两个步骤之间编辑和调整文本

## 访问方式

### 启动服务

```bash
cd audio_preprocessing
python api_server.py
```

服务将在 `http://localhost:8001` 启动。

### 访问页面

- **音频处理页面**: http://localhost:8001/
- **文本处理页面**: http://localhost:8001/text
- **API文档**: http://localhost:8001/docs

## 使用流程

### 步骤 1: 中文书面化

1. 访问 http://localhost:8001/text
2. 在"中文书面化"标签页中：
   - 配置模型名称和API地址
   - 调整模型参数（温度、Top-P、Top-K等）
   - 粘贴或输入会议逐字稿文本
3. 点击"执行中文书面化"按钮
4. 等待处理完成，查看结果和日志

### 步骤 2: 人工校对（可选）

1. 处理完成后，结果会显示在"处理结果"区域
2. 可以直接编辑结果文本
3. 点击"使用书面化结果"按钮，将结果填入翻译输入框

### 步骤 3: 英文翻译

1. 切换到"英文翻译"标签页
2. 配置翻译模型和参数
3. 确认热词表（可选）
4. 点击"使用书面化结果"或手动输入中文文本
5. 点击"执行英文翻译"按钮
6. 等待翻译完成

## 配置说明

### 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 模型名称 | LLM模型名称 | qwen2.5-7b-instruct-1m |
| API地址 | LM Studio服务地址 | http://127.0.0.1:1234 |
| 温度 | 控制输出随机性 | 0.6 (formatter) / 0.5 (translator) |
| Top-P | 核采样参数 | 0.8 (formatter) / 0.85 (translator) |
| Top-K | 采样数量 | 40 (formatter) / 30 (translator) |
| 随机种子 | 输出稳定性控制 | 42 |

### 提示词模板

- 提示词模板从 `formatted_prompt_templates/` 目录自动加载
- 支持在页面上直接编辑提示词
- 修改后需要点击"重新加载"来恢复默认模板

### 热词表

- 格式: `中文:英文`（每行一个）
- 示例:
  ```
  张总:Mr. Zhang
  数据治理:data governance
  ```
- 从 `hotword/中译英对照词库.txt` 自动加载

## API端点

### 文本处理相关

| 端点 | 方法 | 功能 |
|------|------|------|
| `/text` | GET | 返回文本处理页面 |
| `/api/text/prompt/{type}` | GET | 获取提示词模板 |
| `/api/text/hotwords` | GET | 获取热词表 |
| `/api/text/run/{type}` | POST | 执行处理（流式） |

### 请求示例

```python
import requests

# 中文书面化
response = requests.post(
    'http://localhost:8001/api/text/run/formatter',
    json={
        'text': '【说话人:0】那个...我觉得这个项目...',
        'config': {
            'model_name': 'qwen2.5-7b-instruct-1m',
            'lm_studio_url': 'http://127.0.0.1:1234',
            'model_options': {
                'temperature': 0.6,
                'top_p': 0.8,
                'top_k': 40,
                'seed': 42
            }
        }
    },
    stream=True
)

# 处理流式响应
for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        if data['type'] == 'log':
            print(f"[{data['level']}] {data['message']}")
        elif data['type'] == 'result':
            print(data['content'], end='')
        elif data['type'] == 'complete':
            print(f"\n处理完成！输出: {data['stats']['output_length']} 字符")
```

## 日志说明

### 日志级别

- **info** - 一般信息
- **success** - 成功操作
- **warning** - 警告信息
- **error** - 错误信息
- **process** - 处理进度

### 日志操作

- **清空** - 清除当前日志显示
- **导出** - 下载日志为文本文件

## 技术栈

- **后端**: FastAPI + Python
- **前端**: Tailwind CSS + Vanilla JavaScript
- **流式传输**: Server-Sent Events (SSE)
- **LLM**: 支持Ollama/LM Studio兼容API

## 故障排除

### 问题: 无法连接到LLM服务

**解决方案**:
1. 确认LM Studio或Ollama服务正在运行
2. 检查API地址配置是否正确
3. 查看日志窗口中的详细错误信息

### 问题: 处理速度慢

**解决方案**:
1. 检查网络连接
2. 调整输入文本长度
3. 考虑使用更快的模型

### 问题: 结果不理想

**解决方案**:
1. 调整温度参数（降低可提高一致性）
2. 修改提示词模板
3. 添加更多热词

## 开发说明

### 目录结构

```
audio_preprocessing/
├── api_server.py              # 主API服务器
├── audio_preprocessing_gpu.py # GPU音频处理器
├── templates/
│   ├── index.html             # 音频处理页面
│   └── text_processor.html    # 文本处理页面（新增）
└── static/                    # 静态资源目录

../formatted_prompt_templates/
├── chinese_formatter_prompt.xml    # 中文格式化提示词
└── english_translator_prompt.xml   # 英文翻译提示词

../hotword/
└── 中译英对照词库.txt         # 热词对照表
```

### 扩展功能

如需添加新的处理类型：

1. 在 `api_server.py` 中添加新的处理类型
2. 在 `text_processor.html` 中添加对应的配置界面
3. 更新提示词模板和API端点

## 版本历史

### v1.0 (2026-03-17)

- 初始版本
- 支持中文书面化和英文翻译
- 可视化配置界面
- 流式日志输出
- 人工校对支持

## 联系支持

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 文档: [项目Wiki](https://github.com/your-repo/wiki)
