# 知识图谱预处理系统 - 完整指南

## 📖 系统概述

本系统基于 **JanusGraph** 图数据库，用于存储和分析会议纪要的人工校对行为模式，通过学习历史校对数据，优化未来的自动书面化质量。

### 核心功能

1. **批量上传** - 支持批量上传音频文件和对应的人工校对文本
2. **自动书面化** - 调用大模型对音频转文字结果进行书面化
3. **智能对比** - 使用 LLM 分析人工校对与大模型输出的差异
4. **模式提取** - 从校对行为中提取可复用的模式
5. **知识图谱** - 将所有数据存储到 JanusGraph 中，支持复杂查询

---

## 🏗️ 技术架构

### 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **图数据库** | JanusGraph 1.0.0 | 图数据库中间件 |
| **后端存储** | Apache Cassandra 4.1 | 分布式 NoSQL 数据库 |
| **索引引擎** | Elasticsearch 8.11 | 全文搜索和索引 |
| **查询语言** | Gremlin | 图遍历语言 |
| **Python 驱动** | gremlinpython 3.7.1 | Python 客户端 |
| **分析引擎** | 本地 LLM | 差异分析和模式提取 |

### 数据模型

#### 节点类型

```
┌─────────────┐
│  Document   │ 文档节点
│  - filename │
│  - date     │
│  - audio_path│
└─────────────┘
       │
       │ has_speaker
       ↓
┌─────────────┐
│   Speaker   │ 说话人节点
│  - name     │
│  - role     │
└─────────────┘
       │
       │ contains
       ↓
┌──────────────────┐
│ OriginalSentence │ 原始句子（ASR）
│  - text          │
│  - timestamp     │
└──────────────────┘
       │
       │ formalized_to
       ↓
┌────────────────────┐
│ FormalalizedSentence│ 书面化句子（LLM）
│  - text             │
│  - model            │
│  - timestamp        │
└────────────────────┘
       │
       │ corrected_to
       ↓
┌──────────────────┐
│  HumanSentence   │ 人工校对句子
│  - text          │
│  - corrector     │
│  - timestamp     │
└──────────────────┘
       │
       │ has_correction
       ↓
┌──────────────────┐
│   Correction     │ 校对行为
│  - type          │
│  - original      │
│  - corrected     │
│  - reason        │
│  - pattern       │
└──────────────────┘
       │
       │ follows_pattern
       ↓
┌──────────────────┐
│    Pattern       │ 校对模式
│  - name          │
│  - frequency     │
│  - confidence    │
└──────────────────┘
```

#### 边类型

| 边类型 | 从 → 到 | 说明 |
|--------|---------|------|
| `has_speaker` | Document → Speaker | 文档包含说话人 |
| `contains` | Document → OriginalSentence | 文档包含原始句子 |
| `formalized_to` | OriginalSentence → FormalalizedSentence | 书面化关系 |
| `corrected_to` | FormalalizedSentence → HumanSentence | 校对关系 |
| `has_correction` | HumanSentence → Correction | 包含校对行为 |
| `follows_pattern` | Correction → Pattern | 遵循模式 |

---

## 🚀 快速开始

### 1. 安装 Java JDK

**下载**：https://adoptium.net/
- 选择 **Temurin 17 (LTS)** > **Windows** > **x64**
- 安装并配置环境变量

**验证**：
```powershell
java -version
# 应显示：openjdk version "17.0.x"
```

### 2. 使用 Docker 启动服务（推荐）

进入项目目录：
```powershell
cd D:\pyworkspace\WFWReshapingTranslation\janusgraph
```

一键启动所有服务：
```powershell
docker-compose up -d
```

等待约 1 分钟，直到所有服务启动完成。

### 3. 验证安装

访问以下 URL 验证服务：

- **Elasticsearch**: http://localhost:9200
- **Gremlin Console**: http://localhost:8182

---

## 📁 文件结构

```
janusgraph/
├── docker-compose.yml          # Docker Compose 配置
├── start.bat                   # Windows 启动脚本
├── janusgraph-config.properties # JanusGraph 配置
├── INSTALL_GUIDE.md            # 详细安装指南
├── graph_manager.py            # 图数据库管理器
├── llm_analyzer.py             # LLM 对比分析器
└── README.md                   # 本文件
```

---

## 💻 使用示例

### 初始化知识图谱

```python
from janusgraph.graph_manager import JanusGraphManager

# 创建管理器
manager = JanusGraphManager()

# 连接到 JanusGraph
if manager.connect():
    # 初始化 Schema
    manager.initialize_schema()
    print("✓ 知识图谱初始化完成")

    # 关闭连接
    manager.close()
```

### 添加文档和说话人

```python
# 创建文档节点
document_id = manager.create_document(
    filename="meeting_001.mp3",
    date="2026-04-02",
    audio_path="/path/to/meeting_001.mp3"
)

# 创建说话人节点
speaker_id = manager.get_or_create_speaker(
    name="张总",
    role="主持人"
)
```

### 分析校对行为

```python
from janusgraph.llm_analyzer import TextComparisonAnalyzer

# 创建分析器
analyzer = TextComparisonAnalyzer(
    lm_studio_url="http://127.0.0.1:1234",
    model_name="qwen2.5-7b-instruct"
)

# 分析差异
formalized_text = "【说话人:0】我觉得这个项目应该尽快推进。"
human_text = "【张总】我认为这个项目需要加快进度，确保按时完成。"

corrections = analyzer.analyze_corrections(formalized_text, human_text)

# 查看结果
for correction in corrections:
    print(f"{correction['type']}: {correction['reason']}")
```

---

## 🌐 Web 界面

### 访问预处理页面

启动 API 服务：
```powershell
cd audio_preprocessing
python api_server.py
```

访问：
- **知识图谱预处理**: http://localhost:8001/knowledge-graph

### 功能说明

1. **上传文件** - 拖拽文件夹到上传区域
2. **自动识别** - 系统自动匹配音频和文本文件
3. **批量处理** - 一键处理所有文件
4. **实时日志** - 查看处理进度和日志
5. **统计分析** - 查看处理统计信息

---

## 🔧 Gremlin 查询示例

### 查询所有校对行为

```gremlin
g.V().hasLabel('correction').valueMap()
```

### 查询最常用的校对模式

```gremlin
g.V().hasLabel('pattern')
  .order()
    .by('frequency', decr)
  .limit(10)
  .valueMap()
```

### 查询某个文档的所有校对

```gremlin
g.V().has('document', 'filename', 'meeting_001.mp3')
  .out('contains')
  .out('formalized_to')
  .out('corrected_to')
  .out('has_correction')
  .valueMap()
```

### 查询相似校对行为

```gremlin
g.V().hasLabel('correction')
  .has('original_text', textContains('项目'))
  .valueMap()
```

---

## 📊 下一步功能

### 待实现功能

- [ ] Web 界面集成批量处理 API
- [ ] 实时处理进度推送（WebSocket）
- [ ] 图谱可视化展示
- [ ] 模式推荐引擎
- [ ] 自动应用校对模式
- [ ] 导出分析报告

### 高级功能

- [ ] 分布式处理支持
- [ ] 实时学习更新
- [ ] A/B 测试框架
- [ ] 模式置信度计算
- [ ] 多模态分析（音频+文本）

---

## 📞 支持和反馈

如有问题或建议，请联系开发团队。

---

**系统版本**: 1.0.0
**最后更新**: 2026-04-02
**维护者**: AI 开发团队
