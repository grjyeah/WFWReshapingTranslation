# Neo4j 知识图谱系统 - 快速启动指南

## ✅ 你已完成

- ✅ Neo4j 已安装并运行
- ✅ 已成功登录 Neo4j Browser: http://localhost:7474/browser/

---

## 🚀 快速开始（3 步）

### 第 1 步：初始化知识图谱

1. 打开 Neo4j Browser: http://localhost:7474/browser/
2. 复制以下命令并执行：

```cypher
// 创建约束
CREATE CONSTRAINT document_filename_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE;
CREATE CONSTRAINT speaker_name_unique IF NOT EXISTS FOR (s:Speaker) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT sentence_id_unique IF NOT EXISTS FOR (s:Sentence) REQUIRE s.id IS UNIQUE;

// 创建全文索引
CREATE FULLTEXT INDEX sentence_text_index IF NOT EXISTS FOR (s:Sentence) ON EACH [s.text];

// 验证安装
MATCH (n) RETURN labels(n) as node_type, count(n) as count;
```

或者直接执行完整初始化脚本：
```
D:\pyworkspace\WFWReshapingTranslation\neo4j_graph\init_schema.cypher
```

### 第 2 步：启动 Web 服务

```powershell
cd D:\pyworkspace\WFWReshapingTranslation\audio_preprocessing
python api_server.py
```

### 第 3 步：访问页面

打开浏览器访问：**http://localhost:8001/knowledge-graph**

---

## 🔑 Neo4j 连接信息

**重要**：请确认你的 Neo4j 密码，然后在 `api_server.py` 中修改：

```python
# 第 817 行左右
NEO4J_PASSWORD = "你的密码"  # 默认是 password123
```

---

## 📊 知识图谱数据模型

### 节点类型

| 节点 | 说明 | 属性示例 |
|------|------|----------|
| **Document** | 文档 | filename, date, audio_path |
| **Speaker** | 说话人 | name, role |
| **Sentence** | 句子 | id, text, type (original/formalized/human) |
| **Correction** | 校对行为 | type, original_text, corrected_text, reason |
| **Pattern** | 校对模式 | name, frequency, confidence |

### 关系类型

```
Document -[:HAS_SENTENCE]-> Sentence
Speaker -[:SPOKE]-> Sentence
Sentence -[:FORMALIZED_TO]-> Sentence (书面化)
Sentence -[:CORRECTED_TO]-> Sentence (人工校对)
Sentence -[:HAS_CORRECTION]-> Correction
Correction -[:FOLLOWS_PATTERN]-> Pattern
```

---

## 🔧 常用 Cypher 查询

### 查询所有文档
```cypher
MATCH (d:Document)
RETURN d
ORDER BY d.date DESC;
```

### 查询某个文档的统计
```cypher
MATCH (d:Document {filename: "meeting_001.mp3"})
OPTIONAL MATCH (d)-[:HAS_SENTENCE]->(s:Sentence)
OPTIONAL MATCH (s)-[:HAS_CORRECTION]->(c:Correction)
RETURN
  count(DISTINCT s) as total_sentences,
  count(DISTINCT c) as total_corrections;
```

### 查询最常用的校对模式
```cypher
MATCH (p:Pattern)
RETURN p.name, p.frequency
ORDER BY p.frequency DESC
LIMIT 10;
```

### 全文搜索相似句子
```cypher
CALL db.index.fulltext.queryNodes('sentence_text_index', '项目推进')
YIELD node, score
RETURN node.text, score
ORDER BY score DESC
LIMIT 5;
```

### 查询某个说话人的所有校对行为
```cypher
MATCH (s:Speaker {name: "张总"})-[:SPOKE]->(sent:Sentence)-[:HAS_CORRECTION]->(c:Correction)
RETURN c.type, c.original_text, c.corrected_text
ORDER BY c.type;
```

---

## 🧪 测试连接

### 方法 1：使用 Python 脚本测试

```powershell
cd D:\pyworkspace\WFWReshapingTranslation\neo4j_graph
python graph_manager.py 你的密码
```

### 方法 2：使用 API 测试

```powershell
# 检查连接状态
curl http://localhost:8001/api/graph/health

# 初始化 Schema
curl -X POST http://localhost:8001/api/graph/initialize

# 获取所有文档
curl http://localhost:8001/api/graph/documents
```

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `neo4j_graph/graph_manager.py` | Neo4j 图管理器 |
| `neo4j_graph/llm_analyzer.py` | LLM 对比分析器 |
| `neo4j_graph/init_schema.cypher` | Cypher 初始化脚本 |
| `audio_preprocessing/templates/knowledge_graph.html` | Web 界面 |
| `audio_preprocessing/api_server.py` | API 服务（已集成知识图谱端点） |

---

## 🎯 下一步

### 功能开发优先级

1. **基础功能**（已完成）
   - ✅ Neo4j 连接
   - ✅ Schema 初始化
   - ✅ 基础 CRUD 操作
   - ✅ Web 界面框架

2. **核心功能**（待实现）
   - ⏳ 批量文件上传处理
   - ⏳ ASR 集成（音频转文字）
   - ⏳ 书面化集成
   - ⏳ 对比分析完整流程

3. **高级功能**（待实现）
   - ⏳ 实时处理进度
   - ⏳ 模式推荐引擎
   - ⏳ 图谱可视化
   - ⏳ 统计报表

---

## 🆘 常见问题

### Q: 连接 Neo4j 失败？
**A**: 检查：
1. Neo4j 是否正在运行
2. 密码是否正确（默认是 password123）
3. 端口是否正确（Bolt: 7687, HTTP: 7474）

### Q: 如何查看数据？
**A**: 访问 http://localhost:7474/browser/ 执行 Cypher 查询

### Q: 如何重置图谱？
**A**: 执行 `MATCH (n) DETACH DELETE n` 删除所有数据

---

**准备好了吗？让我们开始！** 🚀

如果遇到问题，请告诉我。
