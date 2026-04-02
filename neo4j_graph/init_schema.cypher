// ============================================
// Neo4j 知识图谱初始化脚本
// 使用方法：复制到 Neo4j Browser 中执行
// http://localhost:7474/browser/
// ============================================

// ========== 1. 创建约束（唯一性约束） ==========

// 文档文件名唯一
CREATE CONSTRAINT document_filename_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE;

// 说话人名称唯一
CREATE CONSTRAINT speaker_name_unique IF NOT EXISTS FOR (s:Speaker) REQUIRE s.name IS UNIQUE;

// 句子 ID 唯一
CREATE CONSTRAINT sentence_id_unique IF NOT EXISTS FOR (s:Sentence) REQUIRE s.id IS UNIQUE;

// 校对行为 ID 唯一
CREATE CONSTRAINT correction_id_unique IF NOT EXISTS FOR (c:Correction) REQUIRE c.id IS UNIQUE;

// 校对模式名称唯一
CREATE CONSTRAINT pattern_name_unique IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE;

// ========== 2. 创建索引 ==========

// 文档日期索引
CREATE INDEX document_date_index IF NOT EXISTS FOR (d:Document) ON (d.date);

// 文档创建时间索引
CREATE INDEX document_created_index IF NOT EXISTS FOR (d:Document) ON (d.created_at);

// 说话人角色索引
CREATE INDEX speaker_role_index IF NOT EXISTS FOR (s:Speaker) ON (s.role);

// 句子类型索引
CREATE INDEX sentence_type_index IF NOT EXISTS FOR (s:Sentence) ON (s.type);

// 句子时间戳索引
CREATE INDEX sentence_timestamp_index IF NOT EXISTS FOR (s:Sentence) ON (s.timestamp);

// 校对行为类型索引
CREATE INDEX correction_type_index IF NOT EXISTS FOR (c:Correction) ON (c.type);

// 校对模式频率索引
CREATE INDEX pattern_frequency_index IF NOT EXISTS FOR (p:Pattern) ON (p.frequency);

// ========== 3. 创建全文索引 ==========

// 句子文本全文索引（支持模糊搜索）
CREATE FULLTEXT INDEX sentence_text_index IF NOT EXISTS FOR (s:Sentence) ON EACH [s.text];

// 校对行为文本全文索引
CREATE FULLTEXT INDEX correction_text_index IF NOT EXISTS FOR (c:Correction) ON EACH [c.original_text, c.corrected_text];

// ========== 4. 创建示例数据（可选） ==========

// 创建示例文档
CREATE (doc:Document {
    filename: "meeting_001.mp3",
    date: "2026-04-02",
    audio_path: "/data/meetings/meeting_001.mp3",
    created_at: datetime()
});

// 创建示例说话人
CREATE (speaker1:Speaker {
    name: "张总",
    role: "主持人",
    created_at: datetime()
});

CREATE (speaker2:Speaker {
    name: "李经理",
    role: "参与者",
    created_at: datetime()
});

// 创建示例句子关系
MATCH (doc:Document {filename: "meeting_001.mp3"})
MATCH (speaker1:Speaker {name: "张总"})
CREATE (doc)-[:HAS_SENTENCE]->(sent1:Sentence {
    id: "sent_001",
    text: "那个...我觉得这个项目应该尽快推进",
    type: "original",
    timestamp: "00:01:23",
    created_at: datetime()
})
CREATE (speaker1)-[:SPOKE]->(sent1);

// 创建书面化句子
CREATE (sent2:Sentence {
    id: "sent_002",
    text: "我认为这个项目应该尽快推进。",
    type: "formalized",
    model: "qwen2.5-7b-instruct",
    created_at: datetime()
});

CREATE (sent1)-[:FORMALIZED_TO]->(sent2);

// 创建人工校对句子
CREATE (sent3:Sentence {
    id: "sent_003",
    text: "我认为这个项目需要加快进度，确保按时完成。",
    type: "human",
    corrector: "人工校对员A",
    created_at: datetime()
});

CREATE (sent2)-[:CORRECTED_TO]->(sent3);

// 创建校对行为
CREATE (sent3)-[:HAS_CORRECTION]->(corr:Correction {
    id: "corr_001",
    type: "修改",
    original_text: "应该尽快推进",
    corrected_text: "需要加快进度，确保按时完成",
    reason: "使表达更具体、更正式",
    created_at: datetime()
});

// 创建校对模式
MERGE (p:Pattern {name: "增强表达的具体性"})
ON CREATE SET p.frequency = 1, p.confidence = 0.8, p.created_at = datetime()
ON MATCH SET p.frequency = p.frequency + 1
CREATE (corr)-[:FOLLOWS_PATTERN]->(p);

// ========== 5. 验证安装 ==========

// 查看所有节点
MATCH (n)
RETURN labels(n) as node_type, count(n) as count
ORDER BY count DESC;

// 查看所有关系类型
MATCH (n)-[r]->(m)
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC;

// 查看示例图谱路径
MATCH path = (d:Document {filename: "meeting_001.mp3"})-[:HAS_SENTENTE]->(s:Sentence)<-[:SPOKE]-(sp:Speaker)
RETURN path
LIMIT 10;

// ========== 6. 常用查询模板 ==========

// 查询所有文档
// MATCH (d:Document) RETURN d ORDER BY d.date DESC;

// 查询某个文档的所有说话人
// MATCH (d:Document {filename: "meeting_001.mp3"})-[:HAS_SENTENTE]->(s:Sentence)<-[:SPOKE]-(sp:Speaker)
// RETURN DISTINCT sp.name, sp.role;

// 查询所有校对行为
// MATCH (c:Correction) RETURN c.type as type, count(c) as count ORDER BY count DESC;

// 查询最常用的校对模式
// MATCH (p:Pattern) RETURN p.name, p.frequency ORDER BY p.frequency DESC LIMIT 10;

// 全文搜索句子
// CALL db.index.fulltext.queryNodes('sentence_text_index', '项目') YIELD node, score
// RETURN node.text, score ORDER BY score DESC LIMIT 5;

// ============================================
// 初始化完成！
// ============================================
