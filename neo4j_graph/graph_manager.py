"""
Neo4j 知识图谱管理器
用于存储和分析会议纪要校对行为
"""
import logging
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class Neo4jGraphManager:
    """Neo4j 知识图谱管理器"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "password123"):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j Bolt URI
            user: 用户名
            password: 密码
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        """连接到 Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 测试连接
            self.driver.verify_connectivity()
            logger.info(f"成功连接到 Neo4j: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"连接 Neo4j 失败: {e}")
            logger.error(f"请检查：")
            logger.error(f"  1. Neo4j 是否正在运行")
            logger.error(f"  2. URI 是否正确: {self.uri}")
            logger.error(f"  3. 用户名密码是否正确")
            return False

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")

    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        执行 Cypher 查询

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            logger.debug(f"查询: {query}")
            logger.debug(f"参数: {parameters}")
            return []

    def initialize_schema(self):
        """初始化图数据库 Schema（创建约束和索引）"""

        queries = [
            # ========== 创建约束（唯一性约束） ==========

            # 文档唯一约束
            "CREATE CONSTRAINT document_filename_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE",

            # 说话人唯一约束
            "CREATE CONSTRAINT speaker_name_unique IF NOT EXISTS FOR (s:Speaker) REQUIRE s.name IS UNIQUE",

            # 句子唯一约束
            "CREATE CONSTRAINT sentence_id_unique IF NOT EXISTS FOR (s:Sentence) REQUIRE s.id IS UNIQUE",

            # 校对行为唯一约束
            "CREATE CONSTRAINT correction_id_unique IF NOT EXISTS FOR (c:Correction) REQUIRE c.id IS UNIQUE",

            # 校对模式唯一约束
            "CREATE CONSTRAINT pattern_name_unique IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE",

            # ========== 创建全文索引 ==========

            # 文档索引
            "CREATE INDEX document_date_index IF NOT EXISTS FOR (d:Document) ON (d.date)",

            # 句子全文索引
            "CREATE FULLTEXT INDEX sentence_text_index IF NOT EXISTS FOR (s:Sentence) ON EACH [s.text]",

            # 校对行为索引
            "CREATE INDEX correction_type_index IF NOT EXISTS FOR (c:Correction) ON (c.type)",
            "CREATE INDEX correction_pattern_index IF NOT EXISTS FOR (c:Correction) ON (c.pattern)",

            # 模式索引
            "CREATE INDEX pattern_frequency_index IF NOT EXISTS FOR (p:Pattern) ON (p.frequency)",
        ]

        for query in queries:
            try:
                self.execute_query(query)
                logger.info(f"执行成功: {query[:60]}...")
            except Exception as e:
                logger.warning(f"执行失败（可能已存在）: {e}")

        logger.info("Schema 初始化完成")

    # ========== 文档操作 ==========

    def create_document(self, filename: str, date: str, audio_path: str) -> str:
        """
        创建文档节点

        Args:
            filename: 文件名
            date: 日期
            audio_path: 音频文件路径

        Returns:
            文档内部 ID
        """
        query = """
        CREATE (d:Document {
            filename: $filename,
            date: $date,
            audio_path: $audio_path,
            created_at: datetime()
        })
        RETURN elementId(d) as id
        """

        try:
            result = self.execute_query(query, {
                "filename": filename,
                "date": date,
                "audio_path": audio_path
            })

            if result:
                doc_id = result[0]['id']
                logger.info(f"创建文档节点: {filename} (ID: {doc_id})")
                return doc_id
            else:
                raise Exception("创建文档失败")
        except Exception as e:
            logger.error(f"创建文档节点失败: {e}")
            raise

    def get_document_by_filename(self, filename: str) -> Optional[Dict]:
        """根据文件名获取文档"""
        query = """
        MATCH (d:Document {filename: $filename})
        RETURN d
        """

        result = self.execute_query(query, {"filename": filename})
        return result[0]['d'] if result else None

    # ========== 说话人操作 ==========

    def create_speaker(self, name: str, role: str = "participant") -> str:
        """创建说话人节点"""
        query = """
        MERGE (s:Speaker {name: $name})
        ON CREATE SET s.role = $role, s.created_at = datetime()
        RETURN elementId(s) as id
        """

        try:
            result = self.execute_query(query, {"name": name, "role": role})
            speaker_id = result[0]['id']
            logger.info(f"创建/获取说话人节点: {name} (ID: {speaker_id})")
            return speaker_id
        except Exception as e:
            logger.error(f"创建说话人节点失败: {e}")
            raise

    def get_or_create_speaker(self, name: str, role: str = "participant") -> str:
        """获取或创建说话人"""
        return self.create_speaker(name, role)

    # ========== 句子操作 ==========

    def add_sentence(self, document_id: str, speaker_id: str, text: str,
                    sentence_type: str, timestamp: str = None) -> str:
        """
        添加句子节点

        Args:
            document_id: 文档内部 ID
            speaker_id: 说话人内部 ID
            text: 句子文本
            sentence_type: 句子类型 (original/formalized/human)
            timestamp: 时间戳

        Returns:
            句子内部 ID
        """
        import uuid
        sentence_id = str(uuid.uuid4())

        query = """
        MATCH (d:Document)
        WHERE elementId(d) = $document_id
        MATCH (s:Speaker)
        WHERE elementId(s) = $speaker_id
        CREATE (d)-[:HAS_SENTENCE]->(sent:Sentence {
            id: $id,
            text: $text,
            type: $type,
            timestamp: $timestamp,
            created_at: datetime()
        })
        CREATE (s)-[:SPOKE]->(sent)
        RETURN elementId(sent) as id
        """

        try:
            result = self.execute_query(query, {
                "document_id": document_id,
                "speaker_id": speaker_id,
                "id": sentence_id,
                "text": text,
                "type": sentence_type,
                "timestamp": timestamp
            })

            if result:
                sent_id = result[0]['id']
                logger.info(f"创建句子节点: {sentence_type} (ID: {sent_id})")
                return sent_id
            else:
                raise Exception("创建句子失败")
        except Exception as e:
            logger.error(f"创建句子节点失败: {e}")
            raise

    def link_formalized_sentence(self, original_id: str, formalized_id: str):
        """关联原始句子和书面化句子"""
        query = """
        MATCH (orig:Sentence)
        WHERE elementId(orig) = $original_id
        MATCH (form:Sentence)
        WHERE elementId(form) = $formalized_id
        CREATE (orig)-[:FORMALIZED_TO]->(form)
        """

        self.execute_query(query, {
            "original_id": original_id,
            "formalized_id": formalized_id
        })

    def link_corrected_sentence(self, formalized_id: str, human_id: str):
        """关联书面化句子和人工校对句子"""
        query = """
        MATCH (form:Sentence)
        WHERE elementId(form) = $formalized_id
        MATCH (human:Sentence)
        WHERE elementId(human) = $human_id
        CREATE (form)-[:CORRECTED_TO]->(human)
        """

        self.execute_query(query, {
            "formalized_id": formalized_id,
            "human_id": human_id
        })

    # ========== 校对行为分析 ==========

    def save_corrections(self, human_sentence_id: str, corrections: List[Dict]):
        """
        保存校对行为到图谱

        Args:
            human_sentence_id: 人工校对句子内部 ID
            corrections: 校对行为列表
        """
        import uuid

        for correction in corrections:
            correction_id = str(uuid.uuid4())

            # 创建校对行为节点
            query = """
            MATCH (human:Sentence)
            WHERE elementId(human) = $human_id
            CREATE (human)-[:HAS_CORRECTION]->(c:Correction {
                id: $id,
                type: $type,
                original_text: $original_text,
                corrected_text: $corrected_text,
                reason: $reason,
                created_at: datetime()
            })
            """

            self.execute_query(query, {
                "human_id": human_sentence_id,
                "id": correction_id,
                "type": correction.get('type', '未知'),
                "original_text": correction.get('original', ''),
                "corrected_text": correction.get('corrected', ''),
                "reason": correction.get('reason', '')
            })

            # 如果识别到模式，创建或更新模式节点
            if correction.get('pattern'):
                self.get_or_create_pattern(correction_id, correction['pattern'])

        logger.info(f"保存 {len(corrections)} 个校对行为")

    def get_or_create_pattern(self, correction_id: str, pattern_name: str):
        """获取或创建校对模式"""
        query = """
        MATCH (c:Correction {id: $correction_id})
        MERGE (p:Pattern {name: $pattern_name})
        ON CREATE SET p.frequency = 1, p.confidence = 0.0, p.created_at = datetime()
        ON MATCH SET p.frequency = p.frequency + 1
        CREATE (c)-[:FOLLOWS_PATTERN]->(p)
        RETURN p
        """

        self.execute_query(query, {
            "correction_id": correction_id,
            "pattern_name": pattern_name
        })

    # ========== 查询操作 ==========

    def get_document_statistics(self, filename: str) -> Dict:
        """获取文档统计信息"""
        query = """
        MATCH (d:Document {filename: $filename})
        OPTIONAL MATCH (d)-[:HAS_SENTENCE]->(sent:Sentence)
        OPTIONAL MATCH (sent)-[:HAS_CORRECTION]->(c:Correction)
        RETURN
            count(DISTINCT sent) as total_sentences,
            count(DISTINCT c) as total_corrections
        """

        result = self.execute_query(query, {"filename": filename})

        if result:
            stats = {
                'total_sentences': result[0]['total_sentences'],
                'total_corrections': result[0]['total_corrections'],
                'correction_types': self._get_correction_type_distribution(filename),
                'speakers': self._get_document_speakers(filename)
            }
            return stats

        return {}

    def _get_correction_type_distribution(self, filename: str) -> Dict[str, int]:
        """获取校对类型分布"""
        query = """
        MATCH (d:Document {filename: $filename})-[:HAS_SENTENCE]->(sent:Sentence)-[:HAS_CORRECTION]->(c:Correction)
        RETURN c.type as type, count(c) as count
        """

        result = self.execute_query(query, {"filename": filename})
        return {r['type']: r['count'] for r in result}

    def _get_document_speakers(self, filename: str) -> List[str]:
        """获取文档中的说话人列表"""
        query = """
        MATCH (d:Document {filename: $filename})-[:HAS_SENTENCE]->(sent:Sentence)<-[:SPOKE]-(s:Speaker)
        RETURN DISTINCT s.name as name
        """

        result = self.execute_query(query, {"filename": filename})
        return [r['name'] for r in result]

    def get_top_patterns(self, limit: int = 10) -> List[Dict]:
        """获取最常用的校对模式"""
        query = """
        MATCH (p:Pattern)
        RETURN p.name as name, p.frequency as frequency, p.confidence as confidence
        ORDER BY p.frequency DESC
        LIMIT $limit
        """

        result = self.execute_query(query, {"limit": limit})
        return [
            {
                'name': r['name'],
                'frequency': r['frequency'],
                'confidence': r['confidence']
            }
            for r in result
        ]

    def search_similar_corrections(self, text: str, limit: int = 5) -> List[Dict]:
        """搜索相似的校对行为（全文搜索）"""
        query = """
        CALL db.index.fulltext.queryNodes('sentence_text_index', $text)
        YIELD node, score
        MATCH (node)-[:HAS_CORRECTION]->(c:Correction)
        RETURN c.original_text as original, c.corrected_text as corrected,
               c.type as type, c.reason as reason, score
        ORDER BY score DESC
        LIMIT $limit
        """

        result = self.execute_query(query, {"text": text, "limit": limit})
        return result

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档列表"""
        query = """
        MATCH (d:Document)
        RETURN d.filename as filename, d.date as date, d.audio_path as audio_path
        ORDER BY d.date DESC
        """

        return self.execute_query(query)

    def get_correction_patterns_by_type(self, correction_type: str) -> List[Dict]:
        """根据校对类型获取模式"""
        query = """
        MATCH (c:Correction {type: $type})-[:FOLLOWS_PATTERN]->(p:Pattern)
        RETURN p.name as pattern, count(c) as frequency
        ORDER BY frequency DESC
        """

        return self.execute_query(query, {"type": correction_type})

    def get_recommended_corrections(self, text: str) -> List[Dict]:
        """
        基于历史模式推荐校对建议

        Args:
            text: 待校对的文本

        Returns:
            推荐的校对建议列表
        """
        # 查找相似的历史校对
        similar = self.search_similar_corrections(text, limit=10)

        # 提取相关模式
        recommendations = []
        for item in similar:
            recommendations.append({
                'original': item['original'],
                'suggested': item['corrected'],
                'type': item['type'],
                'reason': item['reason'],
                'confidence': item['score']
            })

        return recommendations


if __name__ == "__main__":
    # 测试连接
    import sys

    # 从命令行获取密码
    password = sys.argv[1] if len(sys.argv) > 1 else "password123"

    manager = Neo4jGraphManager(password=password)
    if manager.connect():
        print("✓ 成功连接到 Neo4j")

        # 初始化 Schema
        manager.initialize_schema()
        print("✓ Schema 初始化完成")

        # 测试创建文档
        doc_id = manager.create_document(
            filename="test_meeting.mp3",
            date="2026-04-02",
            audio_path="/path/to/test.mp3"
        )
        print(f"✓ 创建测试文档，ID: {doc_id}")

        # 关闭连接
        manager.close()
    else:
        print("✗ 连接失败")
        print("请检查：")
        print("  1. Neo4j 是否正在运行")
        print("  2. 密码是否正确")
        sys.exit(1)
