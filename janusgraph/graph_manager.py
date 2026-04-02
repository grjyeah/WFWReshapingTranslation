"""
JanusGraph 知识图谱管理器
用于存储和分析会议纪要校对行为
"""
import logging
from typing import Dict, List, Optional, Any
from gremlin_python.structure.graph import Graph
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

logger = logging.getLogger(__name__)


class JanusGraphManager:
    """JanusGraph 知识图谱管理器"""

    def __init__(self, gremlin_server_url: str = "ws://localhost:8182/gremlin"):
        """
        初始化 JanusGraph 连接

        Args:
            gremlin_server_url: Gremlin Server WebSocket URL
        """
        self.gremlin_server_url = gremlin_server_url
        self.graph = None
        self.g = None
        self.connection = None

    def connect(self):
        """连接到 JanusGraph"""
        try:
            self.connection = DriverRemoteConnection(
                f'wss://{self.gremlin_server_url.replace("ws://", "").replace("wss://", "")}/gremlin',
                'g'
            )
            self.graph = Graph()
            self.g = traversal().withRemote(self.connection)
            logger.info("成功连接到 JanusGraph")
            return True
        except Exception as e:
            logger.error(f"连接 JanusGraph 失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.connection:
            self.connection.close()
            logger.info("JanusGraph 连接已关闭")

    def initialize_schema(self):
        """初始化图数据库 Schema（创建顶点标签和边标签）"""

        # 创建管理器
        mgmt = self.get_management_system()

        try:
            # ========== 顶点标签定义 ==========

            # 文档顶点
            if not mgmt.containsVertexLabel('document'):
                document = mgmt.makeVertexLabel('document').make()
                logger.info("创建顶点标签: document")

            # 说话人顶点
            if not mgmt.containsVertexLabel('speaker'):
                speaker = mgmt.makeVertexLabel('speaker').make()
                logger.info("创建顶点标签: speaker")

            # 句子顶点（原始、书面化、人工校对）
            for sentence_type in ['original_sentence', 'formalized_sentence', 'human_sentence']:
                label = sentence_type
                if not mgmt.containsVertexLabel(label):
                    mgmt.makeVertexLabel(label).make()
                    logger.info(f"创建顶点标签: {label}")

            # 校对行为顶点
            if not mgmt.containsVertexLabel('correction'):
                mgmt.makeVertexLabel('correction').make()
                logger.info("创建顶点标签: correction")

            # 校对模式顶点
            if not mgmt.containsVertexLabel('pattern'):
                mgmt.makeVertexLabel('pattern').make()
                logger.info("创建顶点标签: pattern")

            # ========== 边标签定义 ==========

            # 说话人关系
            if not mgmt.containsEdgeLabel('has_speaker'):
                mgmt.makeEdgeLabel('has_speaker').make()
                logger.info("创建边标签: has_speaker")

            # 包含关系
            if not mgmt.containsEdgeLabel('contains'):
                mgmt.makeEdgeLabel('contains').make()
                logger.info("创建边标签: contains")

            # 书面化关系
            if not mgmt.containsEdgeLabel('formalized_to'):
                mgmt.makeEdgeLabel('formalized_to').make()
                logger.info("创建边标签: formalized_to")

            # 校对关系
            if not mgmt.containsEdgeLabel('corrected_to'):
                mgmt.makeEdgeLabel('corrected_to').make()
                logger.info("创建边标签: corrected_to")

            # 包含校对行为
            if not mgmt.containsEdgeLabel('has_correction'):
                mgmt.makeEdgeLabel('has_correction').make()
                logger.info("创建边标签: has_correction")

            # 遵循模式
            if not mgmt.containsEdgeLabel('follows_pattern'):
                mgmt.makeEdgeLabel('follows_pattern').make()
                logger.info("创建边标签: follows_pattern")

            # ========== 属性索引定义 ==========

            # 文档属性索引
            self._create_property_index(mgmt, 'filename', 'document', 'String')
            self._create_property_index(mgmt, 'date', 'document', 'String')

            # 说话人属性索引
            self._create_property_index(mgmt, 'name', 'speaker', 'String')
            self._create_property_index(mgmt, 'role', 'speaker', 'String')

            # 句子属性索引（全文搜索）
            self._create_property_index(mgmt, 'text', 'original_sentence', 'String')
            self._create_property_index(mgmt, 'text', 'formalized_sentence', 'String')
            self._create_property_index(mgmt, 'text', 'human_sentence', 'String')

            # 校对行为属性索引
            self._create_property_index(mgmt, 'type', 'correction', 'String')
            self._create_property_index(mgmt, 'pattern', 'correction', 'String')

            # 模式属性索引
            self._create_property_index(mgmt, 'name', 'pattern', 'String')

            # 提交 Schema 变更
            mgmt.commit()
            logger.info("Schema 初始化完成")

        except Exception as e:
            logger.error(f"Schema 初始化失败: {e}")
            mgmt.rollback()
            raise

    def _create_property_index(self, mgmt, property_name: str, vertex_label: str, property_type: str):
        """创建属性索引"""
        try:
            # 检查属性键是否存在
            if not mgmt.containsPropertyKey(property_name):
                # 创建属性键
                property_class = getattr(__import__('org.janusgraph.core.attribute'), property_type)
                mgmt.makePropertyKey(property_name).dataType(property_class).make()

            # 创建索引
            index_name = f"{vertex_label}_{property_name}_index"
            if not mgmt.containsGraphIndex(index_name):
                vertex_label_obj = mgmt.getVertexLabel(vertex_label)
                property_key_obj = mgmt.getPropertyKey(property_name)

                # 创建混合索引（支持全文搜索）
                mgmt.buildIndex(index_name, Vertex.class) \
                    .addKey(property_key_obj) \
                    .indexOnly(vertex_label_obj) \
                    .buildMixedIndex("search")
                logger.info(f"创建索引: {index_name}")

        except Exception as e:
            logger.warning(f"创建索引 {property_name} 失败: {e}")

    def get_management_system(self):
        """获取 JanusGraph 管理系统"""
        return self.graph.openManagement()

    # ========== 文档操作 ==========

    def create_document(self, filename: str, date: str, audio_path: str) -> str:
        """
        创建文档节点

        Args:
            filename: 文件名
            date: 日期
            audio_path: 音频文件路径

        Returns:
            顶点 ID
        """
        try:
            vertex = self.g.addV('document') \
                .property('filename', filename) \
                .property('date', date) \
                .property('audio_path', audio_path) \
                .next()
            self.g.tx().commit()
            logger.info(f"创建文档节点: {filename}")
            return vertex
        except Exception as e:
            logger.error(f"创建文档节点失败: {e}")
            self.g.tx().rollback()
            raise

    def get_document_by_filename(self, filename: str) -> Optional[Dict]:
        """根据文件名获取文档"""
        try:
            result = self.g.V().has('document', 'filename', filename).valueMap().next()
            return result
        except:
            return None

    # ========== 说话人操作 ==========

    def create_speaker(self, name: str, role: str = "participant") -> str:
        """创建说话人节点"""
        try:
            vertex = self.g.addV('speaker') \
                .property('name', name) \
                .property('role', role) \
                .next()
            self.g.tx().commit()
            logger.info(f"创建说话人节点: {name}")
            return vertex
        except Exception as e:
            logger.error(f"创建说话人节点失败: {e}")
            self.g.tx().rollback()
            raise

    def get_or_create_speaker(self, name: str, role: str = "participant") -> str:
        """获取或创建说话人"""
        try:
            existing = self.g.V().has('speaker', 'name', name).next()
            return existing
        except:
            return self.create_speaker(name, role)

    # ========== 句子操作 ==========

    def add_original_sentence(self, document_id: str, speaker_id: str,
                             text: str, timestamp: str) -> str:
        """添加原始句子（ASR 输出）"""
        try:
            vertex = self.g.addV('original_sentence') \
                .property('text', text) \
                .property('timestamp', timestamp) \
                .next()

            # 连接到文档和说话人
            self.g.V(document_id).addE('contains').to(vertex).next()
            self.g.V(speaker_id).addE('has_speaker').to(vertex).next()

            self.g.tx().commit()
            return vertex
        except Exception as e:
            logger.error(f"添加原始句子失败: {e}")
            self.g.tx().rollback()
            raise

    def add_formalized_sentence(self, original_id: str, text: str,
                                model: str, timestamp: str) -> str:
        """添加书面化句子（大模型输出）"""
        try:
            vertex = self.g.addV('formalized_sentence') \
                .property('text', text) \
                .property('model', model) \
                .property('timestamp', timestamp) \
                .next()

            # 连接到原始句子
            self.g.V(original_id).addE('formalized_to').to(vertex).next()

            self.g.tx().commit()
            return vertex
        except Exception as e:
            logger.error(f"添加书面化句子失败: {e}")
            self.g.tx().rollback()
            raise

    def add_human_corrected_sentence(self, formalized_id: str, text: str,
                                    corrector: str, timestamp: str) -> str:
        """添加人工校对句子"""
        try:
            vertex = self.g.addV('human_sentence') \
                .property('text', text) \
                .property('corrector', corrector) \
                .property('timestamp', timestamp) \
                .next()

            # 连接到书面化句子
            self.g.V(formalized_id).addE('corrected_to').to(vertex).next()

            self.g.tx().commit()
            return vertex
        except Exception as e:
            logger.error(f"添加人工校对句子失败: {e}")
            self.g.tx().rollback()
            raise

    # ========== 校对行为分析 ==========

    def analyze_corrections(self, formalized_text: str, human_text: str,
                           llm_analyzer) -> List[Dict]:
        """
        分析校对行为（使用 LLM）

        Args:
            formalized_text: 书面化文本
            human_text: 人工校对文本
            llm_analyzer: LLM 分析器实例

        Returns:
            校对行为列表
        """
        # 调用 LLM 分析差异
        corrections = llm_analyzer.analyze_corrections(formalized_text, human_text)
        return corrections

    def save_corrections(self, human_sentence_id: str, corrections: List[Dict]):
        """保存校对行为到图谱"""
        try:
            for correction in corrections:
                # 创建校对行为节点
                correction_vertex = self.g.addV('correction') \
                    .property('type', correction['type']) \
                    .property('original_text', correction['original']) \
                    .property('corrected_text', correction['corrected']) \
                    .property('reason', correction.get('reason', '')) \
                    .property('pattern', correction.get('pattern', '')) \
                    .next()

                # 连接到人工校对句子
                self.g.V(human_sentence_id).addE('has_correction').to(correction_vertex).next()

                # 如果识别到模式，连接到模式节点
                if correction.get('pattern'):
                    pattern_id = self.get_or_create_pattern(correction['pattern'])
                    self.g.V(correction_vertex).addE('follows_pattern').to(pattern_id).next()

            self.g.tx().commit()
            logger.info(f"保存 {len(corrections)} 个校对行为")
        except Exception as e:
            logger.error(f"保存校对行为失败: {e}")
            self.g.tx().rollback()
            raise

    def get_or_create_pattern(self, pattern_name: str) -> str:
        """获取或创建校对模式"""
        try:
            existing = self.g.V().has('pattern', 'name', pattern_name).next()
            # 增加频率计数
            self.g.V(existing).property('frequency', self.g.V(existing).values('frequency').next() + 1).next()
            self.g.tx().commit()
            return existing
        except:
            # 创建新模式
            vertex = self.g.addV('pattern') \
                .property('name', pattern_name) \
                .property('frequency', 1) \
                .property('confidence', 0.0) \
                .next()
            self.g.tx().commit()
            return vertex

    # ========== 查询操作 ==========

    def get_document_statistics(self, document_id: str) -> Dict:
        """获取文档统计信息"""
        try:
            stats = {
                'total_sentences': 0,
                'total_corrections': 0,
                'correction_types': {},
                'speakers': []
            }

            # 获取句子总数
            stats['total_sentences'] = self.g.V(document_id) \
                .out('contains') \
                .count() \
                .next()

            # 获取校对行为总数
            stats['total_corrections'] = self.g.V(document_id) \
                .out('contains') \
                .out('formalized_to') \
                .out('corrected_to') \
                .out('has_correction') \
                .count() \
                .next()

            # 获取校对类型分布
            correction_types = self.g.V(document_id) \
                .out('contains') \
                .out('formalized_to') \
                .out('corrected_to') \
                .out('has_correction') \
                .values('type') \
                .toList()

            for ct in correction_types:
                stats['correction_types'][ct] = stats['correction_types'].get(ct, 0) + 1

            # 获取说话人列表
            stats['speakers'] = self.g.V(document_id) \
                .out('contains') \
                .out('has_speaker') \
                .values('name') \
                .dedup() \
                .toList()

            return stats
        except Exception as e:
            logger.error(f"获取文档统计失败: {e}")
            return {}

    def get_top_patterns(self, limit: int = 10) -> List[Dict]:
        """获取最常用的校对模式"""
        try:
            patterns = self.g.V().hasLabel('pattern') \
                .order() \
                .by('frequency', decr) \
                .limit(limit) \
                .valueMap() \
                .toList()

            return patterns
        except Exception as e:
            logger.error(f"获取模式失败: {e}")
            return []

    def search_similar_corrections(self, text: str, limit: int = 5) -> List[Dict]:
        """搜索相似的校对行为"""
        try:
            # 全文搜索
            corrections = self.g.V().hasLabel('correction') \
                .has('original_text', textContains(text)) \
                .limit(limit) \
                .valueMap() \
                .toList()

            return corrections
        except Exception as e:
            logger.error(f"搜索校对行为失败: {e}")
            return []


if __name__ == "__main__":
    # 测试连接
    manager = JanusGraphManager()
    if manager.connect():
        print("✓ 成功连接到 JanusGraph")

        # 初始化 Schema
        manager.initialize_schema()
        print("✓ Schema 初始化完成")

        # 关闭连接
        manager.close()
    else:
        print("✗ 连接失败")
