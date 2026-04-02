# JanusGraph 本地部署完整指南

## 📋 前置要求

### 必需组件
1. ✅ **Python 3.8+** - 已安装
2. ⏳ **Java JDK 11 或 17** - 需要手动安装
3. ⏳ **Apache Cassandra** - 后端存储
4. ⏳ **Elasticsearch** - 索引后端

---

## 🔧 手动安装步骤

### 1. 安装 Java JDK

**Windows 安装**：

1. **下载 JDK**：
   - 访问：https://adoptium.net/
   - 选择：**Temurin 17 (LTS)** > **Windows** > **x64** > **MSI Installer**
   - 下载文件：`OpenJDK17U-jdk_x64_windows_hotspot_17.0.x.msi`

2. **安装**：
   - 双击运行安装程序
   - 使用默认设置，一路"下一步"

3. **配置环境变量**：
   - 右键"此电脑" > "属性" > "高级系统设置" > "环境变量"
   - 在"系统变量"中点击"新建"：
     - 变量名：`JAVA_HOME`
     - 变量值：`C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot`（根据实际安装路径）
   - 编辑"Path"变量，添加：`%JAVA_HOME%\bin`

4. **验证安装**：
   - 打开新的 CMD 窗口
   - 运行：`java -version`
   - 应显示：`openjdk version "17.0.x"`

---

### 2. 安装 Apache Cassandra

**方式 A：使用 Docker（推荐）**

```powershell
# 拉取镜像
docker pull cassandra:latest

# 启动容器
docker run -d ^
    --name cassandra ^
    -p 9042:9042 ^
    -p 7000:7000 ^
    -p 7001:7001 ^
    -p 7199:7199 ^
    -v cassandra_data:/var/lib/cassandra ^
    cassandra:latest

# 等待启动（约 30 秒）
# 查看日志确认启动成功
docker logs -f cassandra
```

看到 `Listening for RPC clients` 表示启动成功。

**方式 B：直接安装**

1. 下载：https://cassandra.apache.org/download/
2. 解压到：`D:\apache-cassandra`
3. 运行：`D:\apache-cassandra\bin\assandra.bat`

---

### 3. 安装 Elasticsearch

**方式 A：使用 Docker（推荐）**

```powershell
# 拉取镜像
docker pull elasticsearch:8.11.0

# 启动容器
docker run -d ^
    --name elasticsearch ^
    -p 9200:9200 ^
    -p 9300:9300 ^
    -e "discovery.type=single-node" ^
    -e "xpack.security.enabled=false" ^
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" ^
    -v es_data:/usr/share/elasticsearch/data ^
    elasticsearch:8.11.0

# 等待启动（约 20 秒）
# 测试连接
curl http://localhost:9200
```

返回 JSON 响应表示成功。

**方式 B：直接安装**

1. 下载：https://www.elastic.co/downloads/elasticsearch
2. 解压到：`D:\elasticsearch`
3. 运行：`D:\elasticsearch\bin\elasticsearch.bat`

---

### 4. 下载 JanusGraph

```powershell
# 创建目录
cd D:\pyworkspace\WFWReshapingTranslation\janusgraph

# 下载 JanusGraph（选择版本）
# 访问：https://github.com/JanusGraph/janusgraph/releases
# 下载：janusgraph-1.0.0-hadoop2.zip

# 解压到当前目录
# 或使用以下命令（如果有 PowerShell 5.0+）
Invoke-WebRequest -Uri "https://github.com/JanusGraph/janusgraph/releases/download/v1.0.0/janusgraph-1.0.0-hadoop2.zip" -OutFile "janusgraph.zip"
Expand-Archive -Path janusgraph.zip -DestinationPath .
```

---

### 5. 启动 JanusGraph Server

```powershell
cd D:\pyworkspace\WFWReshapingTranslation\janusgraph\janusgraph-1.0.0

# 使用自定义配置启动
bin\janusgraph-server.bat conf\gremlin-server\gremlin-server-config.yaml
```

---

## ✅ 验证安装

### 测试 Cassandra 连接

```powershell
# 进入 Cassandra 容器（如果是 Docker）
docker exec -it cassandra cqlsh

# 或本地安装的
cqlsh

# 在 CQL shell 中运行
DESCRIBE KEYSPACES;
# 应该看到 system keyspace
```

### 测试 Elasticsearch 连接

```powershell
# 浏览器访问
http://localhost:9200/_cluster/health

# 或使用 curl
curl http://localhost:9200/_cluster/health
```

### 测试 JanusGraph

访问 Gremlin Console：

```powershell
cd D:\pyworkspace\WFWReshapingTranslation\janusgraph\janusgraph-1.0.0
bin\gremlin.bat
```

在 Gremlin Console 中运行：

```groovy
// 连接到 JanusGraph
:remote connect tinkerpop.server conf/remote.yaml session
:remote console

// 测试查询
graph = JanusGraphFactory.open('conf/janusgraph-cql-es.properties')
graph.tx().commit()

// 创建一个测试节点
person = graph.addVertex('person')
person.property('name', '测试用户')
graph.tx().commit()

// 查询
g.V().has('name', '测试用户').values('name')
```

---

## 🐳 Docker Compose 一键部署（推荐）

如果你已安装 Docker Desktop，可以使用以下 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  cassandra:
    image: cassandra:latest
    container_name: cassandra
    ports:
      - "9042:9042"
      - "7000:7000"
      - "7001:7001"
    volumes:
      - cassandra_data:/var/lib/cassandra
    environment:
      - MAX_HEAP_SIZE=512M
      - HEAP_NEWSIZE=100M

  elasticsearch:
    image: elasticsearch:8.11.0
    container_name: elasticsearch
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m

volumes:
  cassandra_data:
  es_data:
```

启动：

```powershell
cd D:\pyworkspace\WFWReshapingTranslation\janusgraph
docker-compose up -d
```

---

## 📝 下一步

安装完成后，我将继续为你创建：

1. ✨ **JanusGraph Python 客户端封装**
2. ✨ **知识图谱 Schema 定义（Gremlin）**
3. ✨ **预处理前端页面**
4. ✨ **对比分析 API**

请告诉我安装进度！
