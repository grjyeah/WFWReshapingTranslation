@echo off
REM JanusGraph 知识图谱服务启动脚本

echo ====================================
echo JanusGraph 知识图谱服务启动
echo ====================================
echo.

REM 检查 Docker 是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker 未运行，请先启动 Docker Desktop
    pause
    exit /b 1
)

echo [1/3] 启动 Cassandra、Elasticsearch 和 JanusGraph...
docker-compose up -d

echo.
echo [2/3] 等待服务启动（约 1 分钟）...
timeout /t 60 /nobreak >nul

echo.
echo [3/3] 验证服务状态...
echo.

REM 检查 Cassandra
echo 检查 Cassandra...
docker exec meeting_cassandra cqlsh -e "describe cluster" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Cassandra 运行正常
) else (
    echo [WARN] Cassandra 可能未完全启动
)

REM 检查 Elasticsearch
echo 检查 Elasticsearch...
curl -s http://localhost:9200/_cluster/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Elasticsearch 运行正常
) else (
    echo [WARN] Elasticsearch 可能未完全启动
)

REM 检查 JanusGraph
echo 检查 JanusGraph...
curl -s http://localhost:8182 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] JanusGraph 运行正常
) else (
    echo [WARN] JanusGraph 可能未完全启动
)

echo.
echo ====================================
echo 服务启动完成！
echo ====================================
echo.
echo 服务地址:
echo   - Cassandra:     localhost:9042
echo   - Elasticsearch: localhost:9200
echo   - JanusGraph:    localhost:8182 (Gremlin WebSocket)
echo.
echo 访问 Neo4j Browser 的替代品:
echo   - 访问 http://localhost:8182 进行 Gremlin 查询
echo   - 或使用 Python 客户端连接
echo.
echo 停止服务: docker-compose down
echo 查看日志: docker-compose logs -f
echo.
pause
