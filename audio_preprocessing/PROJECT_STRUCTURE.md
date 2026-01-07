# 项目文件结构说明

## 📁 目录结构

```
audio_preprocessing/
├── audio_preprocessing_gpu.py   # GPU音频处理核心模块
├── api_server.py                # FastAPI Web服务器
├── requirements.txt             # Python依赖包列表
├── README_WEB.md               # Web版使用说明
├── PROJECT_STRUCTURE.md        # 本文档
├── start_web.bat               # Windows快速启动脚本
├── start_web.sh                # Linux/Mac快速启动脚本
├── templates/                  # 前端模板目录
│   └── index.html             # Web前端界面
└── temp_uploads/              # 临时文件存储目录（自动创建）
```

## 📄 文件说明

### 核心模块

#### audio_preprocessing_gpu.py
GPU加速音频处理的核心实现
- **类**: `GPUAudioProcessor` - GPU音频处理器
- **功能**:
  - 动态范围压缩（DRC）
  - GPU并行处理
  - 分块处理大文件
- **依赖**: PyTorch, pydub, numpy

#### api_server.py
FastAPI Web服务器
- **功能**:
  - RESTful API接口
  - 文件上传/下载
  - 后台任务处理
  - 健康检查
- **端口**: 8000
- **主要端点**:
  - `GET /` - Web界面
  - `POST /api/upload` - 上传音频
  - `POST /api/process/{filename}` - 处理音频
  - `GET /api/status/{task_id}` - 查询状态
  - `GET /api/download/{filename}` - 下载文件
  - `GET /api/health` - 健康检查
  - `DELETE /api/cleanup` - 清理临时文件

### 前端界面

#### templates/index.html
单页Web应用（SPA）
- **技术栈**: HTML5, Tailwind CSS, 原生JavaScript
- **功能**:
  - 拖拽上传音频文件
  - 实时参数调整
  - 进度条显示
  - 处理状态监控
  - 一键下载结果
- **特点**:
  - 响应式设计
  - 现代化UI
  - 无需前端框架

### 配置文件

#### requirements.txt
Python依赖包列表
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
torch>=2.1.0
torchaudio>=2.1.0
pydub>=0.25.1
numpy>=1.24.0
tqdm>=4.66.0
```

#### README_WEB.md
Web版使用说明文档
- 功能介绍
- 安装步骤
- 使用方法
- 参数说明
- API文档
- 常见问题

### 启动脚本

#### start_web.bat (Windows)
Windows快速启动脚本
- 自动检查Python
- 自动安装依赖
- 启动Web服务

#### start_web.sh (Linux/Mac)
Unix系统快速启动脚本
- 自动检查Python3
- 自动安装依赖
- 启动Web服务

## 🔄 工作流程

### 用户操作流程

1. **启动服务**
   ```bash
   # Windows
   start_web.bat

   # Linux/Mac
   chmod +x start_web.sh
   ./start_web.sh
   ```

2. **打开浏览器**
   - 访问 http://localhost:8000

3. **上传音频**
   - 点击上传区域或拖拽文件

4. **调整参数**
   - 压缩阈值
   - 压缩比
   - 启动时间
   - 释放时间

5. **开始处理**
   - 点击"开始处理"按钮
   - 等待处理完成

6. **下载结果**
   - 点击"下载输出文件"

### 技术流程

```
用户上传文件
    ↓
保存到 temp_uploads/
    ↓
创建后台任务
    ↓
GPU处理音频 (audio_preprocessing_gpu.py)
    ↓
保存到 temp_uploads/{name}_output.wav
    ↓
更新任务状态
    ↓
前端轮询状态
    ↓
显示下载按钮
    ↓
用户下载文件
```

## 🎯 设计原则

### SOLID原则应用

**单一职责（S）**
- `audio_preprocessing_gpu.py` - 仅负责音频处理
- `api_server.py` - 仅负责API服务
- `index.html` - 仅负责用户界面

**开闭原则（O）**
- API端点可扩展，无需修改核心代码
- 参数配置灵活，易于添加新参数

**依赖倒置（D）**
- 后端依赖抽象的ProcessRequest
- 前端通过API与后端解耦

### KISS原则
- 使用原生JavaScript，避免复杂框架
- 简洁的API设计
- 直观的用户界面

### DRY原则
- 统一的错误处理
- 复用的UI组件
- 共享的配置常量

### YAGNI原则
- 仅实现必要功能
- 避免过度设计
- 按需添加特性

## 🚀 扩展指南

### 添加新的音频处理功能

1. 在 `audio_preprocessing_gpu.py` 中添加新方法
2. 在 `api_server.py` 中添加对应API端点
3. 在 `index.html` 中添加UI控件

### 添加新的文件格式支持

修改 `api_server.py` 中的 `allowed_extensions`:
```python
allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.新格式'}
```

### 添加用户认证

在 `api_server.py` 中添加:
```python
from fastapi.security import APIKeyHeader

security = APIKeyHeader(name="X-API-Key")

@app.post("/api/upload", dependencies=[Depends(security)])
async def upload_audio(...):
    ...
```

## 📊 性能优化建议

1. **GPU利用率**
   - 调整分块大小（chunk_duration）
   - 监控显存使用
   - 批量处理多个文件

2. **并发处理**
   - 使用线程池处理多个任务
   - 限制同时处理数量
   - 实现任务队列

3. **缓存策略**
   - 缓存已处理的文件
   - 实现结果预览
   - 临时文件自动清理

## 🔒 安全建议

1. **文件验证**
   - 检查文件大小
   - 验证文件格式
   - 扫描恶意内容

2. **访问控制**
   - 添加API密钥
   - 实现用户认证
   - 限制访问频率

3. **数据保护**
   - 定期清理临时文件
   - 加密敏感数据
   - 日志审计

---

**项目遵循工程最佳实践，易于维护和扩展！** ✨
