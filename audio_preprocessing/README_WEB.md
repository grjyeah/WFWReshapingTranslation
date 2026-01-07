# GPU音频处理工具 - Web界面

基于RTX 5090的GPU加速音频动态范围压缩工具，提供友好的Web界面。

## ✨ 特性

- 🚀 **GPU加速**: 使用PyTorch在RTX 5090上实现高速并行处理
- 🎨 **现代界面**: 响应式设计，美观易用
- 📊 **实时反馈**: 处理进度实时显示
- 🔧 **参数可调**: 支持自定义压缩参数
- 📁 **便捷操作**: 支持拖拽上传，一键下载

## 📋 系统要求

- Python 3.8+
- CUDA 11.8+ (GPU加速)
- RTX 5090 或兼容GPU

## 🛠️ 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 安装ffmpeg (音频处理必需)

**Windows:**
```bash
# 使用 Chocolatey
choco install ffmpeg

# 或从官网下载: https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## 🚀 使用方法

### 启动服务器

```bash
cd audio_preprocessing
python api_server.py
```

服务器将在 `http://localhost:8000` 启动。

### 访问Web界面

在浏览器中打开:
- 主界面: `http://localhost:8000` (需要配置静态文件服务)
- API文档: `http://localhost:8000/docs`

### 操作步骤

1. **上传音频**: 点击上传区域或拖拽音频文件
2. **调整参数**: 根据需要调整压缩参数
3. **开始处理**: 点击"开始处理"按钮
4. **下载结果**: 处理完成后下载输出文件

## 🎛️ 参数说明

### 压缩阈值 (Threshold)
- **范围**: -60dB ~ 0dB
- **默认**: -20dB
- **说明**: 超过此阈值的音量将被压缩

### 压缩比 (Ratio)
- **范围**: 1:1 ~ 20:1
- **默认**: 4:1
- **说明**: 控制压缩的强度
  - 1:1 = 无压缩
  - 4:1 = 适中压缩
  - 20:1 = 极限压缩

### 启动时间 (Attack)
- **范围**: 0.1ms ~ 50ms
- **默认**: 5ms
- **说明**: 压缩器启动的响应时间

### 释放时间 (Release)
- **范围**: 10ms ~ 500ms
- **默认**: 50ms
- **说明**: 压缩器恢复的响应时间

## 📁 输出说明

- **位置**: 与输入文件相同目录
- **命名**: 自动添加 `_output` 后缀
- **格式**: WAV (16-bit PCM)
- **示例**: `input.wav` → `input_output.wav`

## 🔧 API接口

### 健康检查
```
GET /api/health
```

### 上传文件
```
POST /api/upload
Content-Type: multipart/form-data
```

### 处理音频
```
POST /api/process/{filename}
Content-Type: application/json

{
  "threshold": -20.0,
  "ratio": 4.0,
  "attack": 5.0,
  "release": 50.0
}
```

### 查询状态
```
GET /api/status/{task_id}
```

### 下载文件
```
GET /api/download/{filename}
```

## 📝 开发说明

### 项目结构

```
audio_preprocessing/
├── api_server.py              # FastAPI服务器
├── audio_preprocessing_gpu.py # GPU处理核心
├── templates/
│   └── index.html            # Web前端界面
├── temp_uploads/             # 临时文件目录
├── requirements.txt          # 依赖列表
└── README_WEB.md            # 本文档
```

### 扩展开发

如需添加新功能:

1. 在 `api_server.py` 中添加新的API端点
2. 在 `index.html` 中添加对应的前端逻辑
3. 更新API文档

## ❓ 常见问题

### Q: GPU不可用怎么办？
A: 程序会自动降级到CPU模式，处理速度会较慢。

### Q: 支持哪些音频格式？
A: WAV, MP3, FLAC, M4A, AAC, OGG

### Q: 处理时间大概多久？
A: RTX 5090上可实现10-20x实时速度，60分钟音频约需3-6分钟。

### Q: 输出文件在哪里？
A: 默认在服务器 `temp_uploads` 目录，也可通过界面下载。

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请提交Issue。

---

**享受GPU加速的音频处理体验！** 🎵⚡
