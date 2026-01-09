# 静态资源目录

此目录用于存放Web界面所需的本地静态资源，确保在断网环境下也能正常使用。

## 文件说明

- **tailwind.js**: Tailwind CSS框架的本地化版本（约398KB）
  - 原始来源：https://cdn.tailwindcss.com
  - 已下载到本地，无需网络连接
  - 提供完整的Tailwind CSS功能

## 添加新的静态资源

如需添加其他CSS或JavaScript库：

1. 将文件放入此目录
2. 在 `templates/index.html` 中引用：
   ```html
   <script src="/static/your-file.js"></script>
   <link rel="stylesheet" href="/static/your-file.css">
   ```

## 服务配置

静态文件通过FastAPI的StaticFiles中间件提供服务（在 `api_server.py` 中配置）：

```python
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
```

访问路径：`http://localhost:8001/static/文件名`
