# 顶点色转纹理贴图工具 - 使用说明

## 简介

这是一个专业的3D模型处理工具，用于将GLB/GLTF格式模型中的顶点色信息转换为纹理贴图。工具提供了友好的图形界面，支持批量处理和多线程加速。

## 功能特点

- ✅ **自动UV展开**：智能生成UV坐标
- ✅ **高质量烘焙**：精确的顶点色到纹理转换
- ✅ **性能优化**：多线程处理，Numba JIT加速
- ✅ **友好界面**：直观的GUI操作界面
- ✅ **依赖管理**：自动检查和安装所需库
- ✅ **灵活配置**：可调节纹理大小、线程数等参数

## 文件说明

```
项目文件结构：
├── vertex_color_to_texture.py   # 核心转换模块
├── vertex_color_gui.py          # GUI界面模块
├── launcher.py                  # 启动器脚本
├── 启动GUI.bat                  # Windows快捷启动
├── launch_gui.sh               # Linux/Mac快捷启动
└── README.md                   # 本说明文档
```

## 快速开始

### Windows用户

1. **双击运行** `启动GUI.bat`
2. 程序会自动检查并安装必需的依赖
3. GUI界面启动后即可使用

### Linux/Mac用户

1. **打开终端**，进入程序目录
2. **添加执行权限**：`chmod +x launch_gui.sh`
3. **运行脚本**：`./launch_gui.sh`

### 手动运行

```bash
# 直接运行GUI
python vertex_color_gui.py

# 或使用启动器（会自动检查依赖）
python launcher.py

# 命令行模式（不使用GUI）
python vertex_color_to_texture.py input.glb -o output.glb -s 2048
```

## GUI界面使用指南

### 1. 主要功能页

- **选择输入文件**：点击"浏览"选择要处理的GLB/GLTF文件

- **设置输出文件**：可自动生成或手动指定

- 调整参数

  ：

  - 纹理大小：256/512/1024/2048/4096像素
  - 工作线程数：根据CPU核心数调整

- **开始转换**：点击"开始转换"按钮

- **查看日志**：底部实时显示处理进度

### 2. 依赖管理页

- **检查状态**：查看各依赖库安装情况
- **一键安装**：自动安装必需的库
- **镜像源选择**：支持清华源、阿里云等国内镜像

### 3. 设置页

- 高级设置

  ：

  - 自动生成输出文件名
  - 处理完成后打开文件夹
  - 保存处理日志

- **UV展开方法**：自动选择/简单投影/XAtlas

- **性能优化**：启用/禁用Numba加速

### 4. 关于页

查看版本信息和使用说明

## 依赖库说明

### 必需依赖

| 库名称  | 用途       | 安装命令              |
| ------- | ---------- | --------------------- |
| trimesh | 3D网格处理 | `pip install trimesh` |
| pillow  | 图像处理   | `pip install pillow`  |
| numpy   | 数值计算   | `pip install numpy`   |

### 可选依赖（提升性能）

| 库名称 | 用途                          | 安装命令             |
| ------ | ----------------------------- | -------------------- |
| numba  | JIT编译加速（5-10倍速度提升） | `pip install numba`  |
| xatlas | 专业UV展开                    | `pip install xatlas` |

### 安装所有依赖

```bash
# 使用清华源快速安装
pip install trimesh pillow numpy numba xatlas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 命令行使用

除了GUI界面，也支持命令行模式：

```bash
# 基本用法
python vertex_color_to_texture.py input.glb

# 指定输出文件
python vertex_color_to_texture.py input.glb -o output.glb

# 设置纹理大小为2048
python vertex_color_to_texture.py input.glb -s 2048

# 指定工作线程数
python vertex_color_to_texture.py input.glb -w 8

# 完整示例
python vertex_color_to_texture.py model.glb -o model_textured.glb -s 4096 -w 16
```

## 常见问题

### Q1: 提示缺少tkinter

**A**: tkinter通常随Python一起安装，如果缺失：

- Ubuntu/Debian: `sudo apt-get install python3-tk`
- Fedora: `sudo dnf install python3-tkinter`
- Windows/macOS: 重新安装Python，确保勾选tkinter

### Q2: 安装依赖失败

**A**: 尝试使用国内镜像源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名
```

### Q3: 处理速度慢

**A**:

1. 安装numba加速库：`pip install numba`
2. 增加工作线程数
3. 降低纹理分辨率（如使用1024而非4096）

### Q4: 生成的纹理有黑边

**A**: 程序已内置膨胀算法处理边缘，如果仍有问题：

1. 尝试增大纹理分辨率
2. 检查模型的顶点色是否完整

### Q5: UV坐标质量不佳

**A**: 安装xatlas获得更好的UV展开：

```bash
pip install xatlas
```

## 性能优化建议

1. **安装Numba**：可获得5-10倍的速度提升
2. **合理设置线程数**：通常设为CPU核心数的1-2倍
3. **批量处理**：将多个小模型合并处理更高效
4. **适当的纹理大小**：根据模型复杂度选择，避免过大

## 技术特性

- **多线程并行**：充分利用多核CPU
- **向量化运算**：NumPy优化的批量计算
- **JIT编译**：Numba即时编译关键函数
- **智能UV展开**：自动选择最佳投影方向
- **边缘膨胀**：避免纹理接缝问题

## 更新日志

### v1.0.0 (2024)

- 初始版本发布
- 支持GLB/GLTF格式
- 实现顶点色到纹理的转换
- 提供GUI界面
- 多线程和Numba加速支持

## 许可证

MIT License - 可自由使用、修改和分发

## 技术支持

如遇到问题，请检查：

1. Python版本是否 >= 3.6
2. 所有文件是否在同一目录
3. 必需的依赖是否已安装
4. 输入的模型文件是否有效

------

*本工具持续更新优化中，欢迎反馈使用体验*