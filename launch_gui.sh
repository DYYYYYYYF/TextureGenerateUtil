REM ===== Linux/Mac Shell脚本：launch_gui.sh =====
#!/bin/bash
# 顶点色转纹理工具 - Linux/Mac启动器

echo "=========================================="
echo "    顶点色转纹理工具 GUI启动器"
echo "=========================================="
echo

# 检查Python版本
if ! command -v python3 &> /dev/null; then
    echo "错误：未检测到Python3！"
    echo "请先安装Python 3.6或更高版本。"
    echo
    echo "安装方法："
    echo "  Ubuntu/Debian: sudo apt-get install python3"
    echo "  Fedora: sudo dnf install python3"
    echo "  macOS: brew install python3"
    exit 1
fi

echo "正在启动程序..."
echo

# 运行启动脚本
python3 launcher.py

if [ $? -ne 0 ]; then
    echo
    echo "程序运行出错，请查看上面的错误信息。"
    read -p "按回车键继续..."
fi