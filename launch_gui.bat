@echo off
REM ===== Windows批处理文件：启动GUI.bat =====
REM 顶点色转纹理工具 - Windows启动器

title 顶点色转纹理工具
echo ==========================================
echo     顶点色转纹理工具 GUI启动器
echo ==========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未检测到Python！
    echo 请先安装Python 3.6或更高版本。
    echo 下载地址：https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo 正在启动程序...
echo.

REM 运行启动脚本
python launcher.py

if %errorlevel% neq 0 (
    echo.
    echo 程序运行出错，请查看上面的错误信息。
    pause
)
