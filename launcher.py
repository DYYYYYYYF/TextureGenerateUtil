#!/usr/bin/env python3
"""
顶点色转纹理工具 - 启动器
自动检查依赖并启动GUI界面
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_and_install_dependencies():
    """检查并安装基础依赖"""
    
    # 基础依赖
    basic_deps = {
        'tkinter': 'tkinter（通常已内置）'
    }
    
    # 检查tkinter
    try:
        import tkinter
        print("✓ tkinter已安装")
    except ImportError:
        print("✗ tkinter未安装")
        print("\ntkinter通常随Python一起安装。")
        print("如果未安装，请根据你的系统执行：")
        print("  Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  Fedora: sudo dnf install python3-tkinter")
        print("  macOS: tkinter应该已经包含在Python中")
        print("  Windows: tkinter应该已经包含在Python中")
        return False
        
    return True

def check_files():
    """检查必需的文件是否存在"""
    required_files = [
        'vertex_color_to_texture.py',
        'vertex_color_gui.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            
    if missing_files:
        print(f"\n错误：缺少必需的文件：")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保所有文件都在同一目录下。")
        return False
        
    return True

def install_basic_dependencies():
    """安装基础Python依赖"""
    print("\n正在检查并安装基础依赖...")
    
    # 使用清华源安装基础包
    packages = ['pillow', 'numpy']
    mirror = 'https://pypi.tuna.tsinghua.edu.cn/simple'
    
    for package in packages:
        try:
            __import__(package if package != 'pillow' else 'PIL')
            print(f"✓ {package}已安装")
        except ImportError:
            print(f"正在安装{package}...")
            cmd = f"{sys.executable} -m pip install {package} -i {mirror}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {package}安装成功")
            else:
                print(f"✗ {package}安装失败")
                print(f"  请手动运行: {cmd}")
                return False
                
    return True

def start_gui():
    """启动GUI界面"""
    try:
        # 导入并运行GUI
        from vertex_color_gui import main
        main()
    except ImportError as e:
        print(f"\n导入GUI模块失败: {e}")
        print("请检查vertex_color_gui.py文件是否存在且无错误。")
        return False
    except Exception as e:
        print(f"\n启动GUI失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def show_error_dialog(title, message):
    """显示错误对话框"""
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except:
        print(f"\n{title}: {message}")

def main():
    """主函数"""
    print("="*50)
    print("顶点色转纹理工具 - 启动器")
    print("="*50)
    
    # 检查Python版本
    if sys.version_info < (3, 6):
        error_msg = f"需要Python 3.6或更高版本，当前版本: {sys.version}"
        show_error_dialog("Python版本错误", error_msg)
        sys.exit(1)
        
    # 检查文件
    if not check_files():
        show_error_dialog("文件缺失", "缺少必需的程序文件，请检查文件完整性。")
        sys.exit(1)
        
    # 检查基础依赖
    if not check_and_install_dependencies():
        show_error_dialog("依赖错误", "无法安装必需的依赖，请查看控制台输出。")
        sys.exit(1)
        
    # 安装基础Python包
    if not install_basic_dependencies():
        show_error_dialog("安装失败", "无法安装基础Python包，请手动安装。")
        sys.exit(1)
        
    print("\n正在启动GUI界面...")
    
    # 启动GUI
    if not start_gui():
        show_error_dialog("启动失败", "无法启动GUI界面，请查看控制台错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()