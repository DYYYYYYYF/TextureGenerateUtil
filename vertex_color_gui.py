#!/usr/bin/env python3
"""
顶点色转纹理贴图工具 - 图形界面版本
提供友好的用户界面进行模型转换
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import sys
import os
import json
import time
import queue
from pathlib import Path
import importlib.util

# 尝试导入主程序模块
try:
    from vertex_color_to_texture import VertexColorToTexture
    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False

class DependencyChecker:
    """依赖检查器"""
    
    REQUIRED_PACKAGES = {
        'trimesh': {
            'name': 'trimesh',
            'description': '3D网格处理库',
            'required': True,
            'install_cmd': 'pip install trimesh'
        },
        'PIL': {
            'name': 'pillow',
            'description': '图像处理库',
            'required': True,
            'install_cmd': 'pip install pillow'
        },
        'numpy': {
            'name': 'numpy',
            'description': '数值计算库',
            'required': True,
            'install_cmd': 'pip install numpy'
        },
        'numba': {
            'name': 'numba',
            'description': 'JIT编译加速',
            'required': False,
            'install_cmd': 'pip install numba'
        },
        'xatlas': {
            'name': 'xatlas',
            'description': 'UV展开库',
            'required': False,
            'install_cmd': 'pip install xatlas'
        }
    }
    
    @classmethod
    def check_dependency(cls, module_name):
        """检查单个依赖是否安装"""
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    
    @classmethod
    def check_all(cls):
        """检查所有依赖"""
        results = {}
        for module, info in cls.REQUIRED_PACKAGES.items():
            results[module] = {
                **info,
                'installed': cls.check_dependency(module)
            }
        return results

class ConsoleRedirector:
    """控制台输出重定向器"""
    
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag
        self.queue = queue.Queue()
        
    def write(self, content):
        self.queue.put((self.tag, content))
        
    def flush(self):
        pass

class VertexColorGUI:
    """主界面类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("顶点色转纹理贴图工具 v1.0")
        self.root.geometry("900x700")
        
        # 设置图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
            
        # 变量
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.texture_size = tk.IntVar(value=1024)
        self.num_workers = tk.IntVar(value=os.cpu_count())
        self.process_thread = None
        self.is_processing = False
        self.console_queue = queue.Queue()
        
        # 创建界面
        self.create_widgets()
        
        # 检查依赖
        self.check_dependencies()
        
        # 启动控制台输出更新
        self.update_console()
        
        # 设置关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """创建界面组件"""
        
        # 创建主容器
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        
        # 标题
        title_label = ttk.Label(main_container, text="GLB/GLTF 顶点色转纹理贴图工具", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # 创建标签页
        notebook = ttk.Notebook(main_container)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_container.rowconfigure(1, weight=1)
        
        # 主要功能页
        self.create_main_tab(notebook)
        
        # 依赖管理页
        self.create_dependency_tab(notebook)
        
        # 设置页
        self.create_settings_tab(notebook)
        
        # 关于页
        self.create_about_tab(notebook)
        
    def create_main_tab(self, notebook):
        """创建主功能标签页"""
        main_frame = ttk.Frame(notebook, padding="10")
        notebook.add(main_frame, text="主要功能")
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        # 输入文件
        ttk.Label(file_frame, text="输入文件:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.input_file, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_input_file).grid(row=0, column=2, padx=5)
        
        # 输出文件
        ttk.Label(file_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_output_file).grid(row=1, column=2, padx=5, pady=5)
        
        # 参数设置区域
        param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        param_frame.columnconfigure(1, weight=1)
        
        # 纹理大小
        ttk.Label(param_frame, text="纹理大小:").grid(row=0, column=0, sticky=tk.W, padx=5)
        size_frame = ttk.Frame(param_frame)
        size_frame.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        self.size_combo = ttk.Combobox(size_frame, textvariable=self.texture_size, 
                                       values=[256, 512, 1024, 2048, 4096], 
                                       state="readonly", width=10)
        self.size_combo.set(1024)
        self.size_combo.pack(side=tk.LEFT)
        ttk.Label(size_frame, text="像素").pack(side=tk.LEFT, padx=5)
        
        # 工作线程数
        ttk.Label(param_frame, text="工作线程数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        thread_frame = ttk.Frame(param_frame)
        thread_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.thread_scale = ttk.Scale(thread_frame, from_=1, to=os.cpu_count()*2, 
                                      variable=self.num_workers, orient=tk.HORIZONTAL, 
                                      length=200, command=self.update_thread_label)
        self.thread_scale.pack(side=tk.LEFT)
        self.thread_label = ttk.Label(thread_frame, text=f"{self.num_workers.get()}")
        self.thread_label.pack(side=tk.LEFT, padx=10)
        
        # 进度条
        progress_frame = ttk.LabelFrame(main_frame, text="进度", padding="10")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="就绪")
        self.status_label.grid(row=1, column=0, padx=5)
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="开始转换", command=self.start_conversion)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止", command=self.stop_conversion, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="清空日志", command=self.clear_console).pack(side=tk.LEFT, padx=5)
        
        # 日志输出区域
        console_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="10")
        console_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # 创建文本框和滚动条
        self.console_text = scrolledtext.ScrolledText(console_frame, height=10, wrap=tk.WORD)
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置文本标签
        self.console_text.tag_config("info", foreground="black")
        self.console_text.tag_config("warning", foreground="orange")
        self.console_text.tag_config("error", foreground="red")
        self.console_text.tag_config("success", foreground="green")
        
    def create_dependency_tab(self, notebook):
        """创建依赖管理标签页"""
        dep_frame = ttk.Frame(notebook, padding="10")
        notebook.add(dep_frame, text="依赖管理")
        
        # 说明
        info_label = ttk.Label(dep_frame, text="检查和安装所需的Python库", font=('Arial', 12))
        info_label.grid(row=0, column=0, pady=10)
        
        # 依赖列表框架
        list_frame = ttk.LabelFrame(dep_frame, text="依赖库状态", padding="10")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        dep_frame.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # 创建树形视图
        columns = ('状态', '描述', '操作')
        self.dep_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', height=10)
        self.dep_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置列
        self.dep_tree.heading('#0', text='库名称')
        self.dep_tree.heading('状态', text='状态')
        self.dep_tree.heading('描述', text='描述')
        self.dep_tree.heading('操作', text='安装命令')
        
        self.dep_tree.column('#0', width=150)
        self.dep_tree.column('状态', width=100)
        self.dep_tree.column('描述', width=200)
        self.dep_tree.column('操作', width=250)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.dep_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.dep_tree.configure(yscrollcommand=scrollbar.set)
        
        # 按钮框架
        btn_frame = ttk.Frame(dep_frame)
        btn_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(btn_frame, text="刷新检查", command=self.check_dependencies).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="一键安装必需库", command=self.install_required).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="安装所有库", command=self.install_all).pack(side=tk.LEFT, padx=5)
        
        # 镜像源选择
        mirror_frame = ttk.LabelFrame(dep_frame, text="PyPI镜像源", padding="10")
        mirror_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        mirror_frame.columnconfigure(1, weight=1)
        
        ttk.Label(mirror_frame, text="选择镜像:").grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.mirror_var = tk.StringVar(value="https://pypi.tuna.tsinghua.edu.cn/simple")
        mirrors = [
            ("官方源", "https://pypi.org/simple"),
            ("清华源", "https://pypi.tuna.tsinghua.edu.cn/simple"),
            ("阿里云", "https://mirrors.aliyun.com/pypi/simple/"),
            ("豆瓣源", "https://pypi.douban.com/simple/")
        ]
        
        for i, (name, url) in enumerate(mirrors):
            ttk.Radiobutton(mirror_frame, text=name, variable=self.mirror_var, 
                           value=url).grid(row=i//2, column=1+i%2, sticky=tk.W, padx=10, pady=2)
        
    def create_settings_tab(self, notebook):
        """创建设置标签页"""
        settings_frame = ttk.Frame(notebook, padding="10")
        notebook.add(settings_frame, text="设置")
        
        # 高级设置
        adv_frame = ttk.LabelFrame(settings_frame, text="高级设置", padding="10")
        adv_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        adv_frame.columnconfigure(1, weight=1)
        
        # 自动生成输出文件名
        self.auto_output = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_frame, text="自动生成输出文件名", 
                       variable=self.auto_output).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # 处理完成后打开文件夹
        self.open_folder = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_frame, text="处理完成后打开输出文件夹", 
                       variable=self.open_folder).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # 保存日志
        self.save_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv_frame, text="保存处理日志", 
                       variable=self.save_log).grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # UV展开方法
        uv_frame = ttk.LabelFrame(settings_frame, text="UV展开方法", padding="10")
        uv_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.uv_method = tk.StringVar(value="auto")
        methods = [
            ("自动选择", "auto"),
            ("简单投影", "simple"),
            ("XAtlas（如果可用）", "xatlas")
        ]
        
        for i, (name, value) in enumerate(methods):
            ttk.Radiobutton(uv_frame, text=name, variable=self.uv_method, 
                           value=value).grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # 性能设置
        perf_frame = ttk.LabelFrame(settings_frame, text="性能优化", padding="10")
        perf_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.use_numba = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="使用Numba JIT加速（如果可用）", 
                       variable=self.use_numba).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.use_cache = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="启用缓存", 
                       variable=self.use_cache).grid(row=1, column=0, sticky=tk.W, pady=5)
        
    def create_about_tab(self, notebook):
        """创建关于标签页"""
        about_frame = ttk.Frame(notebook, padding="20")
        notebook.add(about_frame, text="关于")
        
        # 标题
        title_label = ttk.Label(about_frame, text="顶点色转纹理贴图工具", 
                                font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        version_label = ttk.Label(about_frame, text="版本: 1.0.0", font=('Arial', 12))
        version_label.pack(pady=5)
        
        # 说明
        description = """
这是一个用于将GLB/GLTF模型中的顶点色转换为纹理贴图的工具。

主要功能：
• 自动提取顶点色信息
• 生成UV坐标（如果需要）
• 烘焙顶点色到纹理贴图
• 更新模型使用纹理材质

支持的格式：
• GLB (Binary glTF)
• GLTF (JSON glTF)

性能优化：
• 多线程并行处理
• Numba JIT编译加速
• 向量化运算优化

作者: UncleDon
许可: MIT License
        """
        
        desc_label = ttk.Label(about_frame, text=description, justify=tk.LEFT)
        desc_label.pack(pady=20)
        
        # 链接框架
        link_frame = ttk.Frame(about_frame)
        link_frame.pack(pady=10)
        
        ttk.Button(link_frame, text="查看源代码", 
                  command=lambda: self.open_url("https://github.com")).pack(side=tk.LEFT, padx=5)
        ttk.Button(link_frame, text="报告问题", 
                  command=lambda: self.open_url("https://github.com/issues")).pack(side=tk.LEFT, padx=5)
        
    def browse_input_file(self):
        """浏览输入文件"""
        filename = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[
                ("3D模型文件", "*.glb *.gltf"),
                ("GLB文件", "*.glb"),
                ("GLTF文件", "*.gltf"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.input_file.set(filename)
            
            # 自动生成输出文件名
            if self.auto_output.get():
                base_name = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]
                self.output_file.set(f"{base_name}_textured{ext}")
            
    def browse_output_file(self):
        """浏览输出文件"""
        filename = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".glb",
            filetypes=[
                ("GLB文件", "*.glb"),
                ("GLTF文件", "*.gltf"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.output_file.set(filename)
            
    def update_thread_label(self, value):
        """更新线程数标签"""
        self.thread_label.config(text=f"{int(float(value))}")
        
    def check_dependencies(self):
        """检查依赖"""
        self.dep_tree.delete(*self.dep_tree.get_children())
        
        results = DependencyChecker.check_all()
        
        for module, info in results.items():
            status = "✓ 已安装" if info['installed'] else "✗ 未安装"
            status_color = "green" if info['installed'] else "red"
            required = "必需" if info['required'] else "可选"
            
            item = self.dep_tree.insert('', 'end', text=f"{info['name']} ({required})",
                                        values=(status, info['description'], info['install_cmd']))
            
            # 设置颜色标签
            if info['installed']:
                self.dep_tree.item(item, tags=('installed',))
            else:
                self.dep_tree.item(item, tags=('not_installed',))
                
        # 配置标签颜色
        self.dep_tree.tag_configure('installed', foreground='green')
        self.dep_tree.tag_configure('not_installed', foreground='red')
        
    def install_required(self):
        """安装必需的库"""
        self.install_packages(required_only=True)
        
    def install_all(self):
        """安装所有库"""
        self.install_packages(required_only=False)
        
    def install_packages(self, required_only=True):
        """安装包"""
        packages = []
        for module, info in DependencyChecker.REQUIRED_PACKAGES.items():
            if not DependencyChecker.check_dependency(module):
                if not required_only or info['required']:
                    packages.append(info['name'])
                    
        if not packages:
            messagebox.showinfo("提示", "所有必需的库都已安装！")
            return
            
        # 构建安装命令
        mirror = self.mirror_var.get()
        cmd = f"{sys.executable} -m pip install {' '.join(packages)} -i {mirror}"
        
        # 确认对话框
        if messagebox.askyesno("确认安装", f"将要安装以下包:\n{', '.join(packages)}\n\n是否继续？"):
            self.log("info", f"执行命令: {cmd}")
            
            try:
                # 在新线程中执行安装
                def install():
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.log("success", "安装成功！")
                        self.root.after(0, self.check_dependencies)
                    else:
                        self.log("error", f"安装失败: {result.stderr}")
                        
                thread = threading.Thread(target=install)
                thread.start()
                
            except Exception as e:
                self.log("error", f"安装出错: {str(e)}")
                
    def start_conversion(self):
        """开始转换"""
        # 检查输入
        if not self.input_file.get():
            messagebox.showwarning("警告", "请选择输入文件！")
            return
            
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("错误", "输入文件不存在！")
            return
            
        # 检查必需的依赖
        if not CONVERTER_AVAILABLE:
            messagebox.showerror("错误", "转换器模块未找到！请确保vertex_color_to_texture.py在同一目录下。")
            return
            
        missing_deps = []
        for module, info in DependencyChecker.REQUIRED_PACKAGES.items():
            if info['required'] and not DependencyChecker.check_dependency(module):
                missing_deps.append(info['name'])
                
        if missing_deps:
            messagebox.showerror("错误", f"缺少必需的依赖库: {', '.join(missing_deps)}\n请先安装依赖。")
            return
            
        # 设置界面状态
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar.start()
        self.status_label.config(text="正在处理...")
        
        # 清空日志
        self.clear_console()
        
        # 在新线程中执行转换
        self.process_thread = threading.Thread(target=self.run_conversion)
        self.process_thread.start()
        
    def run_conversion(self):
        """运行转换（在子线程中）"""
        try:
            self.log("info", "="*50)
            self.log("info", "开始转换处理")
            self.log("info", f"输入文件: {self.input_file.get()}")
            self.log("info", f"输出文件: {self.output_file.get() or '自动生成'}")
            self.log("info", f"纹理大小: {self.texture_size.get()}x{self.texture_size.get()}")
            self.log("info", f"工作线程: {self.num_workers.get()}")
            self.log("info", "="*50)
            
            # 重定向输出
            import io
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            
            try:
                # 创建转换器
                converter = VertexColorToTexture(
                    input_path=self.input_file.get(),
                    output_path=self.output_file.get() if self.output_file.get() else None,
                    texture_size=self.texture_size.get(),
                    num_workers=self.num_workers.get()
                )
                
                # 执行转换
                converter.process()
                
                # 获取输出
                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()
                
                if stdout_content:
                    for line in stdout_content.split('\n'):
                        if line.strip():
                            self.log("info", line)
                            
                if stderr_content:
                    for line in stderr_content.split('\n'):
                        if line.strip():
                            self.log("warning", line)
                            
                self.log("success", "转换完成！")
                
                # 如果设置了打开文件夹
                if self.open_folder.get():
                    output_dir = os.path.dirname(self.output_file.get() or self.input_file.get())
                    if sys.platform == 'win32':
                        os.startfile(output_dir)
                    elif sys.platform == 'darwin':
                        subprocess.run(['open', output_dir])
                    else:
                        subprocess.run(['xdg-open', output_dir])
                        
            finally:
                # 恢复输出
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            self.log("error", f"转换失败: {str(e)}")
            import traceback
            self.log("error", traceback.format_exc())
            
        finally:
            # 恢复界面状态
            self.root.after(0, self.conversion_finished)
            
    def conversion_finished(self):
        """转换完成后的处理"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.status_label.config(text="就绪")
        
        # 保存日志
        if self.save_log.get():
            log_file = f"conversion_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(self.console_text.get(1.0, tk.END))
            self.log("info", f"日志已保存到: {log_file}")
            
    def stop_conversion(self):
        """停止转换"""
        if self.process_thread and self.process_thread.is_alive():
            # 注意：强制停止线程可能导致问题
            messagebox.showinfo("提示", "正在停止处理，请稍候...")
            self.is_processing = False
            
    def clear_console(self):
        """清空控制台"""
        self.console_text.delete(1.0, tk.END)
        
    def log(self, level, message):
        """写入日志"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # 将消息放入队列
        self.console_queue.put((level, formatted_message))
        
    def update_console(self):
        """更新控制台输出"""
        try:
            while True:
                level, message = self.console_queue.get_nowait()
                
                # 插入文本
                self.console_text.insert(tk.END, message, level)
                
                # 自动滚动到底部
                self.console_text.see(tk.END)
                
        except queue.Empty:
            pass
            
        # 定时更新
        self.root.after(100, self.update_console)
        
    def open_url(self, url):
        """打开URL"""
        import webbrowser
        webbrowser.open(url)
        
    def on_closing(self):
        """关闭窗口事件"""
        if self.is_processing:
            if messagebox.askokcancel("确认", "正在处理中，确定要退出吗？"):
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    """主函数"""
    root = tk.Tk()
    app = VertexColorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()