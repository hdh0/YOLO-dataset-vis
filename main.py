import re
import time
import json
import tkinter as tk

from collections import OrderedDict
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# 设置matplotlib中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    import matplotlib.font_manager as fm
    
    # 获取系统字体列表
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # 按优先级尝试设置中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    
    for font in chinese_fonts:
        if font in font_list:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        # 如果没有找到中文字体，使用默认字体并警告
        print("警告：未找到合适的中文字体，中文可能显示为方块")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False

# 初始化字体设置
setup_chinese_font()

def cv2_imread_unicode(file_path):
    """
    解决OpenCV读取中文路径图像的问题
    """
    try:
        # 方法1：使用numpy读取
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        # 将字节数据转换为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)
        # 解码图像
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"读取图像失败: {file_path}, 错误: {str(e)}")
        return None

class AnnotationVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO数据集可视化工具")
        self.root.geometry("1400x900")
        
        # 数据存储
        self.current_image_path = None
        self.current_label_path = None
        self.image_list = []
        self.current_index = 0
        self.label_map = {}
        self.show_boxes = True
        self.show_segments = True
        self.show_labels = True
        self.alpha = 0.3
        
        # 颜色配置
        self.colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 青色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (0, 128, 128),  # 深青色
            (128, 128, 0),  # 橄榄色
        ]
        
        # 新增：图像缓存 & 防抖
        self.image_cache = OrderedDict()
        self.cache_size = 30  # 可根据内存调节
        self.pending_update = None  # 防抖 after id
        self.scale_dragging = False  # 新增：是否正在拖动
        self.last_preview_time = 0   # 新增：上次快速预览时间戳
        
        self.setup_ui()
        self.load_default_label_map()
        
        # 键盘绑定
        self.root.bind('<Left>', lambda e: self.step_index(-1))
        self.root.bind('<Right>', lambda e: self.step_index(1))
        self.root.bind('<Control-Left>', lambda e: self.step_index(-10))
        self.root.bind('<Control-Right>', lambda e: self.step_index(10))
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 右侧显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(control_frame)
        self.setup_display_area(display_frame)
        
    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 文件选择区域
        file_frame = ttk.LabelFrame(parent, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="选择图像文件夹", 
                  command=self.select_image_folder).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="选择标签文件夹", 
                  command=self.select_label_folder).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="加载标签映射", 
                  command=self.load_label_map).pack(fill=tk.X, pady=2)
        
        # 导航区域
        nav_frame = ttk.LabelFrame(parent, text="图像导航", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill=tk.X)
        ttk.Button(nav_buttons, text="上一张", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_buttons, text="下一张", command=self.next_image).pack(side=tk.RIGHT)
        
        # 新增：跳转到指定图像
        jump_frame = ttk.Frame(nav_frame)
        jump_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(jump_frame, text="快速跳转:").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(jump_frame, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(jump_frame, text="Go", width=4, command=self.jump_to_image).pack(side=tk.LEFT)
        self.jump_entry.bind('<Return>', lambda e: self.jump_to_image())
        
        self.image_info_label = ttk.Label(nav_frame, text="未选择图像")
        self.image_info_label.pack(pady=5)
        
        # 新增：平滑滑块（实时切换）
        slider_frame = ttk.Frame(nav_frame)
        slider_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(slider_frame, text="快速切换:").pack(anchor=tk.W)
        self.index_var = tk.DoubleVar(value=0)
        self.index_scale = ttk.Scale(slider_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                     variable=self.index_var, command=self.on_scale_move)
        self.index_scale.pack(fill=tk.X)
        # 新增：绑定拖动按下/释放事件
        self.index_scale.bind('<ButtonPress-1>', lambda e: self.on_scale_press())
        self.index_scale.bind('<ButtonRelease-1>', lambda e: self.on_scale_release())
        self.scale_value_label = ttk.Label(slider_frame, text="0/0")
        self.scale_value_label.pack(anchor=tk.E)
        
        # 显示选项
        options_frame = ttk.LabelFrame(parent, text="显示选项", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.show_boxes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="显示检测框", 
                       variable=self.show_boxes_var,
                       command=self.update_display).pack(anchor=tk.W)
        
        self.show_segments_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="显示分割区域", 
                       variable=self.show_segments_var,
                       command=self.update_display).pack(anchor=tk.W)
        
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="显示标签", 
                       variable=self.show_labels_var,
                       command=self.update_display).pack(anchor=tk.W)
        
        # 透明度控制
        ttk.Label(options_frame, text="分割透明度:").pack(anchor=tk.W, pady=(10, 0))
        self.alpha_var = tk.DoubleVar(value=0.3)
        alpha_scale = ttk.Scale(options_frame, from_=0.0, to=1.0, 
                               variable=self.alpha_var, orient=tk.HORIZONTAL,
                               command=self.update_alpha)
        alpha_scale.pack(fill=tk.X)
        
        # 标签映射编辑
        label_frame = ttk.LabelFrame(parent, text="标签映射", padding=10)
        label_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 添加标签
        add_frame = ttk.Frame(label_frame)
        add_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(add_frame, text="ID:").pack(side=tk.LEFT)
        self.class_id_entry = ttk.Entry(add_frame, width=5)
        self.class_id_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(add_frame, text="名称:").pack(side=tk.LEFT)
        self.class_name_entry = ttk.Entry(add_frame, width=10)
        self.class_name_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(add_frame, text="添加", command=self.add_label).pack(side=tk.LEFT, padx=5)
        
        # 标签列表
        self.label_listbox = tk.Listbox(label_frame, height=8)
        self.label_listbox.pack(fill=tk.BOTH, expand=True)
        self.label_listbox.bind('<Delete>', self.delete_label)
        
        # # 保存/导出功能
        # save_frame = ttk.LabelFrame(parent, text="保存/导出", padding=10)
        # save_frame.pack(fill=tk.X)
        
        # ttk.Button(save_frame, text="保存当前图像", 
        #           command=self.save_current_image).pack(fill=tk.X, pady=2)
        # ttk.Button(save_frame, text="批量导出", 
        #           command=self.batch_export).pack(fill=tk.X, pady=2)
        # ttk.Button(save_frame, text="保存标签映射", 
        #           command=self.save_label_map).pack(fill=tk.X, pady=2)
        
    def setup_display_area(self, parent):
        """设置显示区域"""
        # 创建matplotlib图形
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("请选择图像文件夹和标签文件夹")
        self.ax.axis('off')
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
    def auto_detect_label_folder(self):
        """自动检测标签文件夹"""
        if not hasattr(self, 'image_folder'):
            return
            
        image_path = Path(self.image_folder)
        
        # 常见的标签文件夹模式
        possible_patterns = [
            # 如果图像在 images/train，标签在 labels/train
            str(image_path).replace('images', 'labels'),
            # 如果图像在 train，标签在对应的 labels/train
            str(image_path.parent / 'labels' / image_path.name),
            # 如果图像在某个文件夹，标签在同级的 labels 文件夹
            str(image_path.parent / 'labels'),
            # 如果图像在某个文件夹，标签在同一文件夹的 labels 子文件夹
            str(image_path / 'labels'),
        ]
        
        for pattern in possible_patterns:
            if Path(pattern).exists():
                self.label_folder = pattern
                print(f"自动检测到标签文件夹: {pattern}")
                messagebox.showinfo("自动检测", f"已自动检测到标签文件夹:\n{pattern}")
                return
        
        print("未能自动检测到标签文件夹，请手动选择")
        
    def select_image_folder(self):
        """选择图像文件夹"""
        folder = filedialog.askdirectory(title="选择图像文件夹")
        if folder:
            self.image_folder = folder
            self.auto_detect_label_folder()
            self.load_image_list()
            
    def select_label_folder(self):
        """选择标签文件夹"""
        folder = filedialog.askdirectory(title="选择标签文件夹")
        if folder:
            self.label_folder = folder
            if hasattr(self, 'image_list') and self.image_list:
                self.update_display()
                
    def load_image_list(self):
        """加载图像列表（自然数字顺序）"""
        if not hasattr(self, 'image_folder'):
            return
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.image_list = []
        for ext in extensions:
            self.image_list.extend(Path(self.image_folder).glob(f'*{ext}'))
            self.image_list.extend(Path(self.image_folder).glob(f'*{ext.upper()}'))
        # 自然排序：按文件名中的数字顺序
        def natural_key(p: Path):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.stem)]
        self.image_list = sorted(set(self.image_list), key=natural_key)
        self.current_index = 0
        if self.image_list:
            if hasattr(self, 'index_scale'):
                self.index_scale.config(from_=0, to=len(self.image_list) - 1)
                self.index_var.set(0)
                self.scale_value_label.config(text=f"1/{len(self.image_list)}")
            print(f"已加载 {len(self.image_list)} 张图像 (自然排序)")
            if hasattr(self, 'label_folder'):
                self.check_label_coverage()
            self.update_display()
        else:
            messagebox.showwarning("警告", "在选择的文件夹中没有找到图像文件")
    
    def check_label_coverage(self):
        """检查标签覆盖率"""
        if not hasattr(self, 'label_folder') or not self.image_list:
            return
            
        total_images = len(self.image_list)
        labeled_images = 0
        
        for image_path in self.image_list:
            label_name = image_path.stem + '.txt'
            label_path = Path(self.label_folder) / label_name
            if label_path.exists():
                labeled_images += 1
        
        coverage_rate = (labeled_images / total_images) * 100 if total_images > 0 else 0
        print(f"标签覆盖率: {labeled_images}/{total_images} ({coverage_rate:.1f}%)")
        
        if coverage_rate < 100:
            messagebox.showwarning("标签覆盖率", 
                                 f"标签覆盖率: {coverage_rate:.1f}%\n"
                                 f"已标注: {labeled_images} 张\n"
                                 f"总图像: {total_images} 张\n"
                                 f"缺失标签: {total_images - labeled_images} 张")
        else:
            messagebox.showinfo("标签覆盖率", "所有图像都有对应的标签文件！")
            
    def load_label_map(self):
        """加载标签映射文件"""
        file_path = filedialog.askopenfilename(
            title="选择标签映射文件",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.label_map = json.load(f)
                        # 确保键是整数
                        self.label_map = {int(k): v for k, v in self.label_map.items()}
                else:
                    # 文本格式: id:name
                    self.label_map = {}
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if ':' in line:
                                id_str, name = line.strip().split(':', 1)
                                self.label_map[int(id_str)] = name
                                
                self.update_label_listbox()
                self.update_display()
                messagebox.showinfo("成功", f"已加载 {len(self.label_map)} 个标签映射")
            except Exception as e:
                messagebox.showerror("错误", f"加载标签映射失败: {str(e)}")
                
    def load_default_label_map(self):
        """加载默认标签映射"""
        self.label_map = {
            0: "0",
            1: "1",
            2: "2",
            3: "3"
        }
        self.update_label_listbox()
        
    def update_label_listbox(self):
        """更新标签列表框"""
        self.label_listbox.delete(0, tk.END)
        for class_id, name in sorted(self.label_map.items()):
            self.label_listbox.insert(tk.END, f"{class_id}: {name}")
            
    def add_label(self):
        """添加新标签"""
        try:
            class_id = int(self.class_id_entry.get())
            class_name = self.class_name_entry.get().strip()
            
            if class_name:
                self.label_map[class_id] = class_name
                self.update_label_listbox()
                self.class_id_entry.delete(0, tk.END)
                self.class_name_entry.delete(0, tk.END)
                self.update_display()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的类别ID")
            
    def delete_label(self, event):
        """删除选中的标签"""
        selection = self.label_listbox.curselection()
        if selection:
            index = selection[0]
            item = self.label_listbox.get(index)
            class_id = int(item.split(':')[0])
            del self.label_map[class_id]
            self.update_label_listbox()
            self.update_display()
            
    def prev_image(self):
        """上一张图像"""
        self.step_index(-1)
            
    def next_image(self):
        """下一张图像"""
        self.step_index(1)

    # 新增：获取缓存图像
    def get_cached_image(self, path):
        key = str(path)
        if key in self.image_cache:
            img = self.image_cache[key]
            # LRU 访问更新顺序
            self.image_cache.move_to_end(key)
            return img
        img = cv2_imread_unicode(key)
        if img is not None:
            self.image_cache[key] = img
            if len(self.image_cache) > self.cache_size:
                self.image_cache.popitem(last=False)
        return img

    # 新增：预加载附近图像
    def preload_images(self, center, radius=3):
        if not self.image_list:
            return
        for i in range(center - radius, center + radius + 1):
            if 0 <= i < len(self.image_list):
                p = self.image_list[i]
                if str(p) not in self.image_cache:
                    self.get_cached_image(p)

    # 新增：开始拖动
    def on_scale_press(self):
        self.scale_dragging = True
        # 取消可能的延迟任务，立即进入快速预览模式
        if self.pending_update:
            self.root.after_cancel(self.pending_update)
            self.pending_update = None

    # 新增：结束拖动
    def on_scale_release(self):
        self.scale_dragging = False
        # 松开后进行一次完整渲染
        self.update_display(fast=False)

    # 修改：滑块移动回调（区分快速预览和完整渲染）
    def on_scale_move(self, val):
        if not self.image_list:
            return
        idx = int(float(val) + 0.5)
        idx = max(0, min(len(self.image_list) - 1, idx))
        if idx == self.current_index:
            return
        self.current_index = idx
        self.scale_value_label.config(text=f"{self.current_index+1}/{len(self.image_list)}")
        # 拖动中：快速预览（限制频率）
        if self.scale_dragging:
            now = time.time()
            if now - self.last_preview_time >= 0.05:  # 20 FPS 左右
                self.last_preview_time = now
                self.update_display(fast=True)
        else:
            # 非拖动，使用防抖进行完整渲染
            if self.pending_update:
                self.root.after_cancel(self.pending_update)
            self.pending_update = self.root.after(30, self.update_display)

    # 修改：步进时直接完整渲染
    def step_index(self, delta):
        if not self.image_list:
            return
        new_idx = max(0, min(len(self.image_list) - 1, self.current_index + delta))
        if new_idx != self.current_index:
            self.current_index = new_idx
            self.index_var.set(self.current_index)
            self.update_display(fast=False)

    # 修改：update_display 支持 fast 模式
    def update_display(self, fast=False):
        """更新显示
        fast=True 时仅显示图像（不绘制标注）以提高拖动流畅度"""
        if not self.image_list or not hasattr(self, 'label_folder'):
            return
        current_image = self.image_list[self.current_index]
        image = self.get_cached_image(current_image)
        if image is None:
            self.ax.clear()
            self.ax.set_title("无法读取图像")
            self.ax.axis('off')
            self.canvas.draw()
            self.pending_update = None
            return
        # 预加载相邻
        self.preload_images(self.current_index)
        h, w = image.shape[:2]
        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_title(f"{current_image.name} ({self.current_index + 1}/{len(self.image_list)})" + (" [预览]" if fast else ""))
        self.ax.axis('off')
        annotations = []
        if not fast:
            label_path = Path(self.label_folder) / f"{current_image.stem}.txt"  # 修复未定义
            annotations = self.read_yolo_annotations(label_path)
            for ann in annotations:
                class_id = ann['class_id']
                color = np.array(self.colors[class_id % len(self.colors)]) / 255.0
                label = self.label_map.get(class_id, f"class_{class_id}")
                if ann['type'] == 'bbox' and self.show_boxes_var.get():
                    x_center, y_center = ann['x_center'] * w, ann['y_center'] * h
                    width, height = ann['width'] * w, ann['height'] * h
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
                    self.ax.add_patch(rect)
                    if self.show_labels_var.get():
                        self.ax.text(x1, y1 - 5, label, color=color, fontsize=10,
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                elif ann['type'] == 'segment' and self.show_segments_var.get():
                    points = ann['points']
                    pixel_points = [(x * w, y * h) for x, y in points]
                    if len(pixel_points) >= 3:
                        polygon = plt.Polygon(pixel_points, closed=True,
                                              facecolor=color, alpha=self.alpha,
                                              edgecolor=color, linewidth=2)
                        self.ax.add_patch(polygon)
                        if self.show_labels_var.get() and pixel_points:
                            x, y = pixel_points[0]
                            self.ax.text(x, y - 5, label, color=color, fontsize=10,
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        # 信息标签
        if fast:
            info_text = f"图像: {current_image.name}\n快速预览中..."
        else:
            num_annotations = len(annotations)
            bbox_count = sum(1 for ann in annotations if ann['type'] == 'bbox')
            segment_count = sum(1 for ann in annotations if ann['type'] == 'segment')
            info_text = (f"图像: {current_image.name}\n总标注: {num_annotations}\n"
                         f"检测框: {bbox_count}, 分割: {segment_count}")
        self.image_info_label.config(text=info_text)
        self.canvas.draw()
        if hasattr(self, 'scale_value_label'):
            self.scale_value_label.config(text=f"{self.current_index+1}/{len(self.image_list)}")
        if not fast:
            self.pending_update = None

    # 新增：更新透明度回调
    def update_alpha(self, val=None):
        self.alpha = self.alpha_var.get()
        # 非快速模式更新
        self.update_display(fast=False)

    def read_yolo_annotations(self, label_path):
        """读取YOLO格式的标注"""
        annotations = []
        if not Path(label_path).exists():
            return annotations
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    if len(coords) == 4:  # 检测框格式
                        x_center, y_center, width, height = coords
                        annotations.append({
                            'type': 'bbox',
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
                    else:  # 分割格式
                        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                        annotations.append({
                            'type': 'segment',
                            'class_id': class_id,
                            'points': points
                        })
        return annotations
        
    def save_current_image(self):
        """保存当前可视化图像"""
        if not self.image_list:
            messagebox.showwarning("警告", "没有可保存的图像")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存图像",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg")]
        )
        
        if file_path:
            self.fig.savefig(file_path, bbox_inches='tight', dpi=300)
            messagebox.showinfo("成功", f"图像已保存到: {file_path}")
            
    def batch_export(self):
        """批量导出可视化图像"""
        if not self.image_list:
            messagebox.showwarning("警告", "没有可导出的图像")
            return
            
        output_dir = filedialog.askdirectory(title="选择输出文件夹")
        if not output_dir:
            return
            
        # 创建进度窗口
        progress_window = tk.Toplevel(self.root)
        progress_window.title("批量导出进度")
        progress_window.geometry("400x100")
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(self.image_list))
        progress_bar.pack(pady=20, padx=20, fill=tk.X)
        
        status_label = ttk.Label(progress_window, text="准备开始...")
        status_label.pack()
        
        def export_images():
            original_index = self.current_index
            
            for i, image_path in enumerate(self.image_list):
                self.current_index = i
                self.update_display()
                
                output_path = Path(output_dir) / f"{image_path.stem}_visualized.png"
                self.fig.savefig(output_path, bbox_inches='tight', dpi=300)
                
                progress_var.set(i + 1)
                status_label.config(text=f"正在处理: {image_path.name}")
                progress_window.update()
                
            self.current_index = original_index
            self.update_display()
            
            progress_window.destroy()
            messagebox.showinfo("完成", f"已导出 {len(self.image_list)} 张图像到 {output_dir}")
            
        # 在主线程中执行导出
        self.root.after(100, export_images)
        
    def save_label_map(self):
        """保存标签映射"""
        file_path = filedialog.asksaveasfilename(
            title="保存标签映射",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.label_map, f, ensure_ascii=False, indent=2)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for class_id, name in sorted(self.label_map.items()):
                            f.write(f"{class_id}:{name}\n")
                            
                messagebox.showinfo("成功", f"标签映射已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def jump_to_image(self):
        """根据输入的文件名（可不含扩展名）跳转到对应图像"""
        if not self.image_list:
            return
        name = self.jump_entry.get().strip()
        if not name:
            return
        stem = Path(name).stem  # 去掉可能的扩展名
        target_index = None
        for i, p in enumerate(self.image_list):
            if p.stem == stem:
                target_index = i
                break
        if target_index is None:
            # 尝试数字匹配（输入纯数字且存在）
            if name.isdigit():
                for i, p in enumerate(self.image_list):
                    if p.stem.isdigit() and int(p.stem) == int(name):
                        target_index = i
                        break
        if target_index is not None:
            self.current_index = target_index
            if hasattr(self, 'index_var'):
                self.index_var.set(target_index)
            self.update_display(fast=False)
        else:
            messagebox.showwarning("未找到", f"未找到图像: {name}")

def main():
    """主函数"""
    # 测试中文字体设置
    print("正在检查中文字体支持...")
    setup_chinese_font()
    print(f"当前字体设置: {plt.rcParams['font.sans-serif']}")
    
    # 创建GUI
    root = tk.Tk()
    app = AnnotationVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()