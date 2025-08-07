#!/usr/bin/env python3
"""
GLB/GLTF顶点色转纹理贴图工具 - 优化版
使用多线程、向量化操作和JIT编译加速处理
"""

import numpy as np
import trimesh
import sys
import os
from PIL import Image
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time

# 尝试导入numba进行JIT加速
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    print("提示: 安装numba可获得更快的处理速度: pip install numba")
    NUMBA_AVAILABLE = False
    # 定义dummy装饰器
    def njit(parallel=False, cache=False):
        def decorator(func):
            return func
        return decorator
    jit = njit
    prange = range

# NumPy优化设置
np.seterr(divide='ignore', invalid='ignore')

class VertexColorToTexture:
    def __init__(self, input_path, output_path=None, texture_size=1024, num_workers=None):
        """
        初始化转换器
        
        Args:
            input_path: 输入的GLB/GLTF文件路径
            output_path: 输出路径，默认为输入文件名_textured
            texture_size: 生成纹理的尺寸，默认1024x1024
            num_workers: 工作线程数，默认为CPU核心数
        """
        self.input_path = input_path
        self.texture_size = texture_size
        self.num_workers = num_workers or mp.cpu_count()
        
        # 设置输出路径
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            ext = os.path.splitext(input_path)[1]
            self.output_path = f"{base_name}_textured{ext}"
        else:
            self.output_path = output_path
            
        # 输出目录
        self.output_dir = os.path.dirname(self.output_path)
        if not self.output_dir:
            self.output_dir = "."
            
        print(f"使用 {self.num_workers} 个工作线程")
        if NUMBA_AVAILABLE:
            print("Numba JIT加速已启用")
            
    def load_mesh(self):
        """加载GLB/GLTF文件"""
        print(f"加载模型: {self.input_path}")
        
        # 使用trimesh加载
        self.scene = trimesh.load(self.input_path, force='scene', process=False)
        
        if isinstance(self.scene, trimesh.Scene):
            # 如果是场景，获取所有网格
            self.meshes = []
            for name, geom in self.scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    self.meshes.append((name, geom))
                    print(f"  找到网格: {name}")
                    print(f"    顶点数: {len(geom.vertices)}")
                    print(f"    面数: {len(geom.faces)}")
        else:
            # 单个网格
            self.meshes = [("default", self.scene)]
            
        print(f"共找到 {len(self.meshes)} 个网格")
        
    def get_or_generate_uv_coordinates(self, mesh):
        """获取或生成UV坐标（向量化版本）"""
        uv_coords = None
        
        # 检查现有UV坐标
        if hasattr(mesh, 'visual'):
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                uv_coords = mesh.visual.uv
                print("  使用现有UV坐标")
                
        if uv_coords is None or len(uv_coords) == 0:
            print("  生成新的UV坐标...")
            uv_coords = self.generate_uv_coordinates_fast(mesh.vertices)
            
        return uv_coords
        
    @staticmethod
    @njit(parallel=True, cache=True)
    def generate_uv_coordinates_numba(vertices):
        """使用Numba加速的UV坐标生成"""
        n_vertices = len(vertices)
        uv_coords = np.zeros((n_vertices, 2), dtype=np.float32)
        
        # 计算边界
        min_x = vertices[:, 0].min()
        max_x = vertices[:, 0].max()
        min_y = vertices[:, 1].min()
        max_y = vertices[:, 1].max()
        min_z = vertices[:, 2].min()
        max_z = vertices[:, 2].max()
        
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_z = max_z - min_z
        
        # 避免除零
        if range_x == 0:
            range_x = 1.0
        if range_y == 0:
            range_y = 1.0
        if range_z == 0:
            range_z = 1.0
            
        # 选择最大的两个维度
        ranges = np.array([range_x, range_y, range_z])
        max_idx = np.argmax(ranges)
        ranges[max_idx] = -1
        second_idx = np.argmax(ranges)
        
        # 并行生成UV坐标
        for i in prange(n_vertices):
            if max_idx == 0:  # X最大
                if second_idx == 1:  # XY投影
                    uv_coords[i, 0] = (vertices[i, 0] - min_x) / range_x
                    uv_coords[i, 1] = (vertices[i, 1] - min_y) / range_y
                else:  # XZ投影
                    uv_coords[i, 0] = (vertices[i, 0] - min_x) / range_x
                    uv_coords[i, 1] = (vertices[i, 2] - min_z) / range_z
            elif max_idx == 1:  # Y最大
                if second_idx == 0:  # YX投影
                    uv_coords[i, 0] = (vertices[i, 1] - min_y) / range_y
                    uv_coords[i, 1] = (vertices[i, 0] - min_x) / range_x
                else:  # YZ投影
                    uv_coords[i, 0] = (vertices[i, 1] - min_y) / range_y
                    uv_coords[i, 1] = (vertices[i, 2] - min_z) / range_z
            else:  # Z最大
                if second_idx == 0:  # ZX投影
                    uv_coords[i, 0] = (vertices[i, 2] - min_z) / range_z
                    uv_coords[i, 1] = (vertices[i, 0] - min_x) / range_x
                else:  # ZY投影
                    uv_coords[i, 0] = (vertices[i, 2] - min_z) / range_z
                    uv_coords[i, 1] = (vertices[i, 1] - min_y) / range_y
                    
        return uv_coords
        
    def generate_uv_coordinates_fast(self, vertices):
        """快速生成UV坐标（向量化版本）"""
        if NUMBA_AVAILABLE:
            return self.generate_uv_coordinates_numba(vertices.astype(np.float32))
            
        # NumPy向量化版本
        vertices = np.asarray(vertices, dtype=np.float32)
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        range_bounds = max_bounds - min_bounds
        
        # 避免除零
        range_bounds[range_bounds == 0] = 1.0
        
        # 选择最大的两个维度
        axes = np.argsort(range_bounds)[-2:]
        
        # 向量化计算UV坐标
        uv_coords = np.zeros((len(vertices), 2), dtype=np.float32)
        uv_coords[:, 0] = (vertices[:, axes[0]] - min_bounds[axes[0]]) / range_bounds[axes[0]]
        uv_coords[:, 1] = (vertices[:, axes[1]] - min_bounds[axes[1]]) / range_bounds[axes[1]]
        
        return uv_coords
        
    def get_vertex_colors(self, mesh):
        """获取顶点颜色"""
        vertex_colors = None
        
        if hasattr(mesh, 'visual'):
            if hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors
                
        if vertex_colors is None:
            print("  警告：未找到顶点色，使用默认白色")
            vertex_colors = np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255
        else:
            # 确保颜色值在0-255范围内
            if vertex_colors.max() <= 1.0:
                vertex_colors = (vertex_colors * 255).astype(np.uint8)
            else:
                vertex_colors = vertex_colors.astype(np.uint8)
                
        # 确保有Alpha通道
        if vertex_colors.shape[1] == 3:
            alpha = np.ones((vertex_colors.shape[0], 1), dtype=np.uint8) * 255
            vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)
            
        return vertex_colors
        
    @staticmethod
    @njit(parallel=True, cache=True)
    def rasterize_triangles_numba(pixels, weights, faces, uv_coords, vertex_colors, texture_size):
        """使用Numba加速的三角形光栅化"""
        n_faces = len(faces)
        
        for face_idx in prange(n_faces):
            face = faces[face_idx]
            
            # 获取三角形顶点
            v0_idx, v1_idx, v2_idx = face[0], face[1], face[2]
            
            # UV坐标转像素坐标
            u0 = uv_coords[v0_idx, 0] * (texture_size - 1)
            v0 = uv_coords[v0_idx, 1] * (texture_size - 1)
            u1 = uv_coords[v1_idx, 0] * (texture_size - 1)
            v1 = uv_coords[v1_idx, 1] * (texture_size - 1)
            u2 = uv_coords[v2_idx, 0] * (texture_size - 1)
            v2 = uv_coords[v2_idx, 1] * (texture_size - 1)
            
            # 计算边界框
            min_x = max(0, int(min(u0, u1, u2)))
            max_x = min(texture_size - 1, int(max(u0, u1, u2)) + 1)
            min_y = max(0, int(min(v0, v1, v2)))
            max_y = min(texture_size - 1, int(max(v0, v1, v2)) + 1)
            
            # 预计算边缘函数的系数
            dx01 = u1 - u0
            dy01 = v1 - v0
            dx12 = u2 - u1
            dy12 = v2 - v1
            dx20 = u0 - u2
            dy20 = v0 - v2
            
            # 三角形面积的两倍
            area2 = dx01 * dy20 - dy01 * dx20
            
            if abs(area2) < 0.01:
                continue
                
            inv_area2 = 1.0 / area2
            
            # 光栅化
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # 计算重心坐标
                    px = float(x)
                    py = float(y)
                    
                    w0 = (dx12 * (py - v1) - dy12 * (px - u1)) * inv_area2
                    w1 = (dx20 * (py - v2) - dy20 * (px - u2)) * inv_area2
                    w2 = 1.0 - w0 - w1
                    
                    # 检查是否在三角形内
                    if w0 >= -0.01 and w1 >= -0.01 and w2 >= -0.01:
                        # 确保权重归一化
                        w0 = max(0.0, w0)
                        w1 = max(0.0, w1)
                        w2 = max(0.0, w2)
                        sum_w = w0 + w1 + w2
                        if sum_w > 0:
                            w0 /= sum_w
                            w1 /= sum_w
                            w2 /= sum_w
                            
                            # 插值颜色
                            for c in range(4):
                                color = (vertex_colors[v0_idx, c] * w0 + 
                                       vertex_colors[v1_idx, c] * w1 + 
                                       vertex_colors[v2_idx, c] * w2)
                                pixels[y, x, c] = pixels[y, x, c] * weights[y, x] + color
                                
                            weights[y, x] += 1.0
                            
        return pixels, weights
        
    def rasterize_triangles_batch(self, faces, uv_coords, vertex_colors):
        """批量光栅化三角形（向量化版本）"""
        texture_size = self.texture_size
        pixels = np.zeros((texture_size, texture_size, 4), dtype=np.float64)
        weights = np.zeros((texture_size, texture_size), dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            # 使用Numba加速版本
            pixels, weights = self.rasterize_triangles_numba(
                pixels, weights,
                faces.astype(np.int32),
                uv_coords.astype(np.float32),
                vertex_colors.astype(np.float32),
                texture_size
            )
        else:
            # 使用多线程处理
            n_faces = len(faces)
            batch_size = max(1, n_faces // self.num_workers)
            
            def process_batch(start_idx, end_idx):
                local_pixels = np.zeros((texture_size, texture_size, 4), dtype=np.float64)
                local_weights = np.zeros((texture_size, texture_size), dtype=np.float64)
                
                for face_idx in range(start_idx, min(end_idx, n_faces)):
                    face = faces[face_idx]
                    tri_uvs = uv_coords[face]
                    tri_colors = vertex_colors[face]
                    
                    # 转换UV坐标到像素坐标
                    tri_pixels = tri_uvs * (texture_size - 1)
                    
                    # 光栅化单个三角形
                    self.rasterize_single_triangle(
                        local_pixels, local_weights,
                        tri_pixels, tri_colors
                    )
                    
                return local_pixels, local_weights
                
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, n_faces, batch_size):
                    futures.append(executor.submit(process_batch, i, i + batch_size))
                    
                # 合并结果
                for future in futures:
                    local_pixels, local_weights = future.result()
                    pixels += local_pixels
                    weights += local_weights
                    
        # 归一化权重
        mask = weights > 0
        pixels[mask] = pixels[mask] / weights[mask, np.newaxis]
        
        return pixels.astype(np.uint8)
        
    def rasterize_single_triangle(self, pixels, weights, tri_pixels, tri_colors):
        """光栅化单个三角形（优化版本）"""
        # 使用向量化的边界框计算
        min_bound = np.maximum(0, np.floor(tri_pixels.min(axis=0)).astype(int))
        max_bound = np.minimum(self.texture_size - 1, np.ceil(tri_pixels.max(axis=0)).astype(int))
        
        # 创建网格点
        x_range = np.arange(min_bound[0], max_bound[0] + 1)
        y_range = np.arange(min_bound[1], max_bound[1] + 1)
        xx, yy = np.meshgrid(x_range, y_range)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        if len(points) == 0:
            return
            
        # 批量计算重心坐标
        v0 = tri_pixels[2] - tri_pixels[0]
        v1 = tri_pixels[1] - tri_pixels[0]
        v2 = points - tri_pixels[0]
        
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return
            
        inv_denom = 1 / denom
        
        dot02 = np.sum(v2 * v0, axis=1)
        dot12 = np.sum(v2 * v1, axis=1)
        
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1 - u - v
        
        # 找出在三角形内的点
        inside = (u >= -0.01) & (v >= -0.01) & (w >= -0.01) & (u <= 1.01) & (v <= 1.01) & (w <= 1.01)
        valid_points = points[inside]
        
        if len(valid_points) == 0:
            return
            
        valid_u = np.clip(u[inside], 0, 1)
        valid_v = np.clip(v[inside], 0, 1)
        valid_w = np.clip(w[inside], 0, 1)
        
        # 归一化权重
        weight_sum = valid_u + valid_v + valid_w
        valid_u /= weight_sum
        valid_v /= weight_sum
        valid_w /= weight_sum
        
        # 插值颜色
        bary = np.stack([valid_w, valid_v, valid_u], axis=1)
        interpolated_colors = np.dot(bary, tri_colors)
        
        # 更新像素
        for i, point in enumerate(valid_points):
            x, y = int(point[0]), int(point[1])
            pixels[y, x] = pixels[y, x] * weights[y, x] + interpolated_colors[i]
            weights[y, x] += 1.0
            
    def bake_vertex_colors_to_texture(self, mesh, uv_coords):
        """烘焙顶点色到纹理（优化版本）"""
        print("  烘焙顶点色到纹理...")
        start_time = time.time()
        
        # 获取数据
        vertex_colors = self.get_vertex_colors(mesh)
        faces = mesh.faces
        
        # 批量光栅化
        pixels = self.rasterize_triangles_batch(faces, uv_coords, vertex_colors)
        
        # 创建图像
        texture_img = Image.fromarray(pixels, mode='RGBA')
        
        # 膨胀填充空隙
        texture_img = self.dilate_texture_fast(texture_img)
        
        elapsed = time.time() - start_time
        print(f"    烘焙完成，耗时: {elapsed:.2f}秒")
        
        return texture_img
        
    def dilate_texture_fast(self, texture_img, iterations=3):
        """快速膨胀纹理（向量化版本）"""
        pixels = np.array(texture_img, dtype=np.uint8)
        
        # 创建卷积核
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.float32) / 8
        
        for _ in range(iterations):
            # 找出需要填充的像素
            mask = pixels[:, :, 3] < 128
            
            if not np.any(mask):
                break
                
            # 使用卷积进行膨胀
            new_pixels = pixels.copy()
            
            for c in range(4):
                # 简单的卷积实现
                padded = np.pad(pixels[:, :, c], 1, mode='edge')
                for y in range(self.texture_size):
                    for x in range(self.texture_size):
                        if mask[y, x]:
                            neighborhood = padded[y:y+3, x:x+3]
                            alpha_neighborhood = np.pad(pixels[:, :, 3], 1, mode='edge')[y:y+3, x:x+3]
                            
                            # 只考虑不透明的邻居
                            valid_mask = alpha_neighborhood >= 128
                            if np.any(valid_mask):
                                new_pixels[y, x, c] = np.mean(neighborhood[valid_mask])
                                
            # 更新Alpha通道
            new_pixels[mask, 3] = np.where(
                np.sum(new_pixels[mask, :3], axis=1) > 0,
                255,
                0
            )
            
            pixels = new_pixels
            
        return Image.fromarray(pixels, mode='RGBA')
        
    def update_mesh_with_texture(self, mesh, texture_img, uv_coords):
        """更新网格使用纹理"""
        print("  更新网格材质...")
        
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_img,
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        
        texture_visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            material=material
        )
        
        mesh.visual = texture_visual
        return mesh
        
    def process_mesh(self, mesh_data):
        """处理单个网格（用于并行处理）"""
        name, mesh = mesh_data
        print(f"\n处理网格: {name}")
        
        # 生成或获取UV坐标
        uv_coords = self.get_or_generate_uv_coordinates(mesh)
        
        # 烘焙顶点色到纹理
        texture_img = self.bake_vertex_colors_to_texture(mesh, uv_coords)
        
        return name, mesh, texture_img, uv_coords
        
    def process(self):
        """执行转换流程"""
        try:
            total_start = time.time()
            
            # 加载模型
            self.load_mesh()
            
            # 并行处理网格（如果有多个）
            if len(self.meshes) > 1 and self.num_workers > 1:
                print(f"\n并行处理 {len(self.meshes)} 个网格...")
                
                # 使用线程池处理
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(self.process_mesh, self.meshes))
            else:
                # 串行处理
                results = []
                for mesh_data in self.meshes:
                    results.append(self.process_mesh(mesh_data))
                    
            # 保存结果
            processed_meshes = []
            for i, (name, mesh, texture_img, uv_coords) in enumerate(results):
                # 保存纹理
                texture_filename = f"texture_{name}_{i}.png"
                texture_path = os.path.join(self.output_dir, texture_filename)
                texture_img.save(texture_path)
                print(f"  纹理已保存: {texture_path}")
                
                # 更新网格
                updated_mesh = self.update_mesh_with_texture(mesh, texture_img, uv_coords)
                processed_meshes.append((name, updated_mesh))
                
            # 保存模型
            print(f"\n保存模型: {self.output_path}")
            
            if isinstance(self.scene, trimesh.Scene):
                new_scene = trimesh.Scene()
                for name, mesh in processed_meshes:
                    new_scene.add_geometry(mesh, node_name=name)
                if hasattr(self.scene, 'graph'):
                    new_scene.graph = self.scene.graph
                new_scene.export(self.output_path)
            else:
                processed_meshes[0][1].export(self.output_path)
                
            total_elapsed = time.time() - total_start
            print(f"\n转换完成！总耗时: {total_elapsed:.2f}秒")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将GLB/GLTF模型的顶点色转换为纹理贴图（优化版）')
    parser.add_argument('input', help='输入的GLB/GLTF文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径', default=None)
    parser.add_argument('-s', '--size', type=int, default=1024, 
                       help='纹理尺寸 (默认: 1024)')
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='工作线程数 (默认: CPU核心数)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
        
    # 创建转换器并执行
    converter = VertexColorToTexture(
        input_path=args.input,
        output_path=args.output,
        texture_size=args.size,
        num_workers=args.workers
    )
    converter.process()

if __name__ == "__main__":
    main()