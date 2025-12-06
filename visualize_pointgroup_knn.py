import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class ClusterDemoFixed:
    def __init__(self, mesh_path=None, pcd_path=None):
        self.cur_vertex_idx = 0
        self.vis_mode = 0 # 0: Gradient (彩虹渐变), 1: Cluster (聚类随机色)
        
        # 1. 加载或生成数据
        if mesh_path is None:
            print("生成测试数据...")
            self.mesh, self.dense_pcd = self._generate_dummy_data()
        else:
            self.mesh = o3d.io.read_triangle_mesh(mesh_path)
            self.dense_pcd = o3d.io.read_point_cloud(pcd_path)
            
        # 2. 计算 KDTree (Point -> Vertex)
        print("计算 KDTree...")
        self.mesh_verts = np.asarray(self.mesh.vertices)
        self.dense_points = np.asarray(self.dense_pcd.points)
        
        self.tree = cKDTree(self.mesh_verts)
        # k=1: 让每个密集点找到最近的一个 Mesh 顶点
        _, self.point_to_vertex_indices = self.tree.query(self.dense_points, k=1)
        
        # 3. 准备可视化几何体
        self._prepare_geometries()
        
        # 4. 启动可视化
        self._run_vis()

    def _generate_dummy_data(self):
        # 生成一个球体 Mesh
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        mesh.compute_vertex_normals()
        # 生成密集点云
        pcd = mesh.sample_points_uniformly(number_of_points=5000)
        # 加点噪声
        points = np.asarray(pcd.points)
        noise = np.random.normal(0, 0.03, points.shape)
        pcd.points = o3d.utility.Vector3dVector(points + noise)
        return mesh, pcd

    def _prepare_geometries(self):
        # === 准备两种颜色模式 ===
        
        # 1. Gradient Mode (彩虹渐变) - 类似于 Heatmap
        # 找到变化最大的轴 (X, Y, Z)
        ptp = np.ptp(self.dense_points, axis=0) # Peak to peak (max - min)
        main_axis = np.argmax(ptp) # 0, 1, or 2
        
        # 归一化坐标 [0, 1]
        coords = self.dense_points[:, main_axis]
        norm_coords = (coords - coords.min()) / (coords.max() - coords.min())
        
        # 使用 Matplotlib 的 'jet' 配色方案 (蓝->青->黄->红)
        cmap = plt.get_cmap("jet")
        # cmap(x) 返回 RGBA，我们要 RGB
        self.colors_gradient = cmap(norm_coords)[:, :3]

        # 2. Cluster Mode (聚类随机色)
        np.random.seed(42) 
        # 为每个 Mesh 顶点生成随机颜色
        vertex_colors_random = np.random.uniform(0, 1, (len(self.mesh_verts), 3))
        # 广播到每个密集点
        self.colors_cluster = vertex_colors_random[self.point_to_vertex_indices]
        
        # === 初始化显示 ===
        # 默认使用 Gradient 模式
        self.current_base_colors = self.colors_gradient if self.vis_mode == 0 else self.colors_cluster
        self.dense_pcd.colors = o3d.utility.Vector3dVector(self.current_base_colors)
        
        # B. Base Mesh 线框 (青色)
        self.mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        self.mesh_lines.paint_uniform_color([0, 1, 1]) 

        # C. Mesh 顶点 (红色大点)
        self.vertex_pcd = o3d.geometry.PointCloud()
        self.vertex_pcd.points = o3d.utility.Vector3dVector(self.mesh_verts)
        self.vertex_pcd.paint_uniform_color([1, 0, 0]) # 红色

    def _toggle_mode(self, vis):
        """回调函数：切换颜色模式 (Gradient <-> Cluster)"""
        self.vis_mode = 1 - self.vis_mode # Toggle 0/1
        
        mode_name = "Gradient (彩虹)" if self.vis_mode == 0 else "Cluster (随机聚类)"
        print(f"切换显示模式: {mode_name}")
        
        # 更新当前的基础颜色
        self.current_base_colors = self.colors_gradient if self.vis_mode == 0 else self.colors_cluster
        
        # 如果当前没有高亮特定的点，直接刷新
        # 这里简单处理：直接重置为基础颜色（取消任何高亮状态）
        self.dense_pcd.colors = o3d.utility.Vector3dVector(self.current_base_colors)
        vis.update_geometry(self.dense_pcd)
        return False

    def _highlight_next(self, vis):
        """回调函数：高亮下一个顶点的聚类"""
        self.cur_vertex_idx = (self.cur_vertex_idx + 1) % len(self.mesh_verts)
        self._update_highlight(vis)
        return False

    def _highlight_random(self, vis):
        """回调函数：随机高亮一个顶点的聚类"""
        self.cur_vertex_idx = np.random.randint(0, len(self.mesh_verts))
        self._update_highlight(vis)
        return False

    def _update_highlight(self, vis):
        """核心逻辑：改变颜色"""
        target_idx = self.cur_vertex_idx
        print(f"查看顶点 [{target_idx}]: 属于它的点云将变绿...")

        # 1. 复制当前模式的基础颜色 (作为底色)
        new_colors = self.current_base_colors.copy()
        
        # 2. 找到属于该顶点的点 (Mask)
        mask = (self.point_to_vertex_indices == target_idx)
        
        # 3. 染色 (亮绿色) - 无论什么模式，选中的都变绿
        new_colors[mask] = [0, 1, 0]
        
        # 4. 同时高亮那个 Mesh 顶点 (变为黄色)
        vert_colors = np.full((len(self.mesh_verts), 3), [1, 0, 0]) # 默认红
        vert_colors[target_idx] = [1, 1, 0] # 选中变黄
        
        # 应用更新
        self.dense_pcd.colors = o3d.utility.Vector3dVector(new_colors)
        self.vertex_pcd.colors = o3d.utility.Vector3dVector(vert_colors)
        
        vis.update_geometry(self.dense_pcd)
        vis.update_geometry(self.vertex_pcd)

    def _run_vis(self):
        # 使用支持按键回调的 Visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Fixed Demo (Press SPACE, N, or C)", width=1000, height=800)
        
        # 添加几何体
        vis.add_geometry(self.dense_pcd)
        vis.add_geometry(self.mesh_lines)
        vis.add_geometry(self.vertex_pcd)
        
        # === 渲染选项 ===
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0]) # 纯黑背景
        opt.point_size = 4.0
        opt.line_width = 1.0
        
        # 注册按键
        # 32: Space -> Random
        vis.register_key_callback(32, self._highlight_random)
        # 78: 'N' -> Next
        vis.register_key_callback(78, self._highlight_next)
        # 67: 'C' -> Color Mode Toggle
        vis.register_key_callback(67, self._toggle_mode)
        
        print("\n=== 操作说明 ===")
        print(" [Space] 键 : 随机选中一个顶点并高亮它的聚类")
        print(" [N]     键 : 顺序查看下一个顶点")
        print(" [C]     键 : 切换颜色模式 (彩虹渐变 / 随机聚类)")
        print(" [Esc]   键 : 退出")
        print("================\n")
        
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    demo = ClusterDemoFixed(mesh_path="0002_original_1000f.ply", pcd_path="0002_pointcloud_160000.ply")
