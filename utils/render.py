import torch
import torch.nn as nn
import numpy as np

# 尝试导入 nvdiffrast
try:
    import nvdiffrast.torch as dr
    HAS_NVDIFFRAST = True
except ImportError:
    HAS_NVDIFFRAST = False
    print("[Warn] nvdiffrast not found. Renderer will fail.")

# 尝试导入 PyTorch3D 用于相机生成 (可选，但很方便)
try:
    from pytorch3d.renderer import look_at_view_transform
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False

class DifferentiableNormalRenderer(nn.Module):
    """
    基于 nvdiffrast 的可微法线渲染器。
    性能远优于 PyTorch3D MeshRenderer，适合高分辨率 Mesh。
    """
    def __init__(self, image_size=512, device='cuda', cameras_per_batch=4, fov=60.0, dist_multiplier=1.0):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.cameras_per_batch = cameras_per_batch
        self.fov = fov
        self.dist_multiplier = dist_multiplier
        
        if not HAS_NVDIFFRAST:
            raise ImportError("nvdiffrast is required.")
            
        # 初始化 Context (优先使用 CUDA，如果失败尝试 GL - 但在无头服务器上 GL 可能有问题)
        try:
            self.ctx = dr.RasterizeCudaContext(device=device)
        except Exception as e:
            print(f"[Warn] CudaContext failed, trying GL: {e}")
            self.ctx = dr.RasterizeGLContext(device=device)

    def get_projection_matrix(self, batch_size):
        """
        构建透视投影矩阵 (OpenGL style)
        """
        aspect = 1.0 # Square image
        near = 0.1
        far = 100.0
        
        fovy = np.deg2rad(self.fov)
        f = 1.0 / np.tan(fovy / 2.0)
        
        # OpenGL Projection Matrix
        # [ f/aspect, 0, 0, 0 ]
        # [ 0, f, 0, 0 ]
        # [ 0, 0, (f+n)/(n-f), (2fn)/(n-f) ]
        # [ 0, 0, -1, 0 ]
        
        proj = torch.zeros(batch_size, 4, 4, device=self.device)
        proj[:, 0, 0] = f / aspect
        proj[:, 1, 1] = f
        proj[:, 2, 2] = (far + near) / (near - far)
        proj[:, 2, 3] = (2 * far * near) / (near - far)
        proj[:, 3, 2] = -1.0
        
        return proj

    def get_random_mvp(self, batch_size, dist=2.5):
        """
        生成 Model-View-Projection 矩阵
        """
        if HAS_PYTORCH3D:
            # 使用 PyTorch3D 生成 R, T (World -> View)
            elev = torch.rand(batch_size) * 180 - 90
            azim = torch.rand(batch_size) * 360
            R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=self.device)
            
            # PyTorch3D 的 R 是 row-major，T 是行向量
            # View Matrix (4x4)
            # [ R^T  0 ]   (注意 PyTorch3D 文档说 R 是 row-major，但 look_at 返回的通常需要转置才能符合标准数学定义？
            # [ T    1 ]    PyTorch3D 的变换通常是 x @ R + T
            
            # 我们需要构建标准的 OpenGL View Matrix: V @ P_world
            # PyTorch3D look_at_view_transform 返回的 R, T 使得:
            # Points_view = Points_world @ R + T
            
            # 构建 View 矩阵 (Column-major for matmul on right? No, let's stick to standard)
            # MVP = Proj @ View @ Model
            # OpenGL 习惯列向量: v_clip = M_proj @ M_view @ v_world
            # PyTorch 习惯行向量: v_clip = v_world @ M_view^T @ M_proj^T
            
            # nvdiffrast 习惯行向量输入: v_clip = v_world @ MVP^T
            # 所以我们需要构建 MVP^T (即 P3D 风格的变换矩阵)
            
            # View Matrix (Transposed for PyTorch multiplication)
            # [ R   0 ]
            # [ T   1 ]
            # 注意: PyTorch3D 的 R, T 是用于 x @ R + T
            
            view = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            view[:, :3, :3] = R
            view[:, 3, :3] = T
            
            # Projection Matrix (Transposed)
            proj = self.get_projection_matrix(batch_size).transpose(1, 2)
            
            # 坐标系修正: PyTorch3D View (+X Left, +Y Up, +Z Front) -> OpenGL View (+X Right, +Y Up, +Z Back)
            # 需要翻转 X 和 Z 轴
            # Scale matrix: diag(-1, 1, -1, 1)
            scale = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0], device=self.device))
            scale = scale.unsqueeze(0).expand(batch_size, -1, -1)
            
            # MVP^T = View @ Scale @ Proj
            mvp = torch.bmm(torch.bmm(view, scale), proj)
            
            # 坐标系修正: PyTorch3D NDC (+X Left, +Y Up) vs nvdiffrast (+X Right, +Y Up)
            # PyTorch3D World: +X Left, +Y Up, +Z Front
            # 我们通常不需要太纠结，只要 Pred 和 GT 用一样的矩阵即可。
            
            return mvp, R.transpose(1, 2) # Return R for normal transform if needed
            
        else:
            raise NotImplementedError("PyTorch3D required for camera generation currently.")

    def forward(self, verts, faces, gt_verts=None, gt_faces=None):
        """
        Args:
            verts: (B, V, 3)
            faces: (B, F, 3) or (F, 3)
            gt_verts: (B, V_gt, 3)
            gt_faces: (B, F_gt, 3) or (F_gt, 3)
            
        Returns:
            pred_img: (B*K, H, W, 3)
            gt_img: (B*K, H, W, 3)
        """
        B = verts.shape[0]
        K = self.cameras_per_batch
        total_batch = B * K
        
        # 1. 准备 MVP 矩阵
        mvp, _ = self.get_random_mvp(total_batch, dist=self.dist_multiplier) # (BK, 4, 4)
        
        # 2. 准备数据
        # 扩展 Verts: (B, V, 3) -> (B, K, V, 3) -> (BK, V, 3)
        verts_expanded = verts.unsqueeze(1).expand(-1, K, -1, -1).reshape(total_batch, -1, 3)
        
        # Faces 需要转换为 int32 for nvdiffrast
        # nvdiffrast rasterize expects tri to be (F, 3) for shared topology across batch
        faces = faces.to(torch.int32)
        if faces.ndim == 3 and faces.shape[0] == 1:
            faces = faces.squeeze(0) # (1, F, 3) -> (F, 3)
            
        if faces.ndim == 2: # Shared faces (F, 3)
             faces_expanded = faces.contiguous()
        else:
             # (B, F, 3) with B > 1. 
             # nvdiffrast simple rasterize doesn't support batched faces directly without ranges.
             # For this project, we assume B=1 per forward call usually.
             # If B > 1, we must assume user knows what they are doing or fail.
             # But let's try to flatten if they are same? No, can't guarantee.
             # We'll just take the first one and warn? Or crash?
             # Let's assume (BK, F, 3) is NOT supported and we only support shared faces for now.
             print("[Warn] Batched faces (B>1) detected. Using faces from first batch item for all.")
             faces_expanded = faces[0].contiguous()

        # 3. Vertex Transform (World -> Clip)
        # v_clip = [x, y, z, 1] @ MVP
        v_homo = torch.cat([verts_expanded, torch.ones_like(verts_expanded[..., :1])], dim=-1) # (BK, V, 4)
        v_clip = torch.bmm(v_homo, mvp) # (BK, V, 4)
        
        # 4. Rasterize
        # ranges: 如果 faces 都是一样长的，直接传入 faces tensor 即可 (Batched mode)
        # nvdiffrast 支持 Batched Rasterization
        # NOTE: nvdiffrast requires contiguous tensors for cuda kernels
        v_clip = v_clip.contiguous()
        faces_expanded = faces_expanded.contiguous()
        rast, _ = dr.rasterize(self.ctx, v_clip, faces_expanded, (self.image_size, self.image_size))
        
        # 5. Interpolate Normals
        # 计算 Vertex Normals (如果没传，需要计算)
        v_normals = self.compute_vertex_normals(verts_expanded, faces_expanded)
        
        # Interpolate Normals
        pred_normals, _ = dr.interpolate(v_normals, rast, faces_expanded)
        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
        pred_normals = dr.antialias(pred_normals, rast, v_clip, faces_expanded)
        
        # Extract Depth from rasterization output
        # rast[..., 2] contains the depth (z) in NDC space
        # Convert from NDC depth to world depth if needed, or use directly
        pred_depth = rast[..., 2:3]  # (BK, H, W, 1)
        
        # Background masking
        # rast[..., 3] > 0 means valid
        mask = rast[..., 3:4] > 0
        pred_img = pred_normals * mask # 0 for bg
        pred_depth = pred_depth * mask  # 0 for bg
        
        # GT Render
        gt_img = None
        gt_depth = None
        if gt_verts is not None and gt_faces is not None:
            # Prepare GT
            gt_verts_ex = gt_verts.unsqueeze(1).expand(-1, K, -1, -1).reshape(total_batch, -1, 3)
            
            if gt_faces.ndim == 3 and gt_faces.shape[0] == 1:
                gt_faces = gt_faces.squeeze(0)
            
            gt_faces = gt_faces.to(torch.int32)
            gt_faces_ex = gt_faces.contiguous()
                
            # GT Transform
            gt_homo = torch.cat([gt_verts_ex, torch.ones_like(gt_verts_ex[..., :1])], dim=-1)
            gt_clip = torch.bmm(gt_homo, mvp)
            
            # GT Rasterize
            gt_clip = gt_clip.contiguous()
            gt_faces_ex = gt_faces_ex.contiguous()
            gt_rast, _ = dr.rasterize(self.ctx, gt_clip, gt_faces_ex, (self.image_size, self.image_size))
            
            # GT Normals
            gt_v_normals = self.compute_vertex_normals(gt_verts_ex, gt_faces_ex)
            gt_normals, _ = dr.interpolate(gt_v_normals, gt_rast, gt_faces_ex)
            gt_normals = torch.nn.functional.normalize(gt_normals, dim=-1)
            gt_normals = dr.antialias(gt_normals, gt_rast, gt_clip, gt_faces_ex)
            
            gt_mask = gt_rast[..., 3:4] > 0
            gt_img = gt_normals * gt_mask
            gt_depth = gt_rast[..., 2:3] * gt_mask  # Extract depth
        else:
            gt_depth = None
            
        return pred_img, gt_img, pred_depth, gt_depth

    def compute_vertex_normals(self, verts, faces):
        """
        Batch 计算 Vertex Normals (Weighted by triangle area)
        verts: (B, V, 3)
        faces: (B, F, 3) or (F, 3)
        """
        B, V, _ = verts.shape
        
        if faces.ndim == 2:
            # Shared faces (F, 3)
            F = faces.shape[0]
            faces_batched = faces.unsqueeze(0).expand(B, -1, -1)
        else:
            F = faces.shape[1]
            faces_batched = faces
        
        # Gather vertices of faces
        # 构造 Batch 索引偏移
        batch_offset = torch.arange(B, device=self.device) * V
        batch_offset = batch_offset.view(-1, 1, 1)
        
        faces_flat = (faces_batched.long() + batch_offset).view(-1, 3) # (B*F, 3)
        verts_flat = verts.reshape(-1, 3) # (B*V, 3)
        
        v0 = verts_flat[faces_flat[:, 0]]
        v1 = verts_flat[faces_flat[:, 1]]
        v2 = verts_flat[faces_flat[:, 2]]
        
        # Face Normals (un-normalized, magnitude = 2 * area)
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1) # (B*F, 3)
        
        # Scatter Add to Vertices
        # zeros: (B*V, 3)
        v_normals = torch.zeros_like(verts_flat)
        
        # scatter_add_ (dim=0)
        # index: (B*F, 1) -> (B*F, 3)
        for i in range(3):
            v_normals.scatter_add_(0, faces_flat[:, i:i+1].expand(-1, 3), face_normals)
            
        # Normalize
        v_normals = torch.nn.functional.normalize(v_normals, dim=1, eps=1e-6)
        
        return v_normals.view(B, V, 3)
