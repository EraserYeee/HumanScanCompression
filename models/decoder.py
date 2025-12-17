import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.subdivision import BarycentricSubdivision

def compute_rotation_matrices(normals: torch.Tensor) -> torch.Tensor:
    """
    根据法线计算每个顶点的局部旋转矩阵 R (B, V, 3, 3)。
    
    坐标系定义:
    Z轴 = Normal (法线方向)
    X轴 = 在切平面上，尽量指向固定的参考方向 (如 [1, 0, 0] 或 [0, 1, 0])
    Y轴 = Z x X (确保右手坐标系)

    Args:
        normals: (B, V, 3) 归一化的法线向量

    Returns:
        R: (B, V, 3, 3) 旋转矩阵，使得 p_local = (p_global - center) @ R^T
            即 R 的行向量分别是 X, Y, Z 轴
    """
    B, V, _ = normals.shape
    
    # 1. Z轴: 法线方向
    z_axis = F.normalize(normals, dim=-1) # (B, V, 3)
    
    # 2. 构造辅助向量来计算 X 轴
    # 策略: 优先使用 [0, 1, 0] (Up)，如果法线与 Up 平行，则使用 [1, 0, 0]
    up = torch.tensor([0.0, 1.0, 0.0], device=normals.device).view(1, 1, 3).expand(B, V, 3)
    
    # 计算点积绝对值，判断是否平行
    dot = torch.abs(torch.sum(z_axis * up, dim=-1, keepdim=True))
    
    # 如果接近平行 (>0.99)，使用 [1, 0, 0] 作为辅助向量
    fallback = torch.tensor([1.0, 0.0, 0.0], device=normals.device).view(1, 1, 3).expand(B, V, 3)
    aux_vector = torch.where(dot > 0.99, fallback, up)
    
    # 3. X轴: aux x Z (垂直于法线，位于切平面)
    x_axis = torch.cross(aux_vector, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1)
    
    # 4. Y轴: Z x X (确保正交)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    # y_axis 已经是归一化的，因为 Z 和 X 正交且归一化
    
    # 5. 构建矩阵 R
    # 我们希望局部坐标 p_local = [x_local, y_local, z_local]
    # x_local = (p - c) dot X_axis
    # y_local = (p - c) dot Y_axis
    # z_local = (p - c) dot Z_axis
    # 所以 R 的行向量应该是 X, Y, Z
    
    # stack(dim=-2) -> (B, V, 3, 3)
    # R[b, v] = [ [x_0, x_1, x_2],
    #             [y_0, y_1, y_2],
    #             [z_0, z_1, z_2] ]
    R = torch.stack([x_axis, y_axis, z_axis], dim=-2)
    
    return R

def positional_encoding(vector: torch.Tensor, extras: list[torch.Tensor], levels: int) -> torch.Tensor:
    """
    复刻 ngf.py 中的 positional_encoding
    Improved: Add PI factor to ensure better frequency coverage in [-1, 1]
    """
    result = list(extras) # Copy list
    
    # Use torch.linspace for frequencies ensures we cover the range evenly if needed, 
    # but 2^i is standard.
    # We add math.pi to ensure sin(x * pi) covers a full cycle in [-1, 1]
    
    for i in range(levels):
        k = (2.0 ** i) * math.pi
        result.append(torch.sin(k * vector))
        result.append(torch.cos(k * vector))
        
    return torch.cat(result, dim=-1)

class NeuralSubdivisionDecoder(nn.Module):
    """
    基于 NGF 逻辑的解码器。
    不改变 Base Mesh 的拓扑数据结构，而是通过重心坐标采样生成细分后的点云和面片。
    
    流程:
    1. 在 Base Mesh 的每个三角形面上生成密集采样点 (重心坐标 A, B)。
    2. 插值得到采样点的:
       - 粗糙位置 lp (Linear Position)
       - 特征 lf (Linear Feature)
       - 法线 ln (Linear Normal)
    3. 神经位移:
       - input = PosEnc(lp) + lf
       - displacement = MLP(input)
       - fine_pos = lp + displacement * ln
    """
    def __init__(self, feature_dim=128, hidden_dim=64, levels=8, rate=4, predict_offset=False):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: MLP 隐藏层维度
            levels: Positional Encoding 的层数 (fflevels)
            rate: 细分等级 (edge subdivision rate)
            predict_offset: If True, predict 3D offset (xyz) instead of scalar displacement
        """
        super().__init__()
        self.fflevels = levels
        self.rate = rate
        self.predict_offset = predict_offset
        self.subdivision = BarycentricSubdivision()
        
        # MLP Input Dim calculation
        # input = lf (feature_dim) + PosEnc(relative_pos) (3 * 2 * levels)
        # Note: relative_pos is 3D vector, same as before
        self.input_dim = feature_dim + 3 * 2 * levels
        
        # MLP Output Dim: 1 for scalar displacement (along normal), 3 for vector offset (xyz)
        out_dim = 3 if self.predict_offset else 1
        
        # MLP_Normal from ngf.py
        # We share the MLP across the 3 vertices' contributions
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # Initialize the last layer to output near-zero values
        # This ensures the deformation starts from the base mesh
        nn.init.uniform_(self.mlp[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def interpolate_barycentric(self, attrs, faces, A, B):
        """
        Args:
            attrs: (B, V, C) 顶点属性
            faces: (B, F, 3) 面片索引
            A, B: (F*K,) 重心坐标 (已被 repeat 适配了所有面)
            
        Returns:
            interpolated: (B, F*K, C)
        """
        # attrs: (B, V, C)
        # faces: (B, F, 3)
        
        # Gather face vertex attributes
        # (B, F, 3) -> (B, F, 3, C)
        # 需要 gather: attrs[b, faces[b, f, i], :]
        
        B_batch, V, C = attrs.shape
        _, F_count, _ = faces.shape
        
        # Flat indices for gather
        # batch_offset: (B, 1, 1)
        batch_offset = (torch.arange(B_batch, device=attrs.device) * V).view(-1, 1, 1)
        flat_faces = (faces + batch_offset).view(-1) # (B*F*3)
        flat_attrs = attrs.view(-1, C) # (B*V, C)
        
        face_attrs = flat_attrs[flat_faces].view(B_batch, F_count, 3, C) # (B, F, 3, C)
        
        # Reshape A, B for broadcasting
        # A, B input: (F*K,) -> need to reshape to match (B, F, K) logic?
        # subdivision.sample_uniform_bary 返回的是 (F*K,)，没有 Batch 维
        # 我们假设所有 Batch 共享相同的细分模式 (A, B 相同)
        
        # A: (F*K,) -> (F, K)
        # 这里的 F 是 faces count (num_triangles)
        # NGF 的逻辑是 A, B 长度为 Nt * K
        
        K = A.shape[0] // F_count 
        
        # A_reshaped: (1, F, K, 1)
        A_reshaped = A.view(1, F_count, K, 1)
        B_reshaped = B.view(1, F_count, K, 1)
        C_reshaped = 1.0 - A_reshaped - B_reshaped
        
        # face_attrs: (B, F, 3, C) -> (B, F, 1, 3, C) for broadcasting with K
        # Wait, we need to combine 3 vertices.
        # v0 = face_attrs[:, :, 0, :] # (B, F, C)
        # v0 * A ? No. A corresponds to points inside the triangle.
        
        # Correct logic:
        # Result shape: (B, F, K, C)
        # v0 * A + v1 * B + v2 * C
        
        v0 = face_attrs[:, :, 0, :].unsqueeze(2) # (B, F, 1, C)
        v1 = face_attrs[:, :, 1, :].unsqueeze(2) # (B, F, 1, C)
        v2 = face_attrs[:, :, 2, :].unsqueeze(2) # (B, F, 1, C)
        
        interp = v0 * A_reshaped + v1 * B_reshaped + v2 * C_reshaped # (B, F, K, C)
        
        return interp.view(B_batch, -1, C) # (B, F*K, C)

    def forward(self, base_verts, base_faces, vertex_features, base_normals):
        """
        Args:
            base_verts: (B, V, 3)
            base_faces: (B, F, 3)
            vertex_features: (B, V, D)
            base_normals: (B, V, 3)

        Returns:
            fine_verts: (B, V_fine, 3)
            fine_faces: (F_fine, 3) shared across batch (if topology is same)
            displacements: (B, V_fine, 1)
        """
        B, V, _ = base_verts.shape
        B = int(B) # Ensure B is int for view()
        _, num_faces, _ = base_faces.shape
        
        # 1. Generate Barycentric Coordinates
        # uv_A, uv_B: (F*K,)
        uv_A, uv_B = self.subdivision.sample_uniform_bary(self.rate, num_triangles=num_faces)
        
        # 2. Interpolate Attributes
        # lp: Linear Position (B, F*K, 3)
        # This is the global position of the subdivision point on the base mesh
        lp = self.interpolate_barycentric(base_verts, base_faces, uv_A, uv_B)
        
        # ln: Linear Normal (B, F*K, 3)
        ln = self.interpolate_barycentric(base_normals, base_faces, uv_A, uv_B)
        ln = F.normalize(ln, dim=-1, p=2)
        
        # 3. Vertex-Centric Neural Displacement
        # Instead of interpolating features and pos-encoding barycentric coords,
        # we compute displacement contributions from each of the 3 vertices
        # based on their local coordinate systems.
        
        # 3.1 Gather vertex attributes for each face
        # Similar logic to interpolate_barycentric but we keep the 3 vertices separate
        batch_offset = (torch.arange(B, device=base_verts.device) * V).view(-1, 1, 1)
        flat_faces = (base_faces + batch_offset).view(-1) # (B*F*3)
        
        # Features: (B, F, 3, D)
        flat_feats = vertex_features.view(-1, vertex_features.shape[-1])
        face_feats = flat_feats[flat_faces].view(B, num_faces, 3, -1)
        
        # Vertices: (B, F, 3, 3)
        flat_verts = base_verts.view(-1, 3)
        face_verts = flat_verts[flat_faces].view(B, num_faces, 3, 3)
        
        # Normals: (B, F, 3, 3) - needed for rotation matrix
        flat_normals = base_normals.view(-1, 3)
        face_normals = flat_normals[flat_faces].view(B, num_faces, 3, 3) # (B, F, 3, 3)
        
        # 3.2 Compute Rotation Matrices for all face vertices
        # We need R for each vertex of each face.
        # But compute_rotation_matrices expects (B, V, 3). 
        # We can flatten face_normals to (B*F*3, 1, 3) or just treat as (B, F*3, 3)
        face_normals_flat = face_normals.view(B, num_faces * 3, 3)
        R_flat = compute_rotation_matrices(face_normals_flat) # (B, F*3, 3, 3)
        R_face = R_flat.view(B, num_faces, 3, 3, 3) # (B, F, 3vertices, 3, 3)
        
        # 3.3 Prepare Inputs for MLP (for each of 3 vertices)
        # We need to process each of the K subdivision points for each Face.
        # lp: (B, F*K, 3). Reshape to (B, F, K, 3)
        K = uv_A.shape[0] // num_faces
        lp_reshaped = lp.view(B, num_faces, K, 3)
        
        # Barycentric weights for weighted sum later
        # uv_A, uv_B: (F*K,) -> (F, K)
        # But since they are uniform across all faces (if using shared topology cache), 
        # actually sample_uniform_bary returns (Nt*K) where it repeats.
        # Let's reshape to (1, F, K, 1)
        w0 = uv_A.view(1, num_faces, K, 1)
        w1 = uv_B.view(1, num_faces, K, 1)
        w2 = 1.0 - w0 - w1
        
        # Initialize accumulated output (hidden state or final disp)
        # We choose to accumulate the MLP outputs (displacement vectors/scalars)
        total_disp = 0
        
        # Loop over 3 vertices of the triangle
        weights = [w0, w1, w2]
        
        for i in range(3):
            # Vertex position: (B, F, 3) -> (B, F, 1, 3)
            v_pos = face_verts[:, :, i, :].unsqueeze(2)
            
            # Relative position in Global Coords
            # delta: (B, F, K, 3)
            delta = lp_reshaped - v_pos
            
            # Rotate to Vertex Local Coords
            # R: (B, F, 3, 3). Need (B, F, 1, 3, 3) to broadcast over K
            # R_i = R_face[:, :, i, :, :] # (B, F, 3, 3)
            # local_pos = (R @ delta.T).T
            # Matmul: (..., 3, 3) @ (..., 3, 1) -> (..., 3, 1)
            R_i = R_face[:, :, i, :, :].unsqueeze(2) # (B, F, 1, 3, 3)
            
            # delta: (B, F, K, 3) -> (B, F, K, 3, 1)
            delta_expanded = delta.unsqueeze(-1)
            
            # local_pos: (B, F, K, 3, 1) -> squeeze -> (B, F, K, 3)
            local_pos = torch.matmul(R_i, delta_expanded).squeeze(-1)
            
            # Feature: (B, F, D) -> (B, F, K, D)
            feat = face_feats[:, :, i, :].unsqueeze(2).expand(-1, -1, K, -1)
            
            # Positional Encoding
            # local_pos is physically scaled (e.g. 0.01). 
            # Ideally we might want to scale it up if it's too small, but let's stick to raw first
            # or maybe scale by avg edge length? For now raw.
            # pos_enc: (B, F, K, 2*3*L)
            # We treat batch/face/k dims as flattened for encoding function if needed, 
            # but our func handles tensor input.
            
            # Input to MLP: Concat(Feature, PosEnc(LocalPos))
            # list wraps feature as extras
            mlp_in = positional_encoding(local_pos, [feat], self.fflevels) 
            
            # Pass through MLP
            # mlp_in: (B, F, K, InputDim) -> (B*F*K, InputDim)
            out = self.mlp(mlp_in.view(-1, self.input_dim))
            
            # Reshape back: (B, F, K, OutDim)
            out = out.view(B, num_faces, K, -1)
            
            # Weighted Accumulation (Barycentric Interpolation of Displacements)
            # weights[i]: (1, F, K, 1) on CPU/GPU?
            # uv_A is on device.
            total_disp = total_disp + out * weights[i].to(out.device)
            
        # Final Displacement
        # total_disp: (B, F, K, OutDim) -> (B, F*K, OutDim)
        disp_flat = total_disp.view(B, -1, total_disp.shape[-1])
        
        if self.predict_offset:
            # Output is (B, F*K, 3) vector offset (Global Coords? No, local weighted sum?)
            # Wait, the MLP predicted displacement in WHICH coordinate system?
            # If we just sum them up, we are summing vectors.
            # "Local Tangent Space" vectors? Or "Global" vectors?
            
            # Strategy: The MLP output `out` should be interpreted as a vector 
            # in the Vertex's Local Frame (Tangent Space) OR Global Frame?
            
            # If we want Rotation Invariance, the MLP MUST predict in Local Frame.
            # Then we must rotate it back to Global before averaging!
            
            # Re-loop to rotate back (or do it inside the loop)
            # Let's redo the loop logic slightly to be correct.
            pass
        else:
            # Scalar displacement along Normal.
            # Scalar is rotation invariant. We just sum scalar values.
            # disp_flat is (B, F*K, 1).
            pass
            
        # --- Correct Logic for Loop with Rotation Back ---
        total_disp_global = 0
        total_disp_scalar = 0
        
        for i in range(3):
            # ... (Same preparation as above)
            v_pos = face_verts[:, :, i, :].unsqueeze(2)
            delta = lp_reshaped - v_pos
            R_i = R_face[:, :, i, :, :].unsqueeze(2) # (B, F, 1, 3, 3)
            local_pos = torch.matmul(R_i, delta.unsqueeze(-1)).squeeze(-1)
            feat = face_feats[:, :, i, :].unsqueeze(2).expand(-1, -1, K, -1)
            
            mlp_in = positional_encoding(local_pos, [feat], self.fflevels)
            out = self.mlp(mlp_in.view(-1, self.input_dim)).view(B, num_faces, K, -1)
            
            weight = weights[i].to(out.device)
            
            if self.predict_offset:
                # out is (B, F, K, 3) in Local Frame.
                # Rotate back to Global: R^T @ out
                # R_i is (..., 3, 3). R_i_T is transpose of last 2 dims.
                # global_out_i = R_i.transpose(-1, -2) @ out
                out_expanded = out.unsqueeze(-1) # (..., 3, 1)
                R_i_T = R_i.transpose(-1, -2)
                global_out_i = torch.matmul(R_i_T, out_expanded).squeeze(-1)
                
                total_disp_global = total_disp_global + global_out_i * weight
            else:
                # out is (B, F, K, 1) scalar.
                total_disp_scalar = total_disp_scalar + out * weight

        # Final application
        if self.predict_offset:
            disp = total_disp_global.view(B, -1, 3) # (B, F*K, 3)
            fine_verts = lp + disp
        else:
            disp = total_disp_scalar.view(B, -1, 1) # (B, F*K, 1)
            fine_verts = lp + disp * ln
        
        # 4. Build Topology (for rendering)
        fine_faces = self.subdivision.build_triangulated_faces(self.rate, num_triangles=num_faces)
        
        return fine_verts, fine_faces, disp
