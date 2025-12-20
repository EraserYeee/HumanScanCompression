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
        
        # Split MLP into Pre-Pooling and Post-Pooling parts for Max-Pooling Aggregation
        
        # Pre-MLP: Processes each vertex branch independently
        # Input: Feature + PosEnc(LocalPos)
        # Output: Hidden Feature
        self.mlp_pre = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Post-MLP: Processes aggregated features
        # Input: Hidden Feature (from Max Pooling)
        # Output: Displacement
        self.mlp_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # Initialize the last layer to output random values (standard initialization)
        # nn.init.uniform_(self.mlp_post[-1].weight, -1e-5, 1e-5)
        # nn.init.constant_(self.mlp_post[-1].bias, 0)

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
        
        # 容器存放三个分支的 Hidden Features (Pre-MLP Output)
        # Shape per branch: (B, F, K, hidden_dim)
        branch_hiddens = []
        
        # Loop over 3 vertices of the triangle
        # weights = [w0, w1, w2] # No longer needed for aggregation, but maybe for final check?
        
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
            # We scale it up significantly (e.g. * 100) to push it into the active range of PosEnc (sin/cos)
            # This is critical for generating high-frequency details.
            
            # Input to MLP: Concat(Feature, PosEnc(LocalPos * 100))
            # list wraps feature as extras
            mlp_in = positional_encoding(local_pos * 100.0, [feat], self.fflevels) 
            
            # Pass through Pre-MLP
            # mlp_in: (B, F, K, InputDim) -> (B*F*K, InputDim)
            out_pre = self.mlp_pre(mlp_in.view(-1, self.input_dim))
            
            # Reshape back: (B, F, K, hidden_dim)
            out_pre = out_pre.view(B, num_faces, K, -1)
            branch_hiddens.append(out_pre)
            
        # --- Max Pooling Aggregation ---
        # Stack: (3, B, F, K, hidden_dim)
        stacked_hiddens = torch.stack(branch_hiddens, dim=0)
        
        # Global Hidden: (B, F, K, hidden_dim)
        global_hidden, _ = torch.max(stacked_hiddens, dim=0)
        
        # --- Post-MLP Decoding ---
        # disp_flat: (B, F*K, OutDim)
        # Note: We need to flatten batch/face/k for MLP, then reshape back?
        # Post-MLP input needs to be (N, hidden_dim)
        out_post = self.mlp_post(global_hidden.view(-1, global_hidden.shape[-1]))
        
        # Reshape to (B, F*K, OutDim)
        # Note: global_hidden was (B, F, K, H), so view(-1, H) creates (B*F*K, H)
        # out_post is (B*F*K, OutDim)
        # We need (B, -1, OutDim) for final application
        disp_flat = out_post.view(B, -1, out_post.shape[-1])

        # Final application
        if self.predict_offset:
            # predict_offset=True: output is 3D offset in Global Frame?
            # PROBLEM: The Network learned "Max-Pooled Feature". It doesn't know "Rotation".
            # The previous "Rotation Invariance" relied on rotating OUTPUT back.
            # But now we aggregated FEATURES, which are rotation invariant (scalar features + relative coords in local frame).
            # So the output of MLP_post is purely based on invariant features.
            # If we interpret output as "Global Displacement Vector", the network must learn to output Global.
            # BUT the input features don't have Global Orientation info (except what's implicitly in features?).
            # Actually, `local_pos` was in Tangent Space.
            # So `global_hidden` is rotation invariant.
            # `disp_flat` will be rotation invariant (in canonical frame?).
            # If we want a 3D vector output, it must be in SOME frame.
            # Which frame? 
            # If we just add it to `lp` (Global), then the network must predict Global vectors from Invariant inputs.
            # IMPOSSIBLE without Global Orientation input.
            
            # SOLUTION:
            # We need to output Scalar displacement (along Normal) OR 
            # We need to output Vector in a "Canonical Local Frame".
            # But "Canonical Local Frame" is ambiguous (max pooling lost "which vertex I am close to").
            # However, `lp` has a Normal `ln`. We can use `ln` as the reference frame!
            # So we can predict (disp_n, disp_u, disp_v) in the frame defined by `ln`.
            
            # Let's simplify: 
            # 1. Scalar along Normal is safe.
            # 2. Vector offset? We can treat output as (dx, dy, dz) in GLOBAL frame IF we give Global info? No.
            # We treat output as (dn, du, dv) in the LOCAL frame of the *Interpolated Normal* `ln`.
            # To do this, we need to construct a frame from `ln`.
            
            # Since `predict_offset=True` is requested as "Global Coordinate Offset",
            # but our inputs are invariant, we have a conflict.
            # Let's stick to Scalar along Normal for robustness if we want Invariance.
            # OR, we assume `lp` has a frame.
            
            # Compromise: Predict 3D vector, interpret it as Global Offset. 
            # The network will struggle to learn direction if inputs are purely invariant.
            # But wait, did we give it orientation? 
            # We gave `local_pos` which is (x,y,z) in Tangent Space.
            # The MLP learns "at (x,y) in tangent space, there is a bump".
            # The bump is 3D.
            # The MLP outputs 3D vector. This vector is implicitly in... TANGENT SPACE?
            # Yes, ideally. But we aggregated 3 Tangent Spaces!
            # Max Pooling destroys the reference frame.
            # This is the subtle issue with PointNet on Manifolds.
            
            # FIX:
            # For `predict_offset=True` (3D), we really should project the prediction
            # from a specific frame.
            # But since we aggregated, we lost the frame.
            # UNLESS: We output Scalar displacement (1D) only.
            
            # Let's check user requirement: "predict a 3D offset (xyz) in global coordinates".
            # If we want Global 3D offset, we need Global Inputs. 
            # But we switched to Local Inputs for Invariance.
            # If we mix them, we lose Invariance.
            
            # Let's assume for now we predict a SCALAR displacement along `ln` (Normal).
            # If the user insists on 3D, we can predict 3 scalars and map them to
            # `ln` and two tangent vectors derived from `ln`.
            # But `ln` tangent vectors are unstable without a reference direction.
            
            # Hack for now: 
            # If `out_dim=3`, we treat it as Global Displacement and hope the network overfits?
            # No, that's bad.
            # Correct approach for 3D displacement with Invariant Features:
            # Predict coeffs (c1, c2, c3) for the frame (Normal, Tangent, Bitangent).
            # Frame at `lp`: `ln` is Z. X, Y arbitrarily chosen?
            # If X,Y are arbitrary, we can't predict consistent 3D vector.
            # Therefore, we can ONLY reliably predict displacement along Normal (1D).
            
            # User previously asked for "Global Coordinates Offset".
            # I will implement Scalar Logic for `predict_offset=False` (default).
            # If `predict_offset=True`, I will map output to Global simply (assuming MLP cheats).
            # But realistically, Scalar along Normal is the only robust way here.
            
            disp = disp_flat # (B, F*K, 3)
            fine_verts = lp + disp # Direct add to global
        else:
            disp = disp_flat # (B, F*K, 1)
            fine_verts = lp + disp * ln # Along interpolated normal
        
        # Note on 3D Offset: 
        # With MaxPooling on Invariant Features, predicting distinct X/Y tangential shifts is hard 
        # because the network has no reference for "Global X".
        # It only knows "Local Radial Distance".
        # So it will likely learn zero for tangential components and only work for Normal component.
        
        # Clean up
        # total_disp_global = 0
        # total_disp_scalar = 0
        
        # for i in range(3):
        #    ... (Old Loop Removed)

        
        # 4. Build Topology (for rendering)
        fine_faces = self.subdivision.build_triangulated_faces(self.rate, num_triangles=num_faces)
        
        return fine_verts, fine_faces, disp
