import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.subdivision import BarycentricSubdivision
from torch_scatter import scatter_mean

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

class SumOfFeatureDecoder(nn.Module):
    """
    New Decoder Architecture: "Sum of Feature"
    
    Logic:
    1. For each of the 3 vertices of a face:
       - Compute relative position: delta_p_i = p_lin - p_anchor_i
       - Map feature and geometry to a hidden vector: h_i = MLP_F(feature_i, PosEnc(delta_p_i))
    2. Sum the hidden vectors: H = sum(h_i)
    3. Predict displacement from aggregated feature:
       - d = MLP_G(H, PosEnc(n_lin))  [if predict_scalar]
       - d = MLP_G(H)                 [if predict_offset, no normal]
    4. Stitch Seams (New):
       - Identify shared subdivision points on edges/vertices.
       - Average the predicted displacements d for these shared points.
       
    Args:
        feature_dim: Dimension of input vertex features
        hidden_dim: Dimension of the intermediate hidden vector H
        levels: Positional Encoding levels
        rate: Subdivision rate
        predict_offset: If True, predict 3D offset (xyz)
        posenc_mode: 0=No PE, 1=PE(LocalPos), 2=PE(LocalPos) + PE(Normal)
    """
    def __init__(self, feature_dim=128, hidden_dim=64, levels=8, rate=4, predict_offset=False, posenc_mode=0, init_mode='near_zero'):
        super().__init__()
        self.fflevels = levels
        self.rate = rate
        self.predict_offset = predict_offset
        self.posenc_mode = posenc_mode
        self.init_mode = init_mode
        self.subdivision = BarycentricSubdivision()
        
        # Normalize hidden_dim: single int -> [int, int] for backward compat
        # (original MLP has 2 hidden layers of equal width)
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim, hidden_dim]
        else:
            hidden_dims = list(hidden_dim)
        
        # --- MLP F (Feature Mapper) ---
        input_dim_F = feature_dim
        if self.posenc_mode == 0:
            input_dim_F += 3
        else:
            input_dim_F += 3 * 2 * levels
        
        layers_f = []
        dims_f = [input_dim_F] + hidden_dims
        for i in range(len(hidden_dims)):
            layers_f.extend([nn.Linear(dims_f[i], dims_f[i + 1]), nn.LeakyReLU()])
        self.mlp_feature_map = nn.Sequential(*layers_f)
        
        # --- MLP G (Predictor) ---
        input_dim_G = hidden_dims[-1]
        if not self.predict_offset:
            if self.posenc_mode == 2:
                input_dim_G += 3 * 2 * levels
            else:
                input_dim_G += 3
        
        out_dim = 3 if self.predict_offset else 1
        
        layers_g = []
        dims_g = [input_dim_G] + hidden_dims + [out_dim]
        for i in range(len(dims_g) - 1):
            layers_g.append(nn.Linear(dims_g[i], dims_g[i + 1]))
            if i < len(dims_g) - 2:
                layers_g.append(nn.LeakyReLU())
        self.mlp_predictor = nn.Sequential(*layers_g)

        if self.init_mode == 'near_zero':
            nn.init.uniform_(self.mlp_predictor[-1].weight, -1e-5, 1e-5)
            nn.init.constant_(self.mlp_predictor[-1].bias, 0)
        elif self.init_mode == 'random':
            pass
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}. Must be 'near_zero' or 'random'")

    def interpolate_barycentric(self, attrs, faces, A, B):
        """
        Helper to interpolate attributes using barycentric coordinates.
        Same as in NeuralSubdivisionDecoder.
        """
        B_batch, V, C = attrs.shape
        _, F_count, _ = faces.shape
        
        # Flat indices for gather
        batch_offset = (torch.arange(B_batch, device=attrs.device) * V).view(-1, 1, 1)
        flat_faces = (faces + batch_offset).view(-1)
        flat_attrs = attrs.view(-1, C)
        
        face_attrs = flat_attrs[flat_faces].view(B_batch, F_count, 3, C)
        
        K = A.shape[0] // F_count 
        
        # Reshape for broadcasting
        A_reshaped = A.view(1, F_count, K, 1)
        B_reshaped = B.view(1, F_count, K, 1)
        C_reshaped = 1.0 - A_reshaped - B_reshaped
        
        v0 = face_attrs[:, :, 0, :].unsqueeze(2) # (B, F, 1, C)
        v1 = face_attrs[:, :, 1, :].unsqueeze(2)
        v2 = face_attrs[:, :, 2, :].unsqueeze(2)
        
        interp = v0 * A_reshaped + v1 * B_reshaped + v2 * C_reshaped
        
        return interp.view(B_batch, -1, C)

    def forward(self, base_verts, base_faces, vertex_features, base_normals):
        """
        Args:
            base_verts: (B, V, 3)
            base_faces: (B, F, 3)
            vertex_features: (B, V, D)
            base_normals: (B, V, 3)

        Returns:
            fine_verts: (B, V_fine, 3)
            fine_faces: (F_fine, 3)
            displacements: (B, V_fine, 1) or (B, V_fine, 3)
        """
        B, V, _ = base_verts.shape
        B = int(B)
        _, num_faces, _ = base_faces.shape
        
        # 1. Generate Barycentric Coordinates
        uv_A, uv_B = self.subdivision.sample_uniform_bary(self.rate, num_triangles=num_faces)
        
        # 2. Interpolate Attributes (Linear)
        # lp: Linear Position (B, F*K, 3)
        lp = self.interpolate_barycentric(base_verts, base_faces, uv_A, uv_B)
        
        # ln: Linear Normal (B, F*K, 3)
        ln = self.interpolate_barycentric(base_normals, base_faces, uv_A, uv_B)
        ln = F.normalize(ln, dim=-1, p=2)
        
        # 3. Sum of Features Logic
        
        # 3.1 Gather vertex attributes
        batch_offset = (torch.arange(B, device=base_verts.device) * V).view(-1, 1, 1)
        flat_faces = (base_faces + batch_offset).view(-1)
        
        # Features: (B, F, 3, D)
        flat_feats = vertex_features.view(-1, vertex_features.shape[-1])
        face_feats = flat_feats[flat_faces].view(B, num_faces, 3, -1)
        
        # Vertices: (B, F, 3, 3)
        flat_verts = base_verts.view(-1, 3)
        face_verts = flat_verts[flat_faces].view(B, num_faces, 3, 3)
        
        # Reshape lp to (B, F, K, 3)
        K = uv_A.shape[0] // num_faces
        lp_reshaped = lp.view(B, num_faces, K, 3)
        
        # Initialize Aggregated Feature H
        # H shape: (B, F, K, hidden_dim)
        H_agg = 0
        
        # Loop over 3 vertices to sum mapped features
        for i in range(3):
            # Vertex position: (B, F, 1, 3)
            v_pos = face_verts[:, :, i, :].unsqueeze(2)
            
            # Relative position (Global Relative): (B, F, K, 3)
            delta = lp_reshaped - v_pos
            
            # Feature: (B, F, K, D)
            feat = face_feats[:, :, i, :].unsqueeze(2).expand(-1, -1, K, -1)
            
            # Prepare Input for MLP F
            mlp_f_in_list = [feat]
            
            if self.posenc_mode == 0:
                mlp_f_in_list.append(delta)
            else:
                # Mode 1 or 2: Apply PosEnc to DeltaP
                pe_delta = positional_encoding(delta, [], self.fflevels)
                mlp_f_in_list.append(pe_delta)
            
            # Concat
            mlp_f_in = torch.cat(mlp_f_in_list, dim=-1) # (B, F, K, InputDim_F)
            
            # Map Feature
            # Flatten for MLP: (B*F*K, InputDim_F)
            h_i = self.mlp_feature_map(mlp_f_in.view(-1, mlp_f_in.shape[-1]))
            
            # Reshape back and Accumulate
            h_i = h_i.view(B, num_faces, K, -1)
            H_agg = H_agg + h_i
            
        # 4. Final Prediction using MLP G
        
        # Prepare Input for MLP G
        mlp_g_in_list = [H_agg] # Start with Aggregated Feature
        
        # Add Normal info if needed
        if not self.predict_offset:
            # Reshape ln: (B, F, K, 3)
            ln_reshaped = ln.view(B, num_faces, K, 3)
            
            if self.posenc_mode == 2:
                pe_norm = positional_encoding(ln_reshaped, [], self.fflevels)
                mlp_g_in_list.append(pe_norm)
            else:
                # Mode 0 or 1: Raw Normal
                mlp_g_in_list.append(ln_reshaped)
                
        # Concat
        mlp_g_in = torch.cat(mlp_g_in_list, dim=-1)
        
        # Predict
        out = self.mlp_predictor(mlp_g_in.view(-1, mlp_g_in.shape[-1]))
        
        # Reshape back: (B, F, K, OutDim)
        out = out.view(B, num_faces, K, -1)
        
        # Flatten to (B, F*K, OutDim)
        disp_flat = out.view(B, -1, out.shape[-1])
        
        # 5. Stitch Seams: Average displacements for shared points
        # Compute merge indices using topology
        # merge_idx: (B, F*K) values in [0, Num_Unique-1]
        merge_idx = self.subdivision.compute_merge_indices(base_faces, self.rate)
        
        # Compute averaged displacements
        # disp_flat: (B, N, D)
        # merge_idx: (B, N)
        # We need to flatten Batch dim for scatter_mean, then reshape back
        # But compute_merge_indices already handles batch offset internally for "unique values" logic?
        # WAIT. My compute_merge_indices returns (B, F*K) where indices are LOCAL to each batch item?
        # Let's check compute_merge_indices implementation:
        # It calls `torch.unique` on keys offset by batch. So the indices returned by `return_inverse` 
        # are GLOBAL indices across the whole batch [0, Total_Unique_Across_Batch - 1].
        # So we can just flatten disp and use flattened merge_idx.
        
        disp_all = disp_flat.view(-1, disp_flat.shape[-1]) # (B*N, D)
        idx_all = merge_idx.view(-1) # (B*N,)
        
        # Scatter Mean
        # averaged_disp_unique: (Total_Unique, D)
        averaged_disp_unique = scatter_mean(disp_all, idx_all, dim=0)
        
        # Map back to full points
        # disp_stitched: (B*N, D)
        disp_stitched = averaged_disp_unique[idx_all]
        
        # Reshape to (B, F*K, D)
        disp_flat = disp_stitched.view(B, -1, disp_flat.shape[-1])
        
        # 6. Apply Displacement
        if self.predict_offset:
            # Vector offset
            fine_verts = lp + disp_flat
        else:
            # Scalar displacement
            fine_verts = lp + disp_flat * ln
            
        # Build Topology
        fine_faces = self.subdivision.build_triangulated_faces(self.rate, num_triangles=num_faces)
        
        return fine_verts, fine_faces, disp_flat
