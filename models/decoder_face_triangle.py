import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.subdivision import BarycentricSubdivision
from torch_scatter import scatter_mean
from .timing_hooks import prof_start, prof_split
from .grouper import _compute_face_basis


def positional_encoding_2d(uv: torch.Tensor, levels: int) -> torch.Tensor:
    """Fourier positional encoding for 2D parametric coordinates (u, v).

    Args:
        uv: (..., 2)
        levels: number of frequency bands

    Returns:
        (..., 2 * 2 * levels)  sin/cos for each frequency and each input dim
    """
    parts = []
    for i in range(levels):
        k = (2.0 ** i) * math.pi
        parts.append(torch.sin(k * uv))
        parts.append(torch.cos(k * uv))
    return torch.cat(parts, dim=-1)


class FaceTriangleDecoder(nn.Module):
    """Decoder that operates in per-face triangle parametric coordinates.

    For each subdivision point (u_sub, v_sub) inside a face:
      1.  MLP predicts (D1, D2, D3) from [face_feat, PE(u_sub, v_sub)]
      2.  ortho_frame=True  → disp = D1*e1_hat + D2*e2_perp + D3*n  (orthonormal)
          ortho_frame=False → disp = D1*e1    + D2*e2      + D3*n  (parametric)
      3.  Base position = v0 + u_sub*e1 + v_sub*e2
      4.  Shared seam points are averaged via scatter_mean on world displacements
      5.  Fine position = base position + stitched displacement
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim=64,
        levels: int = 8,
        rate: int = 4,
        posenc_mode: int = 1,
        init_mode: str = "near_zero",
        predict_offset: bool = False,
        ortho_frame: bool = True,
    ):
        super().__init__()
        self.fflevels = levels
        self.rate = rate
        self.posenc_mode = posenc_mode
        self.init_mode = init_mode
        self.ortho_frame = ortho_frame
        self.subdivision = BarycentricSubdivision()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim, hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if posenc_mode == 0:
            pe_dim = 2
        else:
            pe_dim = 2 * 2 * levels

        input_dim = feature_dim + pe_dim
        out_dim = 3

        layers = []
        dims = [input_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)

        if init_mode == "near_zero":
            nn.init.uniform_(self.mlp[-1].weight, -1e-5, 1e-5)
            nn.init.constant_(self.mlp[-1].bias, 0)
        elif init_mode == "random":
            pass
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

    def forward(self, base_verts, base_faces, face_features, base_normals):
        """
        Args:
            base_verts:    (B, V, 3)
            base_faces:    (B, F, 3) LongTensor
            face_features: (B, F, D)  per-face feature from encoder
            base_normals:  (B, V, 3)  vertex normals (unused, kept for API compat)

        Returns:
            fine_verts:    (B, V_fine, 3)
            fine_faces:    (F_fine, 3)
            displacements: (B, V_fine, 3)  world-space displacements
        """
        B_batch = int(base_verts.shape[0])
        _, num_faces, _ = base_faces.shape
        do_log = getattr(self, "_profile_do_log", False)
        t0 = prof_start() if do_log else None

        v0, e1, e2, n, _, _, _, _ = _compute_face_basis(base_verts, base_faces)

        if self.ortho_frame:
            disp_b1 = F.normalize(e1, dim=-1)                      # (B, F, 3)
            disp_b2 = torch.cross(n, disp_b1, dim=-1)              # ⊥ n and e1_hat
        else:
            disp_b1 = e1
            disp_b2 = e2

        uv_A, uv_B = self.subdivision.sample_uniform_bary(
            self.rate, num_triangles=num_faces
        )
        K = uv_A.shape[0] // num_faces

        # Parametric coords: p = v0 + u*e1 + v*e2  =>  u = B, v = 1-A-B
        u_param = uv_B.view(num_faces, K)                          # (F, K)
        v_param = (1.0 - uv_A - uv_B).view(num_faces, K)          # (F, K)

        # Base subdivision positions in world coords (uses original e1, e2)
        u_3d = u_param.view(1, num_faces, K, 1)                    # broadcast with B
        v_3d = v_param.view(1, num_faces, K, 1)
        lp = (
            v0.unsqueeze(2)
            + u_3d * e1.unsqueeze(2)
            + v_3d * e2.unsqueeze(2)
        )                                                           # (B, F, K, 3)

        # MLP input: [face_feat, PE(u, v)]
        uv = torch.stack([u_param, v_param], dim=-1)               # (F, K, 2)
        if self.posenc_mode == 0:
            pos_enc = uv.unsqueeze(0).expand(B_batch, -1, -1, -1)  # (B, F, K, 2)
        else:
            pos_enc = positional_encoding_2d(uv, self.fflevels)     # (F, K, pe_dim)
            pos_enc = pos_enc.unsqueeze(0).expand(B_batch, -1, -1, -1)

        feat_exp = face_features.unsqueeze(2).expand(-1, -1, K, -1)  # (B, F, K, D)
        mlp_in = torch.cat([feat_exp, pos_enc], dim=-1)             # (B, F, K, D+pe)

        out = self.mlp(mlp_in.reshape(-1, mlp_in.shape[-1]))
        out = out.view(B_batch, num_faces, K, 3)                    # (D1, D2, D3)

        d1 = out[..., 0:1]                                         # (B, F, K, 1)
        d2 = out[..., 1:2]
        d3 = out[..., 2:3]
        disp_world = (
            d1 * disp_b1.unsqueeze(2)
            + d2 * disp_b2.unsqueeze(2)
            + d3 * n.unsqueeze(2)
        )                                                           # (B, F, K, 3)

        disp_flat = disp_world.view(B_batch, -1, 3)                # (B, F*K, 3)
        lp_flat = lp.view(B_batch, -1, 3)                          # (B, F*K, 3)

        # Stitch seams via scatter_mean on world-space displacements
        merge_idx = self.subdivision.compute_merge_indices(base_faces, self.rate)

        disp_all = disp_flat.reshape(-1, 3)                         # (B*F*K, 3)
        idx_all = merge_idx.view(-1)                                # (B*F*K,)

        averaged = scatter_mean(disp_all, idx_all, dim=0)
        disp_stitched = averaged[idx_all].view(B_batch, -1, 3)

        if do_log:
            t0 = prof_split(do_log, t0, "face_tri_mlp_stitch", "dec")

        fine_verts = lp_flat + disp_stitched

        fine_faces = self.subdivision.build_triangulated_faces(
            self.rate, num_triangles=num_faces
        )
        if do_log:
            prof_split(do_log, t0, "face_tri_topology", "dec")

        return fine_verts, fine_faces, disp_stitched
