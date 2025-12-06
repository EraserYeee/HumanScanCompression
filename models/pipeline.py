import torch
import torch.nn as nn
from .grouper import LocalPatchGrouper
from .encoder import LocalFeatureEncoder
from .decoder import NeuralSubdivisionDecoder

class Stage2Pipeline(nn.Module):
    """
    Stage 2 完整管线模型。
    Scan Point Cloud -> Base Mesh Anchored Features -> Fine Mesh
    """
    def __init__(self, config):
        super().__init__()
        self.grouper = LocalPatchGrouper()
        self.encoder = LocalFeatureEncoder(
            input_dim=3, 
            hidden_dim=config.get('enc_hidden_dim', 64),
            output_dim=config.get('feature_dim', 128)
        )
        self.decoder = NeuralSubdivisionDecoder(
            feature_dim=config.get('feature_dim', 128),
            levels=config.get('subdivision_levels', 8),
            rate=config.get('subdivision_rate', 4)
        )

    def forward(self, base_verts, base_faces, base_normals, scan_points):
        """
        Args:
            base_verts: (B, V, 3)
            base_faces: (B, F, 3)
            base_normals: (B, V, 3)
            scan_points: (B, P, 3)

        Returns:
            fine_mesh: (verts, faces)
            aux_data: {local_points, cluster_idx, etc.} 用于调试或可视化
        """
        # 1. Grouping
        local_points, cluster_idx = self.grouper(base_verts, base_normals, scan_points)

        # 2. Encoding
        # 注意: num_verts 需要处理 batch 内可能不一致的情况，通常取 max
        B, V, _ = base_verts.shape
        vertex_features = self.encoder(local_points, cluster_idx, num_verts=V)

        # 3. Decoding
        fine_verts, fine_faces, displacements = self.decoder(base_verts, base_faces, vertex_features, base_normals)

        return fine_verts, fine_faces, displacements

