import torch
import torch.nn as nn
from .grouper import LocalPatchGrouper
from .encoder import LocalFeatureEncoder
from .decoder import NeuralSubdivisionDecoder
from .decoder_sumof_feature import SumOfFeatureDecoder

class Stage2Pipeline(nn.Module):
    """
    Stage 2 完整管线模型。
    Scan Point Cloud -> Base Mesh Anchored Features -> Fine Mesh
    """
    def __init__(self, config):
        super().__init__()
        # Check if we should predict global offset (xyz) instead of scalar displacement
        predict_offset = config.get('predict_offset', False)
        
        # If predicting global offset, we use global relative coordinates in Grouper (no rotation)
        self.grouper = LocalPatchGrouper(use_global_coordinates=predict_offset)
        
        # Check if we should use feature transform (default to True if not specified)
        use_feature_transform = config.get('use_feature_transform', True)
        
        self.encoder = LocalFeatureEncoder(
            input_dim=3, 
            hidden_dim=config.get('enc_hidden_dim', 64),
            output_dim=config.get('feature_dim', 128),
            use_feature_transform=use_feature_transform
        )
        
        # Decoder selection
        # Default to standard NeuralSubdivisionDecoder if not specified
        decoder_type = config.get('decoder_type', 'standard')
        print("Pipeline config received: ", config.keys())
        print("Model config keys:", config.keys())
        print(f"Decoder Type Selected: {decoder_type}")
        
        common_kwargs = {
            'feature_dim': config.get('feature_dim', 128),
            'levels': config.get('subdivision_levels', 8),
            'rate': config.get('subdivision_rate', 4),
            'predict_offset': predict_offset,
            'posenc_mode': config.get('posenc_mode', 1)
        }
        
        if decoder_type == 'sum_of_feature':
            self.decoder = SumOfFeatureDecoder(
                hidden_dim=config.get('dec_hidden_dim', 64),
                **common_kwargs
            )
        else:
            # 0: Local Pos (raw) + Normal (raw)。
            # 1: PosEnc(Local Pos) + Normal (raw)。
            # 2: PosEnc(Local Pos) + PosEnc(Normal)。
            self.decoder = NeuralSubdivisionDecoder(
                    hidden_dim=config.get('dec_hidden_dim', 64),
                    **common_kwargs
            )

    def forward(self, base_verts, base_faces, base_normals, scan_points):
        """
        Args:
            base_verts: (B, V, 3)
            base_faces: (B, F, 3)
            base_normals: (B, V, 3)
            scan_points: (B, P, 3)

        Returns:
            fine_verts: (B, V_fine, 3)
            fine_faces: (F_fine, 3)
            displacements: (B, V_fine, 1)
            trans_feat: (B*V, K, K) or None (for regularization loss)
            vertex_features: (B, V, D) (Debug: check variance)
        """
        # 1. Grouping
        local_points, cluster_idx = self.grouper(base_verts, base_normals, scan_points)

        # 2. Encoding
        # 注意: num_verts 需要处理 batch 内可能不一致的情况，通常取 max
        B, V, _ = base_verts.shape
        vertex_features, trans_feat = self.encoder(local_points, cluster_idx, num_verts=V)

        # 3. Decoding
        fine_verts, fine_faces, displacements = self.decoder(base_verts, base_faces, vertex_features, base_normals)

        return fine_verts, fine_faces, displacements, trans_feat, vertex_features
