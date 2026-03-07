import torch
import torch.nn as nn
from .grouper import LocalPatchGrouper
from .encoder import LocalFeatureEncoder, AttentiveLocalFeatureEncoder, VAEHead
from .decoder import NeuralSubdivisionDecoder
from .decoder_sumof_feature import SumOfFeatureDecoder

class Stage2Pipeline(nn.Module):
    """
    Stage 2 完整管线模型。
    Scan Point Cloud -> Base Mesh Anchored Features -> Fine Mesh
    
    支持:
    - encoder_type: 'standard' (原始 MaxPool PointNet) 或 'attentive' (多头注意力池化)
    - use_vae: 可选的 VAE 瓶颈头 (解耦设计，可独立开关)
    """
    def __init__(self, config):
        super().__init__()
        # Check if we should predict global offset (xyz) instead of scalar displacement
        predict_offset = config.get('predict_offset', False)
        
        # If predicting global offset, we use global relative coordinates in Grouper (no rotation)
        self.grouper = LocalPatchGrouper(use_global_coordinates=predict_offset)
        
        # Optional extra scan feature
        self.use_scan_normal = config.get('use_scan_normal', False)

        # Check if we should use feature transform (default to True if not specified)
        use_feature_transform = config.get('use_feature_transform', True)

        enc_input_dim = 6 if self.use_scan_normal else 3
        
        # === Encoder Selection ===
        encoder_type = config.get('encoder_type', 'standard')
        print(f"Encoder Type Selected: {encoder_type}")
        
        if encoder_type == 'attentive':
            self.encoder = AttentiveLocalFeatureEncoder(
                input_dim=enc_input_dim,
                hidden_dim=config.get('enc_hidden_dim', 64),
                output_dim=config.get('feature_dim', 128),
                num_attention_heads=config.get('num_attention_heads', 4),
                use_max_pool_residual=config.get('use_max_pool_residual', True)
            )
        else:
            self.encoder = LocalFeatureEncoder(
                input_dim=enc_input_dim, 
                hidden_dim=config.get('enc_hidden_dim', 64),
                output_dim=config.get('feature_dim', 128),
                use_feature_transform=use_feature_transform
            )
        
        # === VAE Head (optional, decoupled) ===
        self.use_vae = config.get('use_vae', False)
        if self.use_vae:
            vae_latent_dim = config.get('vae_latent_dim', 128)
            self.vae_head = VAEHead(
                input_dim=config.get('feature_dim', 128),
                latent_dim=vae_latent_dim
            )
            decoder_feature_dim = vae_latent_dim
            print(f"VAE enabled: feature_dim={config.get('feature_dim', 128)} -> latent_dim={vae_latent_dim}")
        else:
            self.vae_head = None
            decoder_feature_dim = config.get('feature_dim', 128)
        
        # === Decoder Selection ===
        decoder_type = config.get('decoder_type', 'standard')
        print("Pipeline config received: ", config.keys())
        print("Model config keys:", config.keys())
        print(f"Decoder Type Selected: {decoder_type}")
        
        # Initialization mode: 'near_zero' or 'random'
        init_mode = config.get('init_mode', 'near_zero')
        
        common_kwargs = {
            'feature_dim': decoder_feature_dim,
            'levels': config.get('subdivision_levels', 8),
            'rate': config.get('subdivision_rate', 4),
            'predict_offset': predict_offset,
            'posenc_mode': config.get('posenc_mode', 1),
            'init_mode': init_mode
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

    def forward(self, base_verts, base_faces, base_normals, scan_points, scan_normals=None):
        """
        Args:
            base_verts: (B, V, 3)
            base_faces: (B, F, 3)
            base_normals: (B, V, 3)
            scan_points: (B, P, 3)
            scan_normals: (B, P, 3) or None

        Returns:
            fine_verts: (B, V_fine, 3)
            fine_faces: (F_fine, 3)
            displacements: (B, V_fine, 1)
            trans_feat: (B*V, K, K) or None (for regularization loss)
            vertex_features: (B, V, D) (Debug: check variance)
            kl_loss: scalar or None (VAE KL divergence loss)
        """
        # 1. Grouping
        local_points, cluster_idx = self.grouper(base_verts, base_normals, scan_points)

        # 2. Encoding
        # 注意: num_verts 需要处理 batch 内可能不一致的情况，通常取 max
        B, V, _ = base_verts.shape
        encoder_input = local_points
        if self.use_scan_normal and scan_normals is not None:
            encoder_input = torch.cat([encoder_input, scan_normals], dim=-1)
        vertex_features, trans_feat = self.encoder(encoder_input, cluster_idx, num_verts=V)

        # 2.5 VAE Bottleneck (optional)
        kl_loss = None
        if self.vae_head is not None:
            vertex_features, kl_loss = self.vae_head(vertex_features)

        # 3. Decoding
        fine_verts, fine_faces, displacements = self.decoder(base_verts, base_faces, vertex_features, base_normals)

        return fine_verts, fine_faces, displacements, trans_feat, vertex_features, kl_loss
