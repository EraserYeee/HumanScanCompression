import torch
import torch.nn as nn
from .grouper import LocalPatchGrouper, VertexKNNGrouper, FaceKNNGrouper, FaceAutoGrouper, FaceLocalGrouper
from .timing_hooks import prof_start, prof_split, profile_is_rank0
from .encoder import (
    LocalFeatureEncoder,
    AttentiveLocalFeatureEncoder,
    CrossAttentionFeatureEncoder,
    PTSAEncoder,
    PTFlashHierarchicalEncoder,
    VAEHead,
)
from .decoder import NeuralSubdivisionDecoder
from .decoder_sumof_feature import SumOfFeatureDecoder
from .decoder_face_triangle import FaceTriangleDecoder
from .rvq import ResidualVQ
from .base_displacement import BaseVertexDisplacementHead

class Stage2Pipeline(nn.Module):
    """
    Stage 2 完整管线模型。
    Scan Point Cloud -> Base Mesh Anchored Features -> Fine Mesh
    
    支持:
    - encoder_type: 'standard' | 'attentive' | 'cross_attention' (MaxPool-as-Q CA)
    - use_vae: 可选的 VAE 瓶颈头 (解耦设计，可独立开关)
    """
    def __init__(self, config):
        super().__init__()
        # Check if we should predict global offset (xyz) instead of scalar displacement
        predict_offset = config.get('predict_offset', False)
        
        # === Encoding Mode ===
        encoding_mode = config.get('encoding_mode', 'vertex')
        self.encoding_mode = encoding_mode

        # === Grouper Selection ===
        grouper_type = config.get('grouper_type', 'scan_knn')
        self.grouper_type = grouper_type

        if encoding_mode == 'face':
            if grouper_type == 'vertex_knn':
                knn_k = config.get('vertex_knn_k', 512)
                knn_chunk = config.get('vertex_knn_chunk_size', 512)
                self.grouper = FaceKNNGrouper(k=knn_k, knn_chunk_size=knn_chunk)
                print(f"Grouper: FaceKNNGrouper (K={knn_k}, chunk={knn_chunk})")
            elif grouper_type == 'auto_knn':
                min_pts = config.get('auto_knn_min_pts', 16)
                knn_chunk = config.get('vertex_knn_chunk_size', 512)
                self.grouper = FaceAutoGrouper(min_pts=min_pts, knn_chunk_size=knn_chunk)
                print(f"Grouper: FaceAutoGrouper (min_pts={min_pts}, buckets={FaceAutoGrouper.BUCKET_SIZES})")
            else:
                self.grouper = FaceLocalGrouper()
                print("Grouper: FaceLocalGrouper (scan_knn, face centroids)")
        else:
            if grouper_type == 'vertex_knn':
                knn_k = config.get('vertex_knn_k', 512)
                knn_chunk = config.get('vertex_knn_chunk_size', 512)
                self.grouper = VertexKNNGrouper(k=knn_k, knn_chunk_size=knn_chunk)
                print(f"Grouper: VertexKNNGrouper (K={knn_k}, chunk={knn_chunk})")
            else:
                self.grouper = LocalPatchGrouper(use_global_coordinates=predict_offset)
                print("Grouper: LocalPatchGrouper (scan_knn)")
        
        # Optional extra scan feature
        self.use_scan_normal = config.get('use_scan_normal', False)

        # Check if we should use feature transform (default to True if not specified)
        use_feature_transform = config.get('use_feature_transform', True)

        enc_input_dim = 6 if self.use_scan_normal else 3
        
        # === Encoder Selection ===
        encoder_type = config.get('encoder_type', 'standard')
        print(f"Encoder Type Selected: {encoder_type}")
        
        if encoder_type in ('pt_sa_attentive', 'pt_sa_max'):
            pooling = 'attentive' if encoder_type == 'pt_sa_attentive' else 'max'
            hidden_dims = config.get('enc_hidden_dim', [64])
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims]
            self.encoder = PTSAEncoder(
                input_dim=enc_input_dim,
                hidden_dims=hidden_dims,
                sa_dim=config.get('pt_sa_dim', 128),
                num_heads=config.get('pt_sa_num_heads', 4),
                num_sa_layers=config.get('pt_sa_num_layers', 1),
                feature_dim=config.get('feature_dim', 512),
                pooling_type=pooling,
                use_ffn=config.get('pt_sa_use_ffn', True),
                pe_num_frequencies=config.get('pt_sa_pe_frequencies', 4),
            )
        elif encoder_type == 'pt_hier_flash':
            hidden_dims = config.get('enc_hidden_dim', [64])
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims]
            self.encoder = PTFlashHierarchicalEncoder(
                input_dim=enc_input_dim,
                hidden_dims=hidden_dims,
                sa_dim=config.get('pt_sa_dim', 128),
                num_heads=config.get('pt_sa_num_heads', 4),
                feature_dim=config.get('feature_dim', 512),
                pe_num_frequencies=config.get('pt_sa_pe_frequencies', 4),
            )
        elif encoder_type == 'cross_attention':
            self.encoder = CrossAttentionFeatureEncoder(
                input_dim=enc_input_dim,
                hidden_dim=config.get('enc_hidden_dim', 64),
                output_dim=config.get('feature_dim', 128),
                num_heads=config.get('num_attention_heads', 4),
                point_embed_hidden_dim=config.get('point_embed_hidden_dim', 48),
                use_ffn=config.get('encoder_use_ffn', True)
            )
        elif encoder_type == 'attentive':
            self.encoder = AttentiveLocalFeatureEncoder(
                input_dim=enc_input_dim,
                hidden_dim=config.get('enc_hidden_dim', 64),
                output_dim=config.get('feature_dim', 128),
                num_attention_heads=config.get('num_attention_heads', 4),
                use_max_pool_residual=config.get('use_max_pool_residual', True),
                attention_temperature=config.get('attention_temperature', 2.0),
                score_init_scale=config.get('score_init_scale', 0.1),
                score_clip_value=config.get('score_clip_value', 5.0)
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

        # === Residual VQ (optional, decoupled) ===
        # 插在 encoder(/VAE) 与 decoder 之间, 把"由粗到细分层"放进 latent 空间:
        # 第 1 层抓低频整体位移, 后层抓细节; 存储的是码字索引而非连续 feature。
        # 关掉时(use_rvq=False)不构造, 对现有训练零影响。
        self.use_rvq = config.get('use_rvq', False)
        if self.use_rvq:
            self.rvq = ResidualVQ(
                dim=decoder_feature_dim,
                num_quantizers=config.get('rvq_num_quantizers', 8),
                codebook_size=config.get('rvq_codebook_size', 2048),
                decay=config.get('rvq_decay', 0.99),
                commitment_weight=config.get('rvq_commitment_weight', 0.25),
                threshold_dead=config.get('rvq_threshold_dead', 1.0),
                dead_code_warmup_steps=config.get('rvq_dead_code_warmup_steps', 1000),
            )
            # 渐进式层激活: 训练中由外部(train loop)设置 self._rvq_num_active;
            # None 表示用满 num_quantizers 层。
            self._rvq_num_active = None
            print(
                f"RVQ enabled: dim={decoder_feature_dim}, "
                f"Q={config.get('rvq_num_quantizers', 8)}, "
                f"N={config.get('rvq_codebook_size', 2048)}"
            )
        else:
            self.rvq = None
        self._last_vq_loss = None
        self._last_rvq_indices = None
        # RVQ / feature 诊断标量(供 train loop 写日志, 监控崩塌前兆)
        self._last_rvq_recon_err = None
        self._last_feat_norm = None

        # === Base Vertex Displacement Head (2.6 轻量两段式, 可选, 解耦) ===
        # 先把 base 顶点"掰正"到更贴合 scan 的位置, fine decoder 再在 corrected
        # base 上细分位移。与 fine decoder 权重独立。详见 plan 文档 Part 2。
        self.use_base_displacement = config.get('use_base_displacement', False)
        if self.use_base_displacement:
            self.base_disp_head = BaseVertexDisplacementHead(
                feature_dim=decoder_feature_dim,
                hidden_dim=config.get('base_disp_hidden_dim', [128, 64]),
                init_scale=config.get('base_disp_init_scale', 1e-4),
                use_normal=config.get('base_disp_use_normal', True),
            )
            print(
                f"BaseVertexDisplacementHead enabled (feature_dim={decoder_feature_dim}, "
                f"reencode=False [2.6 lightweight tier])"
            )
        else:
            self.base_disp_head = None
        self._last_base_disp = None
        self._last_base_disp_reg = None
        
        # === Decoder Selection ===
        decoder_type = config.get('decoder_type', 'standard')
        print("Pipeline config received: ", config.keys())
        print("Model config keys:", config.keys())
        print(f"Decoder Type Selected: {decoder_type}, encoding_mode: {encoding_mode}")
        
        # Initialization mode: 'near_zero' or 'random'
        init_mode = config.get('init_mode', 'near_zero')
        
        if encoding_mode == 'face':
            ortho = config.get('face_ortho_frame', False)
            stitch_gc = config.get('face_stitch_grad_compensate', True)
            self.decoder = FaceTriangleDecoder(
                feature_dim=decoder_feature_dim,
                hidden_dim=config.get('dec_hidden_dim', 64),
                levels=config.get('subdivision_levels', 8),
                rate=config.get('subdivision_rate', 4),
                posenc_mode=config.get('posenc_mode', 1),
                init_mode=init_mode,
                ortho_frame=ortho,
                stitch_grad_compensate=stitch_gc,
            )
            print(
                f"Decoder: FaceTriangleDecoder (ortho_frame={ortho}, stitch_grad_compensate={stitch_gc})"
            )
        else:
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
                self.decoder = NeuralSubdivisionDecoder(
                        hidden_dim=config.get('dec_hidden_dim', 64),
                        **common_kwargs
                )

    def forward(self, base_verts, base_faces, base_normals, scan_points, scan_normals=None, 
                return_attention=False, return_diagnostics=False):
        """
        Args:
            base_verts: (B, V, 3)
            base_faces: (B, F, 3)
            base_normals: (B, V, 3)
            scan_points: (B, P, 3)
            scan_normals: (B, P, 3) or None
            return_attention: bool, 是否返回注意力分数（仅当 encoder_type='attentive' 时有效）
            return_diagnostics: bool, 是否返回诊断信息（attentive / cross_attention 时有效）

        Returns:
            fine_verts: (B, V_fine, 3)
            fine_faces: (F_fine, 3)
            displacements: (B, V_fine, 1)
            trans_feat: (B*V, K, K) or None (for regularization loss)
            vertex_features: (B, V, D) (Debug: check variance)
            kl_loss: scalar or None (VAE KL divergence loss)
            attention_scores: (B*P, H) or None, 注意力分数（仅当 return_attention=True 且 encoder_type='attentive'）
            global_cluster_idx: (B*P,) or None, 全局 cluster 索引（仅当 return_attention=True 且 encoder_type='attentive'）
            diagnostics: dict or None, 诊断信息（仅当 return_diagnostics=True 且 encoder 为 attentive/cross_attention）
        """
        do_log = bool(getattr(self, "_profile_timing", False))
        if do_log:
            self._profile_counter = getattr(self, "_profile_counter", 0) + 1
            iv = int(getattr(self, "_profile_interval", 50))
            do_log = (self._profile_counter % max(iv, 1)) == 0 and profile_is_rank0()
        for _sub in (self.grouper, self.encoder, self.decoder):
            setattr(_sub, "_profile_do_log", do_log)

        # 1. Grouping
        B, V, _ = base_verts.shape
        _, num_faces_dim, _ = base_faces.shape
        t0 = prof_start() if do_log else None

        is_face = self.encoding_mode == 'face'
        is_vertex_knn = self.grouper_type == 'vertex_knn'
        is_auto_knn = self.grouper_type == 'auto_knn'
        is_pt_sa = isinstance(self.encoder, (PTSAEncoder, PTFlashHierarchicalEncoder))

        if is_face:
            if is_auto_knn:
                auto_buckets = self.grouper(
                    base_verts, base_faces, scan_points,
                    scan_normals=scan_normals if self.use_scan_normal else None,
                )
            elif is_vertex_knn:
                grouped_features, local_coords = self.grouper(
                    base_verts, base_faces, scan_points,
                    scan_normals=scan_normals if self.use_scan_normal else None,
                )
            else:
                local_points, cluster_idx = self.grouper(
                    base_verts, base_faces, scan_points,
                    scan_normals=scan_normals if self.use_scan_normal else None,
                )
        elif is_vertex_knn:
            grouped_features, local_coords = self.grouper(
                base_verts, base_normals, scan_points,
                scan_normals=scan_normals if self.use_scan_normal else None,
            )
        else:
            local_points, cluster_idx = self.grouper(base_verts, base_normals, scan_points)
        if do_log:
            prof_split(True, t0, "grouper", "pipe")

        # 2. Encoding
        diagnostics = None
        attention_scores, global_cluster_idx = None, None

        is_attentive = isinstance(self.encoder, AttentiveLocalFeatureEncoder)
        is_cross_attn = isinstance(self.encoder, CrossAttentionFeatureEncoder)

        num_anchors = num_faces_dim if is_face else V

        if is_auto_knn and is_pt_sa:
            t0 = prof_start() if do_log else None
            feat_dim = self.encoder.feature_dim
            vertex_features = torch.zeros(B, num_faces_dim, feat_dim,
                                          device=base_verts.device)
            for gf_bk, lc_bk, fidx_bk in auto_buckets:
                bk_feat, _ = self.encoder(gf_bk, lc_bk)
                vertex_features[0, fidx_bk[0]] = bk_feat[0].to(
                    dtype=vertex_features.dtype
                )            
            trans_feat = None
            if do_log:
                prof_split(True, t0, "encoder(total)", "pipe")
        elif is_pt_sa:
            t0 = prof_start() if do_log else None
            vertex_features, trans_feat = self.encoder(grouped_features, local_coords)
            if do_log:
                prof_split(True, t0, "encoder(total)", "pipe")
        else:
            # Legacy scatter-based encoders
            encoder_input = local_points
            if self.use_scan_normal and scan_normals is not None:
                encoder_input = torch.cat([encoder_input, scan_normals], dim=-1)

            if return_attention and is_attentive:
                t0 = prof_start() if do_log else None
                result = self.encoder(
                    encoder_input, cluster_idx, num_verts=num_anchors,
                    return_attention=True, return_diagnostics=return_diagnostics
                )
                if do_log:
                    prof_split(True, t0, "encoder(total)", "pipe")
                if return_diagnostics:
                    vertex_features, trans_feat, attention_scores, global_cluster_idx, diagnostics = result
                else:
                    vertex_features, trans_feat, attention_scores, global_cluster_idx = result
            elif return_diagnostics and is_attentive:
                t0 = prof_start() if do_log else None
                vertex_features, trans_feat, diagnostics = self.encoder(
                    encoder_input, cluster_idx, num_verts=num_anchors, return_diagnostics=True
                )
                if do_log:
                    prof_split(True, t0, "encoder(total)", "pipe")
            elif is_cross_attn:
                t0 = prof_start() if do_log else None
                if return_diagnostics:
                    vertex_features, trans_feat, diagnostics = self.encoder(
                        encoder_input, cluster_idx, num_verts=num_anchors, return_diagnostics=True
                    )
                else:
                    vertex_features, trans_feat = self.encoder(
                        encoder_input, cluster_idx, num_verts=num_anchors
                    )
                if do_log:
                    prof_split(True, t0, "encoder(total)", "pipe")
            else:
                t0 = prof_start() if do_log else None
                vertex_features, trans_feat = self.encoder(encoder_input, cluster_idx, num_verts=num_anchors)
                if do_log:
                    prof_split(True, t0, "encoder(total)", "pipe")

        # 2.5 VAE Bottleneck (optional)
        kl_loss = None
        if self.vae_head is not None:
            t0 = prof_start() if do_log else None
            vertex_features, kl_loss = self.vae_head(vertex_features)
            if do_log:
                prof_split(True, t0, "vae_head", "pipe")

        # 2.6 Residual VQ (optional): 分层量化, 损失走属性通道(_last_vq_loss),
        # 不改变本 forward 的返回签名(保持与 infer/overfit 的 6 元组解包兼容)。
        self._last_vq_loss = None
        self._last_rvq_indices = None
        self._last_rvq_recon_err = None
        self._last_feat_norm = None
        if self.rvq is not None:
            t0 = prof_start() if do_log else None
            # 崩塌监控: 量化前特征范数 + 量化重建误差(详见 plan Part 1.3)
            with torch.no_grad():
                feat_pre = vertex_features
                self._last_feat_norm = feat_pre.detach().norm(dim=-1).mean()
            vertex_features, rvq_indices, vq_loss = self.rvq(
                vertex_features, num_active=self._rvq_num_active
            )
            self._last_vq_loss = vq_loss
            self._last_rvq_indices = rvq_indices
            with torch.no_grad():
                self._last_rvq_recon_err = (
                    (vertex_features.detach() - feat_pre.detach()).norm(dim=-1).mean()
                )
            if do_log:
                prof_split(True, t0, "rvq", "pipe")
        else:
            with torch.no_grad():
                self._last_feat_norm = vertex_features.detach().norm(dim=-1).mean()

        # 2.7 Base Vertex Displacement (optional, 2.6 轻量档): 先移 base 顶点,
        # fine decoder 在 corrected base 上细分。face features **不** re-encode。
        self._last_base_disp = None
        self._last_base_disp_reg = None
        decode_base_verts = base_verts
        if self.base_disp_head is not None:
            t0 = prof_start() if do_log else None
            base_disp = self.base_disp_head(
                base_verts, base_faces, vertex_features, base_normals
            )
            decode_base_verts = base_verts + base_disp
            self._last_base_disp = base_disp
            self._last_base_disp_reg = (base_disp ** 2).sum(dim=-1).mean()
            if do_log:
                prof_split(True, t0, "base_disp_head", "pipe")

        # 3. Decoding (在 corrected base 上细分; lp / face basis 随 base 位移更新)
        t0 = prof_start() if do_log else None
        fine_verts, fine_faces, displacements = self.decoder(decode_base_verts, base_faces, vertex_features, base_normals)
        if do_log:
            prof_split(True, t0, "decoder(total)", "pipe")

        if return_attention and is_attentive:
            if return_diagnostics:
                return fine_verts, fine_faces, displacements, trans_feat, vertex_features, kl_loss, attention_scores, global_cluster_idx, diagnostics
            else:
                return fine_verts, fine_faces, displacements, trans_feat, vertex_features, kl_loss, attention_scores, global_cluster_idx
        elif return_diagnostics and is_attentive:
            return fine_verts, fine_faces, displacements, trans_feat, vertex_features, kl_loss, diagnostics
        elif return_diagnostics and is_cross_attn:
            return fine_verts, fine_faces, displacements, trans_feat, vertex_features, kl_loss, diagnostics
        else:
            return fine_verts, fine_faces, displacements, trans_feat, vertex_features, kl_loss
