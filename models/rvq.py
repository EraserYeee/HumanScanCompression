"""Residual Vector Quantization (RVQ) for Stage2 face/vertex features.

灵感来源: Mimi / SoundStream / EnCodec 的 RVQ。核心是用 Q 个码本逐层逼近
连续特征 z, 每层只编码"上一层的残差":

    r0 = z
    q_i = VQ_i(r_{i-1});  r_i = r_{i-1} - e(q_i)
    重建: z_hat = sum_i e(q_i)

由于后层只编码前层残差(量纲递减), 第 1 层被迫抓主成分(低频/整体位移倾向),
后层抓越来越细的细节 —— 天然的"由粗到细"归纳偏置。

设计要点(均为成熟做法, 避免踩坑):
- EMA 码本更新(不靠梯度更新码字, 更稳); commitment loss 把编码器拉向码字。
- STE 直通梯度: z_hat = z + (z_hat - z).detach(), 让 decoder 的梯度直达 encoder。
- dead-code 重启: 长期未被使用的码字, 用当前 batch 的随机样本重新初始化。
- 渐进式层激活: num_active 可在训练中从 1 递增到 Q, 稳定训练 (progressive RVQ)。

关键: 这是"重建/压缩"用途, q_1..q_Q 由 scan 一次前馈算出、全部已知,
解码端 sum_i e(q_i) 纯并行求和, **没有自回归**, 因此不存在"串行难学"问题。

接口对齐 pipeline: forward 吃 (B, N, D), 吐 (z_hat (B,N,D), indices (B,N,Q), vq_loss 标量)。
关掉时(use_rvq=False) pipeline 根本不构造本模块, 零影响。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """单个码本的向量量化层, 使用 EMA 更新码字。

    Args:
        dim: 特征维度 D
        codebook_size: 码本大小 N (每个索引占 log2(N) bit)
        decay: EMA 衰减系数
        eps: laplace smoothing 防止除零
        commitment_weight: commitment loss 权重 (相对本层)
        threshold_dead: 码字 cluster_size 低于该阈值视为 dead, 触发重启
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 2048,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_weight: float = 0.25,
        threshold_dead: float = 1.0,
        dead_code_warmup_steps: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.commitment_weight = commitment_weight
        self.threshold_dead = threshold_dead
        self.dead_code_warmup_steps = dead_code_warmup_steps

        # 码本作为 buffer(非参数, 靠 EMA 更新), 随 state_dict 保存/加载
        embed = torch.randn(codebook_size, dim)
        self.register_buffer("embed", embed)                        # (N, D)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))   # (N,)
        self.register_buffer("embed_avg", embed.clone())            # (N, D)
        self.register_buffer("initialized", torch.tensor(0.0))      # 是否已用数据初始化
        # 训练 step 计数(仅训练分支自增), 用于 dead-code warmup。
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _init_embed_from_data(self, x: torch.Tensor):
        """用首个 batch 的样本随机初始化码本(kmeans++ 的廉价替代)。"""
        n = x.shape[0]
        if n >= self.codebook_size:
            idx = torch.randperm(n, device=x.device)[: self.codebook_size]
            chosen = x[idx]
        else:
            # 样本不足时, 有放回采样
            idx = torch.randint(0, n, (self.codebook_size,), device=x.device)
            chosen = x[idx]
        self.embed.copy_(chosen)
        self.embed_avg.copy_(chosen)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(1.0)

    @torch.no_grad()
    def _expire_dead_codes(self, x: torch.Tensor):
        """把长期未使用的码字, 用当前 batch 的随机样本替换。"""
        dead = self.cluster_size < self.threshold_dead
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        n = x.shape[0]
        idx = torch.randint(0, n, (n_dead,), device=x.device)
        self.embed[dead] = x[idx]
        self.embed_avg[dead] = x[idx]
        self.cluster_size[dead] = 1.0

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (M, D) 扁平化后的待量化向量
        Returns:
            quantized: (M, D) 量化后向量(**未做 STE**, 由外层 ResidualVQ 统一处理)
            indices:   (M,)   码字索引
            loss:      标量    commitment loss

        重要约定:
            - 这里只返回 "纯量化值" e(q), 不做单层 STE。
            - 单层 STE 会让 ResidualVQ 累加后梯度被放大 num_active 倍, encoder 梯度爆。
            - STE 由 ResidualVQ.forward 在所有层求和后做一次 (d z_hat / d z = 1)。
            - commitment loss 仍在本层计算: encoder 输出(此层为 residual) 应靠近自己被指派到的码字。
        """
        flatten = x.reshape(-1, self.dim)
        # 量化路径(最近邻搜索 + EMA 码本更新)不应参与 autograd。
        # 若这里错误地追踪计算图，会让图跨 step 挂在模块 buffer 上，出现显存缓慢爬升。
        flatten_detached = flatten.detach()

        if self.training and self.initialized.item() == 0.0:
            self._init_embed_from_data(flatten_detached)

        # L2 最近邻搜索：在 no_grad 下做，避免为 argmin 前的大矩阵建立无用计算图。
        with torch.no_grad():
            dist = (
                flatten_detached.pow(2).sum(1, keepdim=True)
                - 2 * flatten_detached @ self.embed.t()
                + self.embed.pow(2).sum(1, keepdim=True).t()
            )                                                      # (M, N)
            indices = dist.argmin(dim=1)                           # (M,)
        quantized = self.embed[indices]                            # (M, D)  注: embed 是 buffer 不带梯度

        # EMA 更新码本(仅训练): 必须 no_grad，且只用 detached 特征统计。
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.codebook_size).type(flatten_detached.dtype)  # (M, N)
                cluster_size_new = onehot.sum(0)                   # (N,)
                embed_sum = onehot.t() @ flatten_detached          # (N, D)

                self.cluster_size.mul_(self.decay).add_(
                    cluster_size_new, alpha=1 - self.decay
                )
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # laplace smoothing 归一化
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))

                # dead-code 重启需要 warmup: 训练初期 cluster_size 还没爬起来,
                # 立刻判死会把整个码本疯狂洗牌, 码字漂移加剧不稳定。
                self.step_count.add_(1)
                if int(self.step_count.item()) > self.dead_code_warmup_steps:
                    self._expire_dead_codes(flatten_detached)

        # commitment loss: 把 encoder 输出(本层输入 residual) 拉向被指派的码字
        # (码字不靠梯度更新, 故 detach)
        loss = self.commitment_weight * F.mse_loss(quantized.detach(), flatten)

        # 关键: 此处 **不做 STE**。返回 "纯 e(q)" 给上层 ResidualVQ 累加。
        # ResidualVQ 在求和后做一次统一 STE, 保证 d z_hat / d z = 1。
        return quantized, indices, loss


class ResidualVQ(nn.Module):
    """残差向量量化: 堆叠 Q 个 VectorQuantizerEMA, 逐层编码残差。

    Args:
        dim: 特征维度 D
        num_quantizers: 层数 Q (压缩率/精度权衡; 也是"由粗到细"的层数)
        codebook_size: 每层码本大小 N
        decay/commitment_weight/threshold_dead: 透传给每层 VQ
    """

    def __init__(
        self,
        dim: int,
        num_quantizers: int = 8,
        codebook_size: int = 2048,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        threshold_dead: float = 1.0,
        dead_code_warmup_steps: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.layers = nn.ModuleList(
            [
                VectorQuantizerEMA(
                    dim=dim,
                    codebook_size=codebook_size,
                    decay=decay,
                    commitment_weight=commitment_weight,
                    threshold_dead=threshold_dead,
                    dead_code_warmup_steps=dead_code_warmup_steps,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z: torch.Tensor, num_active: int | None = None):
        """
        Args:
            z: (B, N, D) 连续特征 (encoder 输出)
            num_active: 本次激活的层数(渐进式训练用); None 表示全部 Q 层
        Returns:
            z_hat:   (B, N, D) 重建特征(带 STE 直通梯度, d z_hat / d z = 1)
            indices: (B, N, num_active) 各层码字索引(long), 用于压缩存储 / 比特统计
            vq_loss: 标量, 各激活层 commitment loss 的均值

        梯度合约(关键修复, 见单层 forward 注释):
            - 单层 VQ 不做 STE, 返回纯 e(q_i)。
            - z_hat_sum = Σ_i e(q_i)  (前向值: 每层逼近残差, 累加约等于原 z)
            - 在外层做一次统一 STE: z_hat = flat + (z_hat_sum - flat).detach()
            - 这样 d z_hat / d flat = 1, 与单层 VQ 直觉一致, 不会随 num_active 放大。
        """
        if num_active is None:
            num_active = self.num_quantizers
        num_active = max(1, min(num_active, self.num_quantizers))

        B, N, D = z.shape
        flat = z.reshape(-1, D)                                    # (B*N, D)

        # residual 必须保留对 flat 的梯度: 这样每层 commitment loss
        #   loss_i = mse(e(q_i).detach(), residual_i)
        # 才能把梯度传回 encoder, 把 encoder 输出拉向各层码字。
        # 由于 quantized 本身已 detach(单层 VQ 不再做 STE), residual = flat - Σ q_j.detach()
        # 对 flat 的梯度恒为 1, 各层 commitment loss 都能正确反传。
        residual = flat
        z_hat_sum = torch.zeros_like(flat)                         # 求和值, 由 codebook 查得无梯度
        vq_loss = z.new_tensor(0.0)
        idx_list = []

        for i in range(num_active):
            quantized, indices, loss = self.layers[i](residual)
            # quantized 来自 buffer 查表, 本身就不带梯度; 这里 detach 一次保险。
            residual = residual - quantized.detach()
            z_hat_sum = z_hat_sum + quantized
            vq_loss = vq_loss + loss
            idx_list.append(indices)

        vq_loss = vq_loss / num_active

        # 统一 STE: z_hat 前向 = 各层量化值之和, 反向把 render 梯度 1:1 直通到 flat。
        # 这是 RVQ 在重建/压缩任务上的标准做法, 避免 d z_hat / d flat = num_active 放大。
        z_hat = flat + (z_hat_sum - flat).detach()

        indices = torch.stack(idx_list, dim=-1)                    # (B*N, num_active)
        z_hat = z_hat.reshape(B, N, D)
        indices = indices.reshape(B, N, num_active)
        return z_hat, indices, vq_loss

    @torch.no_grad()
    def codebook_usage(self) -> list[float]:
        """返回每层码本的使用率(被激活码字占比), 用于诊断 dead-code。"""
        usage = []
        for layer in self.layers:
            used = (layer.cluster_size > layer.threshold_dead).float().mean().item()
            usage.append(used)
        return usage

    def bits_per_anchor(self, num_active: int | None = None) -> float:
        """每个 anchor(face/vertex) 的存储比特数 = num_active * log2(N)。"""
        import math
        if num_active is None:
            num_active = self.num_quantizers
        return num_active * math.log2(self.codebook_size)
