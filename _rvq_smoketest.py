"""RVQ 冒烟测试: 验证 ResidualVQ 的前向形状、STE 梯度回传、损失、由粗到细性质。
独立运行, 不依赖项目其它模块(rvq.py 是自包含的)。
"""
import sys, math
import torch

sys.path.insert(0, "C:/Users/ruisiye/Documents/scan/HumanScanCompression")
from models.rvq import ResidualVQ, VectorQuantizerEMA

torch.manual_seed(0)

def test_shapes_and_grad():
    B, N, D, Q, Ncb = 1, 200, 64, 8, 512
    rvq = ResidualVQ(dim=D, num_quantizers=Q, codebook_size=Ncb)
    rvq.train()
    z = torch.randn(B, N, D, requires_grad=True)
    z_hat, idx, vq_loss = rvq(z)
    assert z_hat.shape == (B, N, D), z_hat.shape
    assert idx.shape == (B, N, Q), idx.shape
    assert idx.dtype == torch.long
    assert vq_loss.dim() == 0
    # STE: z_hat 对 z 应有梯度(直通)
    (z_hat.sum() + vq_loss).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    gnorm = z.grad.norm().item()
    print(f"[shapes/grad] z_hat={tuple(z_hat.shape)} idx={tuple(idx.shape)} "
          f"vq_loss={vq_loss.item():.4f} z.grad_norm={gnorm:.3f}  OK")
    # bits 统计
    bits = rvq.bits_per_anchor()
    print(f"[bits] {Q} 层 x log2({Ncb}) = {bits:.1f} bit/anchor "
          f"(连续 feature 是 {D}x32={D*32} bit, 压缩比 ~{D*32/bits:.1f}x)  OK")

def test_progressive():
    B, N, D, Q = 1, 128, 32, 8
    rvq = ResidualVQ(dim=D, num_quantizers=Q, codebook_size=256)
    rvq.train()
    z = torch.randn(B, N, D)
    for na in [1, 4, 8]:
        z_hat, idx, _ = rvq(z, num_active=na)
        assert idx.shape[-1] == na, (idx.shape, na)
    print(f"[progressive] num_active in 1/4/8 各自 idx 末维匹配  OK")

def test_coarse_to_fine():
    """验证由粗到细: 第1层重建能量应远大于后层(残差量纲递减)。"""
    B, N, D, Q = 1, 1000, 32, 6
    rvq = ResidualVQ(dim=D, num_quantizers=Q, codebook_size=1024)
    rvq.train()
    z = torch.randn(B, N, D)
    # 多跑几步让 EMA 码本收敛
    for _ in range(30):
        rvq(z)
    rvq.eval()
    flat = z.reshape(-1, D)
    residual = flat.clone()
    layer_energy = []
    for i in range(Q):
        q, _, _ = rvq.layers[i](residual)
        e = q.pow(2).sum(-1).mean().item()
        layer_energy.append(e)
        residual = residual - q
    print(f"[coarse->fine] 各层重建能量: " +
          " ".join(f"L{i}={e:.3f}" for i, e in enumerate(layer_energy)))
    # 第1层能量应为最大(主成分)
    assert layer_energy[0] == max(layer_energy), "第1层应抓最大能量"
    print(f"[coarse->fine] 第1层能量最大(抓主成分)  OK")

def test_recon_improves_with_layers():
    """层数越多, 重建误差应单调下降。"""
    B, N, D, Q = 1, 500, 32, 8
    rvq = ResidualVQ(dim=D, num_quantizers=Q, codebook_size=1024)
    rvq.train()
    z = torch.randn(B, N, D)
    for _ in range(50):
        rvq(z)
    rvq.eval()
    errs = []
    for na in range(1, Q + 1):
        z_hat, _, _ = rvq(z, num_active=na)
        errs.append((z - z_hat).pow(2).mean().item())
    print(f"[recon vs Q] " + " ".join(f"Q{na}={e:.4f}" for na, e in zip(range(1, Q+1), errs)))
    # 允许极小波动, 总体趋势必须下降: 末层误差 < 首层误差
    assert errs[-1] < errs[0], (errs[0], errs[-1])
    print(f"[recon vs Q] 重建误差随层数下降 ({errs[0]:.4f} -> {errs[-1]:.4f})  OK")

if __name__ == "__main__":
    test_shapes_and_grad()
    test_progressive()
    test_coarse_to_fine()
    test_recon_improves_with_layers()
    print("\n=== ALL RVQ SMOKE TESTS PASSED ===")
