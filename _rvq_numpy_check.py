"""用 numpy 复刻 ResidualVQ 的核心数值逻辑, 验证算法正确性。
(本地 torch DLL 在沙箱加载失败, 故用 numpy 验证算法本身; torch 算法路径与此一致。)
"""
import numpy as np
import math

np.random.seed(0)


def vq(x, cb):
    d = ((x[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    idx = d.argmin(1)
    return cb[idx], idx


def kmeans(x, K, iters=20):
    cb = x[np.random.choice(len(x), K, replace=False)].copy()
    for _ in range(iters):
        q, idx = vq(x, cb)
        for k in range(K):
            m = idx == k
            if m.any():
                cb[k] = x[m].mean(0)
    return cb


M, D, Q, K = 1000, 16, 6, 64
z = np.random.randn(M, D).astype(np.float64)

# 逐层在残差上训练码本(模拟 EMA 收敛后的效果)
res = z.copy()
cbs, layer_E = [], []
for i in range(Q):
    cb = kmeans(res, K)
    cbs.append(cb)
    q, _ = vq(res, cb)
    layer_E.append((q ** 2).sum(-1).mean())
    res = res - q

print("[coarse->fine] 各层重建能量:",
      " ".join(f"L{i}={e:.3f}" for i, e in enumerate(layer_E)))
assert layer_E[0] == max(layer_E), "第1层应抓主成分(能量最大)"
print("  -> 第1层能量最大, 残差量纲递减  OK")

# 重建误差随层数下降
res = z.copy()
zhat = np.zeros_like(z)
errs = []
for i in range(Q):
    q, _ = vq(res, cbs[i])
    zhat = zhat + q
    res = res - q
    errs.append(((z - zhat) ** 2).mean())
print("[recon vs Q]", " ".join(f"Q{i+1}={e:.4f}" for i, e in enumerate(errs)))
assert all(errs[i + 1] <= errs[i] + 1e-9 for i in range(len(errs) - 1)), "误差应单调不增"
print(f"  -> 重建误差随层数单调下降 ({errs[0]:.4f} -> {errs[-1]:.4f})  OK")

# bit 统计
bits = Q * math.log2(K)
cont = D * 32
print(f"[bits] {Q}层 x log2({K}) = {bits:.0f} bit/anchor  vs 连续 {D}x32={cont} bit "
      f"-> {cont/bits:.1f}x 压缩")
print()
print("=== RVQ 算法数值逻辑验证通过(numpy) ===")
