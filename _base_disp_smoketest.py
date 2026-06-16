"""Smoke test for the 2.6 lightweight base-vertex-displacement tier.

跑这个验证: BaseVertexDisplacementHead 接线正确 + Stage2Pipeline 在
use_base_displacement=true 下能前向/反向, 且 base 位移头能拿到梯度。

用法 (在有 torch / pytorch3d / torch_scatter 的机器上):
    python _base_disp_smoketest.py
预期: 全部打印 [OK], 最后 "ALL SMOKE TESTS PASSED"。

注意: 这是冒烟测试(只查接线/形状/梯度/有限性), 不验证训练效果。
"""
import os
import sys
import math

import torch

# 允许从仓库根目录直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.base_displacement import BaseVertexDisplacementHead
from models.pipeline import Stage2Pipeline
from utils.train_helpers import load_config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _icosahedron():
    """12 顶点 / 20 面的正二十面体, 顶点归一化到单位球。"""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = torch.tensor([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=torch.float32)
    verts = verts / verts.norm(dim=-1, keepdim=True)
    faces = torch.tensor([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=torch.long)
    normals = verts.clone()  # 单位球: 顶点法线 = 顶点位置
    return verts, faces, normals


def _random_scan(n=2000):
    """单位球面附近的随机点 + 径向法线。"""
    p = torch.randn(n, 3)
    p = p / p.norm(dim=-1, keepdim=True)
    p = p * (1.0 + 0.02 * torch.randn(n, 1))  # 轻微抖动
    nrm = p / p.norm(dim=-1, keepdim=True)
    return p, nrm


def test_head_isolated():
    print("[test] BaseVertexDisplacementHead (isolated) ...")
    torch.manual_seed(0)
    B, V, F, D = 1, 12, 20, 32
    verts, faces, normals = _icosahedron()
    verts = verts.unsqueeze(0).to(DEVICE)
    faces = faces.unsqueeze(0).to(DEVICE)
    normals = normals.unsqueeze(0).to(DEVICE)
    feats = torch.randn(B, F, D, device=DEVICE, requires_grad=True)

    head = BaseVertexDisplacementHead(feature_dim=D, hidden_dim=[64, 32],
                                      init_scale=1e-4, use_normal=True).to(DEVICE)

    disp = head(verts, faces, feats, normals)
    assert disp.shape == (B, V, 3), f"bad shape {disp.shape}"
    assert torch.isfinite(disp).all(), "disp not finite"
    # near-zero 初始化 => step0 位移应极小(恒等起步)
    assert disp.abs().max().item() < 1e-2, f"init disp too large: {disp.abs().max().item()}"
    print(f"  [OK] shape={tuple(disp.shape)}, max|disp|@init={disp.abs().max().item():.2e}")

    # 梯度可达 head 和 feats
    loss = (disp ** 2).sum()
    loss.backward()
    g = head.mlp[0].weight.grad
    assert g is not None and torch.isfinite(g).all(), "no/!finite grad on head"
    assert feats.grad is not None and torch.isfinite(feats.grad).all(), "no grad on feats"
    print("  [OK] gradients flow to head + face features")


def _run_pipeline_case(mode: str):
    print(f"[test] Stage2Pipeline forward/backward (use_base_displacement, mode={mode}) ...")
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "config_ptsa.yaml")
    config = load_config(cfg_path)
    mcfg = config["model"]
    # 缩小以加速冒烟
    mcfg["subdivision_rate"] = 4
    mcfg["use_base_displacement"] = True
    mcfg["base_disp_mode"] = mode
    mcfg["base_disp_knn_k"] = 16
    assert mcfg.get("encoding_mode") == "face", "smoke test 假设 encoding_mode=face"

    torch.manual_seed(0)
    model = Stage2Pipeline(config=mcfg).to(DEVICE)
    model.train()
    if getattr(model, "rvq", None) is not None:
        model._rvq_num_active = model.rvq.num_quantizers

    verts, faces, normals = _icosahedron()
    bv = verts.unsqueeze(0).to(DEVICE)
    bf = faces.unsqueeze(0).to(DEVICE)
    bn = normals.unsqueeze(0).to(DEVICE)
    scan, scan_n = _random_scan(2000)
    scan = scan.unsqueeze(0).to(DEVICE)
    scan_n = scan_n.unsqueeze(0).to(DEVICE)

    out = model(bv, bf, bn, scan, scan_normals=scan_n)
    f_verts, f_faces, disp, trans_feat, vertex_features, kl_loss = out

    assert torch.isfinite(f_verts).all(), "fine_verts not finite"
    assert f_faces.max().item() < f_verts.shape[1], "fine face index OOB"
    assert model._last_base_disp is not None, "base_disp not recorded"
    assert torch.isfinite(model._last_base_disp).all(), "base_disp not finite"
    print(f"  [OK] fine_verts={tuple(f_verts.shape)}, fine_faces={tuple(f_faces.shape)}, "
          f"max|base_disp|={model._last_base_disp.abs().max().item():.2e}")
    if getattr(model, "_last_rvq_recon_err", None) is not None:
        print(f"  [OK] rvq recon_err={float(model._last_rvq_recon_err):.4f}, "
              f"feat_norm={float(model._last_feat_norm):.4f}")

    # dummy 渲染状损失 + fine 位移惩罚 -> backward -> 确认 base 模块收到梯度
    loss = f_verts.pow(2).mean() + (disp ** 2).sum(dim=-1).mean()
    if model._last_vq_loss is not None:
        loss = loss + model._last_vq_loss
    loss.backward()

    base_mod = model.base_predictor if mode == "predict" else model.base_disp_head
    assert base_mod is not None, f"no base module for mode={mode}"
    head_grad = 0.0
    seen = False
    for p in base_mod.parameters():
        if p.grad is not None:
            head_grad += p.grad.norm().item()
            seen = True
    assert seen, "base module got no gradient"
    assert math.isfinite(head_grad), "base module grad not finite"
    assert head_grad > 0, "base module grad is exactly 0 (not wired e2e)"
    print(f"  [OK] base module total grad norm = {head_grad:.4e} (wired e2e)")


def test_pipeline_forward_backward():
    for m in ("predict", "reencode", "lightweight"):
        _run_pipeline_case(m)
        print()


def test_pipeline_disabled():
    print("[test] Stage2Pipeline with use_base_displacement=false (regression) ...")
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "config_ptsa.yaml")
    config = load_config(cfg_path)
    mcfg = config["model"]
    mcfg["subdivision_rate"] = 4
    mcfg["use_base_displacement"] = False

    torch.manual_seed(0)
    model = Stage2Pipeline(config=mcfg).to(DEVICE)
    model.train()
    if getattr(model, "rvq", None) is not None:
        model._rvq_num_active = model.rvq.num_quantizers

    verts, faces, normals = _icosahedron()
    bv, bf, bn = (verts.unsqueeze(0).to(DEVICE), faces.unsqueeze(0).to(DEVICE),
                  normals.unsqueeze(0).to(DEVICE))
    scan, scan_n = _random_scan(2000)
    scan, scan_n = scan.unsqueeze(0).to(DEVICE), scan_n.unsqueeze(0).to(DEVICE)

    out = model(bv, bf, bn, scan, scan_normals=scan_n)
    f_verts = out[0]
    assert torch.isfinite(f_verts).all()
    assert model.base_disp_head is None and model._last_base_disp is None
    print("  [OK] disabled path unaffected (base_disp_head is None)")


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    test_head_isolated()
    print()
    test_pipeline_forward_backward()
    print()
    test_pipeline_disabled()
    print("\nALL SMOKE TESTS PASSED")
