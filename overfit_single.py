"""
overfit_single.py — 高频容量判决探针 (single-mesh overfit)

目的
----
在【动架构 / 上多分辨率 grid 之前】，用一个零风险实验判决高频瓶颈到底是不是
"通用共享 MLP 的表示容量不足"：
    取 1 个 mesh，固定 base，大幅放大 per-face latent / 解码器容量，只在这一个
    mesh 上过拟合（不要求泛化）。
      · 若高频出得来  -> 瓶颈确认是"容量/DOF 不足"，多分辨率 grid 方向正确，放心投入。
      · 若过拟合都出不来 -> 瓶颈在更底层（位移参数化 / 缝合平均 / PE 频带），先查这些。

用法
----
    python overfit_single.py \
        --scan_obj  /path/to/gt_scan.obj \
        --base_obj  /path/to/base_mesh.obj \
        --config    configs/config_ptsa.yaml \
        --preset    baseline \
        --steps     3000 \
        --output_dir overfit_out

容量档位 (--preset)：
    baseline  : 完全用 config 原值（对照组）
    high      : feature_dim/pt_sa_dim/dec_hidden 翻倍左右
    extreme   : 极大容量（feature_dim=2048 等），逼近"表示上限"
也可用 --feature_dim / --pt_sa_dim / --dec_hidden / --subdivision_levels 单独覆盖，
覆盖优先级 > preset。

判决建议：跑 baseline 和 extreme 两组，对比 output_dir 里周期导出的 fine_*.obj
和 render_*.png。若 extreme 明显比 baseline 出更多高频细节 -> 容量是瓶颈。
另可单独扫 --subdivision_levels 8 -> 12/16，看 PE 频带是否是廉价的限制因子。

注意：本脚本不依赖 dataset/LMDB，直接从 scan obj 采样点云；与训练完全解耦，
不会影响任何已有 checkpoint 或配置。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.pipeline import Stage2Pipeline
from utils.render import DifferentiableNormalRenderer
from utils.train_helpers import (
    load_config,
    compute_uniform_laplacian_l1,
    compute_normal_loss_geo,
)

try:
    from pytorch_msssim import ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


# ----------------------------------------------------------------------------
# 容量档位
# ----------------------------------------------------------------------------
def apply_capacity_preset(model_cfg: dict, preset: str):
    """按档位放大模型容量（只改 model 子配置，原地修改）。"""
    if preset == "baseline":
        return
    if preset == "high":
        model_cfg["feature_dim"] = 1024
        model_cfg["pt_sa_dim"] = 256
        model_cfg["dec_hidden_dim"] = [512, 256, 128]
        # PE 频带也适度提高
        model_cfg["subdivision_levels"] = max(model_cfg.get("subdivision_levels", 8), 10)
    elif preset == "extreme":
        model_cfg["feature_dim"] = 2048
        model_cfg["pt_sa_dim"] = 512
        model_cfg["dec_hidden_dim"] = [1024, 512, 256]
        model_cfg["subdivision_levels"] = max(model_cfg.get("subdivision_levels", 8), 12)
    else:
        raise ValueError(f"未知 preset: {preset}")


def apply_overrides(model_cfg: dict, args):
    """命令行单参数覆盖（优先级高于 preset）。"""
    if args.feature_dim is not None:
        model_cfg["feature_dim"] = args.feature_dim
    if args.pt_sa_dim is not None:
        model_cfg["pt_sa_dim"] = args.pt_sa_dim
    if args.dec_hidden is not None:
        # 形如 "512,256,128"
        model_cfg["dec_hidden_dim"] = [int(x) for x in args.dec_hidden.split(",")]
    if args.subdivision_levels is not None:
        model_cfg["subdivision_levels"] = args.subdivision_levels


# ----------------------------------------------------------------------------
# 几何 IO / 预处理（与 infer_stage2.py 对齐）
# ----------------------------------------------------------------------------
def normalize_to_unit(verts_np):
    """中心化 + 缩放到单位球（与 infer_stage2.stage2_normalize 一致）。"""
    bbox_min = verts_np.min(axis=0)
    bbox_max = verts_np.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    v = verts_np - center
    scale = float(np.max(np.linalg.norm(v, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    return v / scale, center, scale


def load_inputs(scan_obj, base_obj, point_num, use_scan_normal, device):
    """加载 scan(GT) 与 base mesh，统一归一化，采样 scan 点云+法线。"""
    gt_mesh = trimesh.load(scan_obj, process=False)
    gt_v = np.asarray(gt_mesh.vertices, dtype=np.float64)
    gt_f = np.asarray(gt_mesh.faces, dtype=np.int64)

    gt_v_norm, center, scale = normalize_to_unit(gt_v)
    gt_mesh_norm = trimesh.Trimesh(vertices=gt_v_norm, faces=gt_f, process=False)

    # base mesh 用 GT 的 center/scale 归一化，保证两者在同一坐标系
    base_mesh = trimesh.load(base_obj, process=False)
    base_v_raw = np.asarray(base_mesh.vertices, dtype=np.float64)
    base_f = np.asarray(base_mesh.faces, dtype=np.int64)
    base_v = (base_v_raw - center) / scale

    # base 顶点法线（面积加权）
    base_tm = trimesh.Trimesh(vertices=base_v, faces=base_f, process=False)
    base_n = np.asarray(base_tm.vertex_normals, dtype=np.float32)

    # 从 GT 表面采样 scan 点 + 对应面法线
    scan_points, face_ids = trimesh.sample.sample_surface(gt_mesh_norm, point_num)
    scan_points = np.asarray(scan_points, dtype=np.float32)
    scan_normals = None
    if use_scan_normal:
        face_normals = gt_mesh_norm.face_normals
        scan_normals = face_normals[face_ids].astype(np.float32)

    out = {
        "gt_v": torch.from_numpy(gt_v_norm.astype(np.float32)).unsqueeze(0).to(device),
        "gt_f": torch.from_numpy(gt_f).long().unsqueeze(0).to(device),
        "base_v": torch.from_numpy(base_v.astype(np.float32)).unsqueeze(0).contiguous().to(device),
        "base_f": torch.from_numpy(base_f).long().unsqueeze(0).contiguous().to(device),
        "base_n": torch.from_numpy(base_n).unsqueeze(0).contiguous().to(device),
        "scan_p": torch.from_numpy(scan_points).unsqueeze(0).contiguous().to(device),
        "scan_n": (torch.from_numpy(scan_normals).unsqueeze(0).contiguous().to(device)
                   if scan_normals is not None else None),
    }
    return out


def export_obj(path, verts, faces):
    v = verts.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(path)


def save_normal_png(path, img_thwc):
    """img: (H,W,3) in [-1,1] -> png"""
    from PIL import Image
    a = img_thwc.detach().cpu().numpy()
    a = ((a + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


# ----------------------------------------------------------------------------
# 渲染 loss（复刻 train_stage2.py 的 normal_loss_type == 'geo' 分支，view-chunk）
# ----------------------------------------------------------------------------
def render_loss(renderer, f_verts, f_faces, gt_v, gt_f, cfg_loss, num_chunks, device):
    """返回 (total_render_loss_tensor, 标量明细 dict, 第一块的 pred/gt 图用于导出)。"""
    w_depth = cfg_loss.get("w_depth_l1", 4.0)
    w_mask = cfg_loss.get("w_mask", 1.0)
    w_geo = cfg_loss.get("w_normal_geo", 4.0)
    normal_loss_type = cfg_loss.get("normal_loss_type", "geo")
    w_nl1 = cfg_loss.get("w_normal_l1", 4.0)
    w_ssim = cfg_loss.get("w_normal_ssim", 0.5)

    f_faces_exp = f_faces.unsqueeze(0)
    total = torch.zeros((), device=device)
    detail = {"depth": 0.0, "mask": 0.0, "normal": 0.0, "ssim": 0.0}
    first_imgs = None

    for vc in range(num_chunks):
        pred_img, gt_img, pred_depth, gt_depth = renderer(f_verts, f_faces_exp, gt_v, gt_f)

        loss_depth = torch.zeros((), device=device)
        loss_mask = torch.zeros((), device=device)
        if pred_depth is not None and gt_depth is not None:
            dmask = (pred_depth.abs() > 1e-6) & (gt_depth.abs() > 1e-6)
            if dmask.any():
                loss_depth = F.l1_loss(pred_depth[dmask], gt_depth[dmask])
            pred_occ = (pred_depth.abs() > 1e-6).float()
            gt_occ = (gt_depth.abs() > 1e-6).float()
            loss_mask = F.l1_loss(pred_occ, gt_occ)

        if normal_loss_type == "geo":
            loss_normal = torch.zeros((), device=device)
            if pred_img is not None and gt_img is not None:
                pred_fg = pred_img.abs().sum(-1, keepdim=True) > 1e-6
                gt_fg = gt_img.abs().sum(-1, keepdim=True) > 1e-6
                mask_and = pred_fg & gt_fg
                if mask_and.any():
                    loss_normal = compute_normal_loss_geo(pred_img, gt_img, mask_and)
            chunk = w_depth * loss_depth + w_geo * loss_normal + w_mask * loss_mask
            detail["normal"] += float(loss_normal) / num_chunks
        else:
            # l1 + ssim 分支（兜底）
            mask_or = (pred_img.abs().sum(-1, keepdim=True) > 1e-6) | \
                      (gt_img.abs().sum(-1, keepdim=True) > 1e-6)
            loss_normal = F.l1_loss(pred_img * mask_or, gt_img * mask_or)
            loss_ssim = torch.zeros((), device=device)
            if w_ssim > 0 and HAS_SSIM:
                p = ((pred_img + 1) * 0.5).permute(0, 3, 1, 2)
                g = ((gt_img + 1) * 0.5).permute(0, 3, 1, 2)
                loss_ssim = 1.0 - ssim(p, g, data_range=1.0)
            chunk = w_depth * loss_depth + w_nl1 * loss_normal + w_ssim * loss_ssim + w_mask * loss_mask
            detail["normal"] += float(loss_normal) / num_chunks
            detail["ssim"] += float(loss_ssim) / num_chunks

        detail["depth"] += float(loss_depth) / num_chunks
        detail["mask"] += float(loss_mask) / num_chunks
        total = total + chunk / num_chunks

        if first_imgs is None:
            first_imgs = (pred_img[0].detach(), gt_img[0].detach())

    return total, detail, first_imgs


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Single-mesh overfit 高频容量探针")
    p.add_argument("--scan_obj", required=True, help="GT/scan 高分辨率 mesh (.obj)")
    p.add_argument("--base_obj", required=True, help="base mesh (.obj)")
    p.add_argument("--config", default="configs/config_ptsa.yaml")
    p.add_argument("--preset", default="baseline", choices=["baseline", "high", "extreme"])
    # 单参数覆盖（优先级高于 preset）
    p.add_argument("--feature_dim", type=int, default=None)
    p.add_argument("--pt_sa_dim", type=int, default=None)
    p.add_argument("--dec_hidden", type=str, default=None, help='如 "512,256,128"')
    p.add_argument("--subdivision_levels", type=int, default=None, help="PE 频带数")
    # 训练
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3, help="overfit 用比训练略大的 lr")
    p.add_argument("--w_laplacian", type=float, default=0.0,
                   help="默认关闭，纯测容量上限；设 >0 可加回平滑正则")
    p.add_argument("--w_seam", type=float, default=0.0,
                   help="seam consistency 权重；>0 验证能否消除共享边 sliver")
    p.add_argument("--resample_every", type=int, default=0,
                   help=">0 则每隔若干 step 重采样 scan 点（默认 0=固定点云，更易过拟合）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # 导出
    p.add_argument("--output_dir", default="overfit_out")
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--views", type=int, default=None, help="覆盖渲染视角数")
    p.add_argument("--image_size", type=int, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- config ---
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    apply_capacity_preset(model_cfg, args.preset)
    apply_overrides(model_cfg, args)
    if args.views is not None:
        cfg["render"]["views_per_sample"] = args.views
    if args.image_size is not None:
        cfg["render"]["image_size"] = args.image_size
    cfg["loss"]["w_laplacian"] = args.w_laplacian

    tag = f"{args.preset}_fd{model_cfg['feature_dim']}_lv{model_cfg['subdivision_levels']}"
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] preset={args.preset} feature_dim={model_cfg['feature_dim']} "
          f"pt_sa_dim={model_cfg.get('pt_sa_dim')} dec_hidden={model_cfg.get('dec_hidden_dim')} "
          f"subdivision_levels={model_cfg['subdivision_levels']} rate={model_cfg.get('subdivision_rate')}")
    print(f"[out] {out_dir}")

    # --- data ---
    use_scan_normal = model_cfg.get("use_scan_normal", False)
    point_num = cfg["data"]["point_num"]
    data = load_inputs(args.scan_obj, args.base_obj, point_num, use_scan_normal, device)
    print(f"[data] base: V={data['base_v'].shape[1]} F={data['base_f'].shape[1]} | "
          f"gt: V={data['gt_v'].shape[1]} F={data['gt_f'].shape[1]} | scan pts={point_num}")

    # --- model / renderer / optim ---
    model = Stage2Pipeline(config=model_cfg).to(device)
    model.train()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_param/1e6:.2f}M")

    total_views = cfg["render"]["views_per_sample"]
    view_chunk = cfg["render"].get("view_chunk_size", total_views)
    num_chunks = max(1, total_views // view_chunk)
    renderer = DifferentiableNormalRenderer(
        image_size=cfg["render"]["image_size"],
        device=device,
        cameras_per_batch=view_chunk,
        dist_multiplier=cfg["render"].get("dist_multiplier", 1.0),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    # --- overfit loop ---
    for step in range(args.steps):
        if args.resample_every > 0 and step > 0 and step % args.resample_every == 0:
            data = load_inputs(args.scan_obj, args.base_obj, point_num, use_scan_normal, device)

        optimizer.zero_grad(set_to_none=True)

        f_verts, f_faces, disp, trans_feat, vfeat, kl = model(
            data["base_v"], data["base_f"], data["base_n"],
            data["scan_p"], scan_normals=data["scan_n"],
        )

        # 索引合法性
        if f_faces.max() >= f_verts.shape[1]:
            print(f"[warn] step {step}: invalid fine faces, skip")
            continue

        r_loss, detail, first_imgs = render_loss(
            renderer, f_verts, f_faces, data["gt_v"], data["gt_f"],
            cfg["loss"], num_chunks, device,
        )

        loss = r_loss
        if args.w_laplacian > 0:
            loss = loss + args.w_laplacian * compute_uniform_laplacian_l1(f_verts[0], f_faces)

        # seam consistency (FaceTriangleDecoder only)
        seam_val = 0.0
        if args.w_seam > 0:
            seam = getattr(model.decoder, "_last_seam_loss", None)
            if seam is not None:
                loss = loss + args.w_seam * seam
                seam_val = float(seam)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 20 == 0 or step == args.steps - 1:
            print(f"[step {step:5d}] loss={float(loss):.5f} "
                  f"normal={detail['normal']:.5f} depth={detail['depth']:.5f} "
                  f"mask={detail['mask']:.5f} seam={seam_val:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if step % args.save_interval == 0 or step == args.steps - 1:
            export_obj(os.path.join(out_dir, f"fine_{step:05d}.obj"), f_verts[0], f_faces)
            if first_imgs is not None:
                save_normal_png(os.path.join(out_dir, f"pred_{step:05d}.png"), first_imgs[0])
                if step == 0:
                    save_normal_png(os.path.join(out_dir, "gt_view.png"), first_imgs[1])
                    export_obj(os.path.join(out_dir, "base.obj"), data["base_v"][0], data["base_f"][0])

    print(f"[done] 结果在 {out_dir}  对比 fine_*.obj 与 pred_*.png 看高频是否随容量提升而出现")


if __name__ == "__main__":
    main()
