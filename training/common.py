"""学習・オフライン評価で共通の処理 (評価指標, 可視化)."""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from omnivla_nav.data_utils import MODALITY_NAMES, denormalize_actions  # noqa: E402
from omnivla_nav.omnivla_model import forward_actions  # noqa: E402
from omnivla_nav.trajectory_io import image_path  # noqa: E402
from omnivla_nav.viz import render_debug  # noqa: E402


def trajectory_errors(pred: torch.Tensor, gt: torch.Tensor, metric_spacing: float) -> Dict[str, torch.Tensor]:
    """pred, gt: (B, 8, 4) 正規化値. 戻り値は per-sample のテンソル (メートル/ラジアン)."""
    d = torch.linalg.norm(pred[..., :2] - gt[..., :2], dim=-1) * metric_spacing  # (B, 8)
    yaw_p = torch.atan2(pred[..., 3], pred[..., 2])
    yaw_g = torch.atan2(gt[..., 3], gt[..., 2])
    dyaw = torch.atan2(torch.sin(yaw_p - yaw_g), torch.cos(yaw_p - yaw_g)).abs()
    return {"ade": d.mean(dim=1), "fde": d[:, -1], "yaw_err": dyaw[:, -1],
            "mse": ((pred - gt) ** 2).mean(dim=(1, 2))}


@torch.no_grad()
def evaluate(vla, head_fn: Callable, pose_projector, loader, num_patches: int, device: torch.device,
             metric_spacing: float, max_batches: Optional[int] = None, viz_dir: Optional[str] = None,
             num_viz: int = 0) -> Dict[str, float]:
    sums: Dict[str, list] = defaultdict(list)
    per_turn: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    per_mod: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n_viz = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        pred = forward_actions(vla, head_fn, pose_projector, batch, num_patches, device).float()
        gt = batch["actions"].to(device)
        errs = trajectory_errors(pred, gt, metric_spacing)
        mods = batch["modality_id"].long().tolist()
        turns = [bool(m.get("turn", False)) for m in batch.get("meta", [{}] * len(mods))]
        for k, v in errs.items():
            vals = v.cpu().numpy().tolist()
            sums[k].extend(vals)
            for m, val in zip(mods, vals):
                per_mod[m][k].append(val)
            for tflag, val in zip(turns, vals):
                per_turn["turn" if tflag else "straight"][k].append(val)
        if viz_dir is not None and n_viz < num_viz:
            os.makedirs(viz_dir, exist_ok=True)
            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()
            gp = batch["goal_pose"].numpy()
            for j, meta in enumerate(batch["meta"]):
                if n_viz >= num_viz:
                    break
                cur = Image.open(image_path(meta["traj_dir"], meta["t"])).convert("RGB")
                goal = Image.open(image_path(meta["traj_dir"], meta["goal_t"])).convert("RGB")
                p_m = denormalize_actions(pred_np[j], metric_spacing)
                g_m = denormalize_actions(gt_np[j], metric_spacing)
                goal_local = (float(gp[j, 0] * metric_spacing), float(gp[j, 1] * metric_spacing))
                lines = [f"{os.path.basename(meta['traj_dir'])}",
                         f"t={meta['t']} goal_t={meta['goal_t']}",
                         f"modality {mods[j]} ({MODALITY_NAMES.get(mods[j], '?')})",
                         f"ADE {errs['ade'][j].item():.3f} m  FDE {errs['fde'][j].item():.3f} m",
                         "green: ground truth / blue,orange: prediction"]
                render_debug(cur, goal, p_m, goal_local=goal_local, lines=lines, gt_waypoints=g_m).save(
                    os.path.join(viz_dir, f"sample_{n_viz:03d}.jpg"), quality=90)
                n_viz += 1
    out = {k: float(np.mean(v)) for k, v in sums.items() if v}
    out["num_samples"] = len(sums.get("ade", []))
    for name, d in per_turn.items():
        if d["ade"]:
            out[f"{name}/ade"] = float(np.mean(d["ade"]))
            out[f"{name}/fde"] = float(np.mean(d["fde"]))
            out[f"{name}/yaw_err_deg"] = float(np.degrees(np.mean(d["yaw_err"])))
            out[f"{name}/n"] = len(d["ade"])
    for m, d in sorted(per_mod.items()):
        name = MODALITY_NAMES.get(m, str(m))
        out[f"{name}/ade"] = float(np.mean(d["ade"]))
        out[f"{name}/fde"] = float(np.mean(d["fde"]))
        out[f"{name}/n"] = len(d["ade"])
    return out


def action_loss(pred: torch.Tensor, gt: torch.Tensor, smooth_weight: float = 0.1):
    """公式 train_omnivla.py の損失 (LeLaN 用の物体位置項を除く): MSE(行動) + w * MSE(隣接 waypoint 差)."""
    pred = pred.float()
    l_act = F.mse_loss(pred, gt)
    l_smooth = F.mse_loss(pred[:, :-1], pred[:, 1:])
    return l_act + smooth_weight * l_smooth, {"l2_action": l_act.detach(), "l2_smooth": l_smooth.detach()}
