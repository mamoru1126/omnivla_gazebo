#!/usr/bin/env python3
"""LoRA アダプタをベースモデルにマージし、公式チェックポイントと同じ形式で保存する.

出力ディレクトリは公式 inference/run_omnivla.py の vla_path / resume_step としても、
本リポジトリの navigator の vla_path としても使える。

  python3 training/merge_lora.py --finetuned_dir /runs/<run>/checkpoints/step_005000 \
      --out_dir /checkpoints/omnivla-gazebo
CPU でマージする場合は 32GB 以上の RAM が必要 (--device cuda:0 なら VRAM 16GB 以上)。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_nav.omnivla_model import (FINETUNE_META, find_checkpoint_step, load_base_vla,  # noqa: E402
                                       read_finetune_meta)

COPY_EXTS = (".json", ".py", ".model", ".txt")
WEIGHT_PREFIXES = ("model-", "model.safetensors", "pytorch_model")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finetuned_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vla_path", default="", help="base model (default: finetune_meta.json の base_vla_path)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from peft import PeftModel

    meta = read_finetune_meta(args.finetuned_dir)
    base = args.vla_path or meta.get("base_vla_path")
    if not base:
        raise SystemExit("--vla_path is required (no finetune_meta.json)")
    step = find_checkpoint_step(args.finetuned_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"base={base}  adapter={args.finetuned_dir}  step={step}")
    vla, processor = load_base_vla(base, torch.device(args.device))
    vla = PeftModel.from_pretrained(vla, os.path.join(args.finetuned_dir, "lora_adapter"))
    vla = vla.merge_and_unload()
    # tokenizer / processor / config / modeling コードなど重み以外のファイルをベースからコピー
    for f in os.listdir(base):
        src = os.path.join(base, f)
        if os.path.isfile(src) and f.endswith(COPY_EXTS) and not f.startswith(WEIGHT_PREFIXES) \
                and "--" not in f and f != "model.safetensors.index.json":
            shutil.copy2(src, os.path.join(args.out_dir, f))
    vla.save_pretrained(args.out_dir, safe_serialization=True, max_shard_size="5GB")
    for module in ("action_head", "pose_projector"):
        shutil.copy2(os.path.join(args.finetuned_dir, f"{module}--{step}_checkpoint.pt"),
                     os.path.join(args.out_dir, f"{module}--{step}_checkpoint.pt"))
    meta = dict(meta)
    meta.update({"merged_from": os.path.abspath(args.finetuned_dir)})
    with open(os.path.join(args.out_dir, FINETUNE_META), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"merged model saved to {args.out_dir} (resume_step={step})")


if __name__ == "__main__":
    main()
