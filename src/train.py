"""微调训练入口：open_clip 基座 + HardNegClipLoss（含蒸馏）+ fp16 AMP（T4 适配）。

本版本修复项（对应审核 B 组）：
- 硬负例 mask：无负例样本不再用空串污染分母，支持每条样本 K>1 个负例
- 训练闭环：warmup + cosine、梯度裁剪、logit_scale 上限、断点续训（含优化器/AMP 状态）
- 每个 epoch 在 val split 上算 t2i R@1，同时记录指标到 metrics.jsonl
- E3 蒸馏：按样本下标从教师缓存取特征，算 B x B 教师分布（不再有 TODO 空实现）

用法：
    python -m src.train --config configs/finetune_e2.yaml
    python -m src.train --config configs/finetune_e3.yaml --resume runs/e3/epoch1.pt

支持三种策略（config.strategy）：full / lock_image / lora
注意：lora 路径的 target_modules 仍待修复（open_clip 无 q_proj/v_proj），见 dev-doc 风险项。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import Flickr30kHardNeg, collate
from src.loss import HardNegClipLoss
from src.optim import build_scheduler
from src.teacher import check_alignment, load_teacher_cache, teacher_similarity


def apply_strategy(model, strategy: str, lora_r: int = 8):
    if strategy == "lock_image":
        for p in model.visual.parameters():
            p.requires_grad = False
    elif strategy == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, cfg)
    return model


def load_val_groups(split_path: Path, limit: int | None = None):
    """val split 按图分组，用于 t2i R@1 监控。"""
    groups: dict[str, dict] = {}
    for line in split_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("image_id") or row["image"])
        groups.setdefault(key, {"image": row["image"], "captions": []})["captions"].append(row["caption"])
    items = list(groups.values())
    return items[:limit] if limit else items


@torch.no_grad()
def evaluate_val(model, preprocess, tokenizer, groups, image_root: Path, device: str,
                 batch_size: int = 64) -> float:
    """返回 t2i R@1（每条 caption 检索全部 val 图）。"""
    model.eval()
    img_feats, txt_feats, cap_to_img = [], [], []
    for start in range(0, len(groups), batch_size):
        chunk = groups[start:start + batch_size]
        images = torch.stack([
            preprocess(Image.open(image_root / g["image"]).convert("RGB")) for g in chunk
        ]).to(device)
        fi = model.encode_image(images)
        img_feats.append(fi / fi.norm(dim=-1, keepdim=True))
        for offset, g in enumerate(chunk):
            cap_to_img.extend([start + offset] * len(g["captions"]))
        flat = [c for g in chunk for c in g["captions"]]
        ft = model.encode_text(tokenizer(flat).to(device))
        txt_feats.append(ft / ft.norm(dim=-1, keepdim=True))

    image_matrix = torch.cat(img_feats)
    text_matrix = torch.cat(txt_feats)
    scores = text_matrix @ image_matrix.t()                       # (N_cap, N_img)
    gt = torch.tensor(cap_to_img, device=device)
    gt_scores = scores[torch.arange(len(gt), device=device), gt]
    ranks = (scores > gt_scores[:, None]).sum(dim=1)
    model.train()
    return float((ranks == 0).float().mean())


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, global_step: int,
                    cfg: dict, val_r1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "val_t2i_r1": val_r1,
        "config": cfg,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", help="从 checkpoint 续训（也可写在 config 的 resume 字段）")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"],
        force_custom_text=cfg.get("strategy") == "lora",
    )
    model = apply_strategy(model, cfg.get("strategy", "full"), cfg.get("lora_r", 8)).to(device)
    if cfg.get("grad_checkpointing") and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    tokenizer = get_tokenizer(cfg["model"])

    train_data = Path(cfg["train_data"])
    image_root = train_data.parent.parent / "images"
    dataset = Flickr30kHardNeg(str(train_data), cfg.get("hardneg_data"), image_root, preprocess)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4),
        collate_fn=lambda b: collate(b, tokenizer), drop_last=True,
    )
    print(f"训练样本 {len(dataset)} 条，其中 {dataset.num_with_negatives()} 条带硬负例")

    # E3 教师缓存：按样本下标取用，长度必须与训练集一致
    teacher = None
    if cfg.get("distill_lambda", 0) > 0:
        teacher = load_teacher_cache(Path(cfg["distill_teacher_cache"]))
        check_alignment(teacher, len(dataset))
        print(f"教师缓存已加载：{teacher['image'].shape[0]} 条，lambda={cfg['distill_lambda']}")

    criterion = HardNegClipLoss(
        cfg.get("hard_negative_weight", 1.0),
        cfg.get("distill_lambda", 0.0),
        cfg.get("kd_temperature", 1.0),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["wd"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["precision"] == "amp"))

    accum = max(1, cfg.get("accum_freq", 1))
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg.get("warmup", 0), total_steps,
                                cfg.get("lr_scheduler", "cosine"))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    start_epoch, global_step = 0, 0
    resume_path = args.resume or cfg.get("resume")
    if resume_path:
        state = torch.load(Path(resume_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("global_step", 0)
        if scheduler is not None:
            scheduler.last_epoch = global_step
        print(f"已从 {resume_path} 恢复：epoch {start_epoch}，step {global_step}")

    val_groups = None
    if cfg.get("val_data"):
        val_groups = load_val_groups(Path(cfg["val_data"]), cfg.get("val_limit"))
        print(f"验证集 {len(val_groups)} 张图")

    metrics_path = out_dir / "metrics.jsonl"
    logit_max = math.log(cfg.get("logit_scale_clamp", 100.0))
    use_amp = cfg["precision"] == "amp" and device == "cuda"

    def optimizer_step():
        nonlocal global_step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, cfg.get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "logit_scale"):
            with torch.no_grad():
                model.logit_scale.clamp_(max=logit_max)
        if scheduler is not None:
            scheduler.step()
        global_step += 1

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch_idx, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device)
            texts = batch["text"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                fi = model.encode_image(images)
                ft = model.encode_text(texts)
                fi = fi / fi.norm(dim=-1, keepdim=True)
                ft = ft / ft.norm(dim=-1, keepdim=True)

                hard_neg, neg_mask = None, None
                if batch["negative"] is not None:
                    bsz, k, length = batch["negative"].shape
                    fn = model.encode_text(batch["negative"].view(bsz * k, length).to(device))
                    fn = fn / fn.norm(dim=-1, keepdim=True)
                    hard_neg = fn.view(bsz, k, -1)
                    neg_mask = batch["neg_mask"].to(device)

                teacher_sim = None
                if teacher is not None:
                    teacher_sim = teacher_similarity(teacher, batch["index"], device)

                if hasattr(model, "logit_scale"):
                    logit_scale = model.logit_scale.exp()
                else:  # SigLIP 等无 logit_scale 的模型走固定温度
                    logit_scale = torch.tensor(1.0 / 0.07, device=device)

                outputs = criterion(fi, ft, logit_scale, hard_neg, neg_mask, teacher_sim)
                loss = outputs["loss"] / accum

            scaler.scale(loss).backward()

            if batch_idx % accum == 0:
                optimizer_step()

            pbar.set_postfix(loss=f"{outputs['loss'].item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # 梯度累积的尾部：本 epoch 剩余的不足 accum 的梯度也要更新一次
        if len(loader) % accum != 0:
            optimizer_step()

        record = {"epoch": epoch, "global_step": global_step, "loss": outputs["loss"].item()}
        if val_groups:
            record["val_t2i_r1"] = evaluate_val(
                model, preprocess, tokenizer, val_groups, image_root, device
            )
            print(f"epoch {epoch} val t2i R@1 = {record['val_t2i_r1']:.4f}")

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        ckpt = out_dir / f"epoch{epoch}.pt"
        save_checkpoint(ckpt, model, optimizer, scaler, epoch, global_step, cfg,
                        record.get("val_t2i_r1", float("nan")))
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
