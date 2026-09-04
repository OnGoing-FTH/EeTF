"""Command-line training entry point for EdgeDynamicViT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.datasets import split_edge_dataset
from engine.train import train_one_epoch
from engine.validate import validate
from losses.sparse_segmentation_loss import SparseEdgeLoss
from main import EdgeDynamicViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="edge_maps")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--keep-ratio", type=float, default=0.3)
    parser.add_argument("--ratio-weight", type=float, default=2.0)
    parser.add_argument("--router-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = split_edge_dataset(
        Path(args.data_root) / args.image_dir,
        Path(args.data_root) / args.mask_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    # B=1 is required because each sample can choose a different target resolution.
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, generator=loader_generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = EdgeDynamicViT(keep_ratio=args.keep_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    segmentation_loss = SparseEdgeLoss()
    start_epoch, best_f1 = 0, float("-inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and device.type == "cuda":
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_f1", best_f1))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, segmentation_loss, device,
            args.keep_ratio, scaler, args.ratio_weight, args.router_weight,
        )
        validation_stats = validate(
            model, val_loader, segmentation_loss, device,
            args.keep_ratio, args.ratio_weight, args.router_weight,
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "args": vars(args),
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if validation_stats["foreground_f1"] > best_f1:
            best_f1 = validation_stats["foreground_f1"]
            checkpoint["best_f1"] = best_f1
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={validation_stats['loss']:.4f} "
            f"val_f1={validation_stats['foreground_f1']:.4f} "
            f"val_iou={validation_stats['foreground_iou']:.4f}"
        )


if __name__ == "__main__":
    main()
