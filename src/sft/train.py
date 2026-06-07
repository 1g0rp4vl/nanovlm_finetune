"""Supervised fine-tuning of NanoVLM on expert MiniGrid trajectories."""

import argparse
import csv
import json
import random
import sys
from collections import Counter, deque
from itertools import islice
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import (
    evaluate_policy,
    generate,
    get_amp_dtype,
    reset_outputs,
    set_trainable,
    write_json,
)


class MiniGridDataset:
    """Expose JSONL rows in NanoVLM's VQA conversation format."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "images": [Image.open(row["image_path"]).convert("RGB")],
            "texts": [{"user": row["prompt"], "assistant": row["target"]}],
        }


def load_rows(path: Path):
    """Load and validate dataset metadata."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) < 2:
        raise ValueError("The dataset must contain at least two samples.")
    if len({row["prompt"] for row in rows}) != 1:
        raise ValueError("A training run must use one prompt.")
    return rows


def balance_left(rows, target_fraction: float, seed: int):
    """Downsample frequent actions until left reaches the requested fraction."""
    left = [row for row in rows if row["action_name"] == "left"]
    other = [row for row in rows if row["action_name"] != "left"]
    if not left or len(left) / len(rows) >= target_fraction:
        return rows

    keep_other = int(len(left) * (1 - target_fraction) / target_fraction)
    rng = random.Random(seed)
    groups = {
        action: [row for row in other if row["action_name"] == action]
        for action in ("right", "forward")
    }
    selected = []
    for group in groups.values():
        count = round(keep_other * len(group) / len(other))
        selected.extend(rng.sample(group, min(count, len(group))))

    balanced = left + selected[:keep_other]
    rng.shuffle(balanced)
    return balanced


def make_loader(rows, tokenizer, image_processor, model, args, shuffle):
    """Create a NanoVLM VQA DataLoader."""
    from data.collators import VQACollator
    from data.datasets import VQADataset

    dataset = VQADataset(
        MiniGridDataset(rows),
        tokenizer,
        image_processor,
        model.cfg.mp_image_token_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=VQACollator(tokenizer, None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def batch_loss(model, batch, device, dtype):
    """Compute NanoVLM's autoregressive answer loss for one batch."""
    non_blocking = device.type == "cuda"
    input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking)
    attention_mask = batch["attention_mask"].to(
        device,
        non_blocking=non_blocking,
    )
    images = torch.cat([image for group in batch["images"] for image in group]).to(
        device,
        non_blocking=non_blocking,
    )
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=dtype is not None,
    ):
        _, loss = model(
            input_ids,
            images,
            attention_mask=attention_mask,
            targets=labels,
        )
    return loss


@torch.no_grad()
def validation_loss(model, loader, device, dtype, batches: int):
    """Average validation loss over at most ``batches`` batches."""
    model.eval()
    losses = [
        batch_loss(model, batch, device, dtype).item()
        for batch in islice(loader, batches)
        if batch and batch["labels"].ne(-100).any()
    ]
    return sum(losses) / len(losses) if losses else 0.0


@torch.no_grad()
def action_metrics(
    model,
    tokenizer,
    image_processor,
    rows,
    device,
    args,
    sample_count: int,
):
    """Measure parsed-action accuracy and generated action distribution."""
    rows = rows[:sample_count]
    _, _, actions = generate(
        model,
        tokenizer,
        image_processor,
        [row["image_path"] for row in rows],
        [row["prompt"] for row in rows],
        device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.generation_batch_size,
        greedy=True,
        dtype=args.dtype,
    )
    counts = Counter(action or "invalid" for action in actions)
    total = max(len(rows), 1)
    return {
        "val_accuracy": sum(
            action == row["action_name"]
            for action, row in zip(actions, rows, strict=True)
        )
        / total,
        "val_parse_rate": sum(action is not None for action in actions) / total,
        **{
            f"val_{action}_frac": counts[action] / total
            for action in ("left", "right", "forward", "invalid")
        },
    }


def append_csv(path: Path, row) -> None:
    """Append one metrics row, writing the header for a new file."""
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    """Parse SFT training options."""
    parser = argparse.ArgumentParser(
        description="Fine-tune NanoVLM on expert MiniGrid trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Dataset metadata.jsonl."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--model", default="lusxvr/nanoVLM", help="NanoVLM model or checkpoint."
    )
    parser.add_argument(
        "--nanovlm-dir",
        type=Path,
        default=Path("external/nanoVLM"),
        help="Cloned nanoVLM repository.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader workers."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Maximum epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="AdamW learning rate.")
    parser.add_argument(
        "--trainable",
        choices=["mp", "decoder", "decoder_mp", "all"],
        default="mp",
        help="NanoVLM modules to fine-tune.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["auto", "bfloat16", "float16", "off"],
        default="auto",
        help="Mixed-precision dtype.",
    )
    parser.add_argument(
        "--min-left-fraction",
        type=float,
        default=0.0,
        help="Downsample frequent actions to reach this left fraction.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Validation fraction."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Train/validation split seed."
    )
    parser.add_argument(
        "--eval-every", type=int, default=10, help="Optimizer steps between evals."
    )
    parser.add_argument(
        "--eval-batches", type=int, default=4, help="Validation loss batches."
    )
    parser.add_argument(
        "--eval-samples", type=int, default=128, help="Generated validation answers."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=20, help="Rollout episodes per eval."
    )
    parser.add_argument(
        "--final-episodes", type=int, default=200, help="Final rollout episodes."
    )
    parser.add_argument(
        "--generation-batch-size", type=int, default=16, help="Generation batch size."
    )
    parser.add_argument(
        "--rollout-max-steps", type=int, default=40, help="Rollout step limit."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generated tokens; inferred as 1 or 48 when omitted.",
    )
    parser.add_argument(
        "--early-stop-accuracy", type=float, default=0.99, help="Accuracy threshold."
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=3, help="Consecutive passing evals."
    )
    parser.add_argument(
        "--save-every-epochs", type=int, default=1, help="Checkpoint interval."
    )
    parser.add_argument(
        "--loss-window", type=int, default=5, help="Moving train-loss window."
    )
    return parser.parse_args()


def main():
    """Run SFT, periodic evaluation, checkpointing, and final evaluation."""
    args = parse_args()
    sys.path.insert(0, str(args.nanovlm_dir.resolve()))
    from data.processors import get_image_processor, get_tokenizer
    from models.vision_language_model import VisionLanguageModel

    rows = load_rows(args.metadata)
    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    train_rows = balance_left(train_rows, args.min_left_fraction, args.seed)
    args.max_new_tokens = args.max_new_tokens or (
        48 if len(rows[0]["target"].split()) > 1 else 1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dtype = get_amp_dtype(args.amp_dtype, device)
    model = VisionLanguageModel.from_pretrained(args.model).to(device)
    set_trainable(model, args.trainable)
    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template,
    )
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        False,
    )
    train_loader = make_loader(
        train_rows,
        tokenizer,
        image_processor,
        model,
        args,
        True,
    )
    val_loader = make_loader(
        val_rows,
        tokenizer,
        image_processor,
        model,
        args,
        False,
    )

    reset_outputs(args.output_dir)
    metrics_path = args.output_dir / "metrics.csv"
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.dtype == torch.float16,
    )
    losses = deque(maxlen=args.loss_window)
    step = 0
    accuracy_streak = 0
    prompt = rows[0]["prompt"]

    def evaluate(epoch):
        model.eval()
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": sum(losses) / len(losses),
            "val_loss": validation_loss(
                model,
                val_loader,
                device,
                args.dtype,
                args.eval_batches,
            ),
            **action_metrics(
                model,
                tokenizer,
                image_processor,
                val_rows,
                device,
                args,
                args.eval_samples,
            ),
            **{
                f"rollout_{key}": value
                for key, value in evaluate_policy(
                    model,
                    tokenizer,
                    image_processor,
                    prompt,
                    device,
                    episodes=args.eval_episodes,
                    seed_start=10_000,
                    max_steps=args.rollout_max_steps,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.generation_batch_size,
                    dtype=args.dtype,
                ).items()
                if key != "action_distribution"
            },
        }
        append_csv(metrics_path, row)
        tqdm.write(
            f"eval step={step} accuracy={row['val_accuracy']:.3f} "
            f"success={row['rollout_success_rate']:.3f} "
            f"return={row['rollout_mean_return']:.3f}"
        )
        return row

    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        evaluated_step = None
        progress = tqdm(train_loader, desc=f"SFT {epoch}/{args.epochs}")
        for batch in progress:
            if not batch or not batch["labels"].ne(-100).any():
                continue
            loss = batch_loss(model, batch, device, args.dtype)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.item())
            progress.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

            if step % args.eval_every == 0:
                row = evaluate(epoch)
                evaluated_step = step
                accuracy_streak = (
                    accuracy_streak + 1
                    if row["val_accuracy"] >= args.early_stop_accuracy
                    else 0
                )
                stopped_early = accuracy_streak >= args.early_stop_patience
                model.train()
            if stopped_early:
                break

        if evaluated_step != step:
            row = evaluate(epoch)
            accuracy_streak = (
                accuracy_streak + 1
                if row["val_accuracy"] >= args.early_stop_accuracy
                else 0
            )
            stopped_early = accuracy_streak >= args.early_stop_patience
        if epoch % args.save_every_epochs == 0:
            model.save_pretrained(str(args.output_dir / f"epoch_{epoch}"))
        if stopped_early:
            break

    final_dir = args.output_dir / "final"
    model.save_pretrained(str(final_dir))
    final_actions = action_metrics(
        model,
        tokenizer,
        image_processor,
        val_rows,
        device,
        args,
        len(val_rows),
    )
    final_rollout = evaluate_policy(
        model,
        tokenizer,
        image_processor,
        prompt,
        device,
        episodes=args.final_episodes,
        seed_start=30_000,
        max_steps=args.rollout_max_steps,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.generation_batch_size,
        dtype=args.dtype,
    )
    summary = {
        "checkpoint": str(final_dir),
        "steps": step,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_action_counts": dict(Counter(row["action_name"] for row in train_rows)),
        **final_actions,
        **{f"eval_{key}": value for key, value in final_rollout.items()},
    }
    write_json(args.output_dir / "final.json", summary)
    write_json(final_dir / "summary.json", summary)
    print(f"Saved final checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
