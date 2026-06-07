"""Evaluate any NanoVLM checkpoint on fixed MiniGrid seeds."""

import argparse
import sys
from pathlib import Path

import torch

from src.config import PROMPTS
from src.utils import (
    evaluate_policy,
    get_amp_dtype,
    resolve_checkpoint,
    write_json,
)


def parse_args():
    """Parse checkpoint evaluation options."""
    parser = argparse.ArgumentParser(
        description="Evaluate a NanoVLM checkpoint on fixed MiniGrid seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON.")
    parser.add_argument(
        "--prompt", choices=PROMPTS, default="policy", help="Prompt format."
    )
    parser.add_argument(
        "--episodes", type=int, default=200, help="Evaluation episodes."
    )
    parser.add_argument("--seed-start", type=int, default=30_000, help="First seed.")
    parser.add_argument("--max-steps", type=int, default=40, help="Episode step limit.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Generation batch size."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generated tokens; inferred as 1 or 48 when omitted.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["auto", "bfloat16", "float16", "off"],
        default="auto",
        help="Mixed-precision dtype.",
    )
    parser.add_argument(
        "--nanovlm-dir",
        type=Path,
        default=Path("external/nanoVLM"),
        help="Cloned nanoVLM repository.",
    )
    return parser.parse_args()


def main():
    """Load a checkpoint, run greedy episodes, and save metrics."""
    args = parse_args()
    args.max_new_tokens = args.max_new_tokens or (
        48 if args.prompt == "plan_action" else 1
    )
    sys.path.insert(0, str(args.nanovlm_dir.resolve()))
    from data.processors import get_image_processor, get_tokenizer
    from models.vision_language_model import VisionLanguageModel

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionLanguageModel.from_pretrained(checkpoint).to(device)
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
    metrics = evaluate_policy(
        model,
        tokenizer,
        image_processor,
        PROMPTS[args.prompt],
        device,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        dtype=get_amp_dtype(args.amp_dtype, device),
    )
    write_json(
        args.output,
        {
            "checkpoint": checkpoint,
            "prompt": args.prompt,
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            **metrics,
        },
    )
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
