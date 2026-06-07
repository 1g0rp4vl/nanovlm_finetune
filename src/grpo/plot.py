"""Plot one or more GRPO metrics JSONL files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_run(spec: str):
    """Parse a ``LABEL=PATH`` command-line value."""
    label, separator, path = spec.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Run must use LABEL=PATH.")
    return label, Path(path)


def load_jsonl(path: Path):
    """Load JSONL metrics into a dataframe."""
    return pd.DataFrame(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def parse_args():
    """Parse GRPO plotting options."""
    parser = argparse.ArgumentParser(
        description="Plot GRPO learning curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="LABEL=metrics.jsonl.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Figure directory."
    )
    return parser.parse_args()


def save_plot(runs, columns, ylabel, output):
    """Plot available raw, moving-average, or evaluation metrics."""
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for label, frame in runs:
        for column, suffix in columns:
            if column not in frame:
                continue
            values = frame[["update", column]].dropna()
            if not values.empty:
                axis.plot(
                    values["update"],
                    values[column],
                    marker="o" if column.startswith("eval/") else None,
                    markersize=2.5,
                    linewidth=1.4,
                    label=f"{label}{suffix}",
                )
    axis.set_xlabel("GRPO update")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    """Load metrics and create the core GRPO comparison figures."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [(label, load_jsonl(path)) for label, path in args.run]
    save_plot(
        runs,
        [
            ("window/train/success_rate", " train"),
            ("eval/success_rate", " eval"),
        ],
        "Success rate",
        args.output_dir / "success",
    )
    save_plot(
        runs,
        [
            ("window/train/reward_mean", " train"),
            ("eval/mean_return", " eval"),
        ],
        "Reward / return",
        args.output_dir / "return",
    )
    save_plot(
        runs,
        [("window/train/kl", "")],
        "KL divergence",
        args.output_dir / "kl",
    )
    save_plot(
        runs,
        [("window/train/group_reward_std", "")],
        "Group reward std",
        args.output_dir / "group_reward_std",
    )
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
