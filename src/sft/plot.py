"""Plot one or more SFT metrics CSV files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_run(spec: str):
    """Parse a ``LABEL=PATH`` command-line value."""
    label, separator, path = spec.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Run must use LABEL=PATH.")
    return label, Path(path)


def parse_args():
    """Parse SFT plotting options."""
    parser = argparse.ArgumentParser(
        description="Plot SFT learning curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="LABEL=metrics.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Figure directory."
    )
    return parser.parse_args()


def save_plot(runs, column, ylabel, output):
    """Plot one metric for all runs that contain it."""
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for label, frame in runs:
        if column in frame:
            axis.plot(
                frame["step"],
                frame[column],
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                label=label,
            )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    """Load metrics and create the core SFT comparison figures."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [(label, pd.read_csv(path)) for label, path in args.run]
    for column, label, filename in (
        ("val_accuracy", "Validation accuracy", "accuracy"),
        ("val_parse_rate", "Parse rate", "parse_rate"),
        ("rollout_success_rate", "Success rate", "success"),
        ("rollout_mean_return", "Mean return", "return"),
        ("val_loss", "Validation loss", "val_loss"),
    ):
        save_plot(runs, column, label, args.output_dir / filename)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
