from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mssc.image_io import load_image
from mssc.complexity import complexity_profile


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MSSC complexity profile for an image."
    )

    parser.add_argument("input", type=Path)

    parser.add_argument(
        "--size",
        default="auto",
        help=(
            "'auto': resize to nearest power-of-two square. Default. "
            "'none': no resize; require a square power-of-two image. "
            "INT: resize to INT x INT."
        ),
    )
    parser.add_argument("--mode", choices=["rgb", "grayscale"], default="rgb")
    parser.add_argument(
        "--value-range",
        choices=["0_1", "minus1_1"],
        default="minus1_1",
    )

    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=None)

    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-plot", type=Path, default=None)

    return parser.parse_args()


def save_profile_csv(profile: np.ndarray, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "C_k"])

        for k, value in enumerate(profile):
            writer.writerow([k, value])


def save_profile_plot(profile: np.ndarray, path: Path) -> None:
    k = np.arange(len(profile))

    fig, ax = plt.subplots()
    ax.plot(k, profile, marker="o")

    ax.set_xlabel("Scale index k")
    ax.set_ylabel("Partial complexity C_k")
    ax.set_title("MSSC profile")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    image = load_image(
        args.input,
        size=parse_size(args.size),
        mode=args.mode,
        value_range=args.value_range,
    )

    profile = complexity_profile(
        image,
        block_size=args.block_size,
        n_steps=args.n_steps,
    )

    print(f"Input: {args.input}")
    print(f"Image shape after loading/resizing: {image.shape}")
    print(f"Block size: {args.block_size}")
    print(f"Number of steps: {len(profile)}")
    print()

    print("Partial complexities:")
    for k, value in enumerate(profile):
        print(f"k={k:<3d} C_k={value:.12g}")

    print()
    print(f"Total complexity: {profile.sum():.12g}")
    print(f"Total complexity without C_0: {profile[1:].sum():.12g}")

    if args.out_csv is not None:
        save_profile_csv(profile, args.out_csv)
        print(f"Saved CSV: {args.out_csv}")

    if args.out_plot is not None:
        save_profile_plot(profile, args.out_plot)
        print(f"Saved plot: {args.out_plot}")


if __name__ == "__main__":
    main()
