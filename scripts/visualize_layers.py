from __future__ import annotations

import argparse
from pathlib import Path

from mssc.image_io import load_image
from mssc.visualize import plot_rg_layers


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize MSSC coarse-graining layers."
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
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = load_image(
        args.input,
        size=parse_size(args.size),
        mode=args.mode,
        value_range=args.value_range,
    )

    plot_rg_layers(
        image,
        block_size=args.block_size,
        n_steps=args.n_steps,
        path=args.out,
    )

    print(f"Saved RG layers: {args.out}")


if __name__ == "__main__":
    main()
