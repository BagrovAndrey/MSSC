from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mssc.image_io import save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical toy images for MSSC experiments."
    )

    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Square image size. Default: 256",
    )
    parser.add_argument(
        "--value-range",
        choices=["0_1", "minus1_1"],
        default="minus1_1",
        help="Saved numeric range before image export. Default: minus1_1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for the noise image. Default: 0",
    )
    parser.add_argument(
        "--stripe-width",
        type=int,
        default=16,
        help="Stripe width for the stripe image. Default: 16",
    )
    parser.add_argument(
        "--checker-tile",
        type=int,
        default=16,
        help="Tile size for the checkerboard image. Default: 16",
    )

    return parser.parse_args()


def validate_size(size: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")
    if size % 2 != 0:
        raise ValueError("size must be even for 2x2 MSSC coarse-graining")


def validate_scale(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def to_output_range(image: np.ndarray, value_range: str) -> np.ndarray:
    if value_range == "0_1":
        return image
    if value_range == "minus1_1":
        return 2.0 * image - 1.0
    raise ValueError("value_range must be '0_1' or 'minus1_1'")


def uniform_image(size: int, level: float = 0.0) -> np.ndarray:
    return np.full((size, size), level, dtype=float)


def noise_image(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(size, size)).astype(float)


def horizontal_stripes(size: int, stripe_width: int) -> np.ndarray:
    validate_scale("stripe_width", stripe_width)
    y = np.arange(size)[:, None]
    stripes = ((y // stripe_width) % 2).astype(float)
    return np.broadcast_to(stripes, (size, size)).copy()


def checkerboard(size: int, tile_size: int) -> np.ndarray:
    validate_scale("checker_tile", tile_size)
    y = np.arange(size)[:, None]
    x = np.arange(size)[None, :]
    return (((y // tile_size) + (x // tile_size)) % 2).astype(float)


def multiscale_pattern(size: int) -> np.ndarray:
    """
    Construct a simple hierarchical image with coarse, medium, and fine structure.

    The top half contains broad horizontal bands.
    The bottom-left quadrant contains medium vertical stripes.
    The bottom-right quadrant contains a fine checkerboard.
    """
    image = np.zeros((size, size), dtype=float)

    top_half = size // 2
    lower_half = size - top_half
    medium_width = max(4, size // 16)
    fine_tile = max(2, size // 32)

    top = horizontal_stripes(size, max(8, size // 8))[:top_half, :]
    image[:top_half, :] = top

    vertical = horizontal_stripes(lower_half, medium_width).T
    image[top_half:, : size // 2] = vertical[:, : size // 2]

    fine = checkerboard(lower_half, fine_tile)
    image[top_half:, size // 2 :] = fine[:, : size - size // 2]

    return image


def generate_images(
    size: int,
    seed: int,
    stripe_width: int,
    checker_tile: int,
) -> dict[str, np.ndarray]:
    return {
        "uniform": uniform_image(size),
        "noise": noise_image(size, seed=seed),
        "stripes": horizontal_stripes(size, stripe_width=stripe_width),
        "checkerboard": checkerboard(size, tile_size=checker_tile),
        "multiscale": multiscale_pattern(size),
    }


def main() -> None:
    args = parse_args()

    validate_size(args.size)
    validate_scale("stripe_width", args.stripe_width)
    validate_scale("checker_tile", args.checker_tile)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = generate_images(
        size=args.size,
        seed=args.seed,
        stripe_width=args.stripe_width,
        checker_tile=args.checker_tile,
    )

    print(f"Writing toy images to: {args.output_dir}")
    print(f"Size: {args.size} x {args.size}")
    print(f"Value range: {args.value_range}")
    print(f"Noise seed: {args.seed}")
    print()

    for name, image in images.items():
        path = args.output_dir / f"{name}.png"
        save_image(to_output_range(image, args.value_range), path)
        print(f"Saved {name:<12s} {path}")


if __name__ == "__main__":
    main()
