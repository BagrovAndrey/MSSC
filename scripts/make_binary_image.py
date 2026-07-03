from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mssc.image_io import load_image, save_image


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an arbitrary image to a square binary black-and-white PNG."
    )
    parser.add_argument("input", type=Path, help="Input image path.")
    parser.add_argument(
        "--size",
        default="auto",
        help=(
            "'auto': resize to nearest power-of-two square. Default. "
            "'none': no resize; require a square power-of-two image. "
            "INT: resize to INT x INT."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "Threshold applied after grayscale conversion in the [-1, 1] range. "
            "Pixels >= threshold become white, others black. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--normalize",
        choices=["minmax", "none"],
        default="minmax",
        help=(
            "Intensity normalization applied to the grayscale image before thresholding. "
            "Default: minmax."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("test_images"),
        help="Output directory. Default: test_images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output file path. Overrides --out-dir naming.",
    )
    return parser.parse_args()


def binarize_image(image: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    gray = np.asarray(image, dtype=float)

    if gray.ndim != 2:
        raise ValueError("binarize_image expects a 2D grayscale image")

    return np.where(gray >= threshold, 1.0, -1.0)


def normalize_grayscale(
    image: np.ndarray,
    mode: str = "minmax",
    eps: float = 1e-15,
) -> np.ndarray:
    gray = np.asarray(image, dtype=float)

    if gray.ndim != 2:
        raise ValueError("normalize_grayscale expects a 2D grayscale image")

    if mode == "none":
        return gray

    if mode != "minmax":
        raise ValueError("normalize mode must be 'minmax' or 'none'")

    vmin = float(np.min(gray))
    vmax = float(np.max(gray))

    if vmax - vmin <= eps:
        return np.zeros_like(gray)

    scaled = (gray - vmin) / (vmax - vmin)
    return 2.0 * scaled - 1.0


def default_output_path(input_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{input_path.stem}_binary.png"


def main() -> None:
    args = parse_args()

    image = load_image(
        args.input,
        size=parse_size(args.size),
        mode="grayscale",
        value_range="minus1_1",
    )
    normalized = normalize_grayscale(image, mode=args.normalize)
    binary = binarize_image(normalized, threshold=args.threshold)

    if args.output is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_output_path(args.input, args.out_dir)
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(binary, output_path)

    print(f"Input: {args.input}")
    print(f"Processed shape: {binary.shape}")
    print("Mode: grayscale -> normalized grayscale -> binary black/white")
    print(f"Normalization: {args.normalize}")
    print(f"Threshold: {args.threshold}")
    print(f"Saved binary image: {output_path}")


if __name__ == "__main__":
    main()
