from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mssc.complexity import complexity_profile
from mssc.image_io import load_image, save_image
from mssc.orientation import local_orientation_coherence_profile, organized_profile
from mssc.shuffle import phase_scramble, tile_shuffle


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    return int(value)


def profile_entropy(profile: np.ndarray, eps: float = 1e-15) -> float:
    """
    Entropy of the normalized scale profile.

    H = -sum_k p_k log p_k,
    p_k = profile_k / sum_j profile_j.

    If the profile has zero total weight, return 0.
    """
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))

    if total <= eps:
        return 0.0

    p = x / total
    p = p[p > eps]

    return float(-np.sum(p * np.log(p)))


def entropic_complexity(profile: np.ndarray, eps: float = 1e-15) -> float:
    """
    Total weight times entropy over scales.

    S = (sum_k profile_k) * H(profile_k / sum_j profile_j)
    """
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))

    if total <= eps:
        return 0.0

    return total * profile_entropy(x, eps=eps)


def summarize_profiles(C: np.ndarray, Q: np.ndarray, O: np.ndarray) -> dict[str, float]:
    return {
        "C_total": float(np.sum(C)),
        "O_total": float(np.sum(O)),
        "H_C": profile_entropy(C),
        "H_O": profile_entropy(O),
        "S_C": entropic_complexity(C),
        "S_O": entropic_complexity(O),
        "Q_mean": float(np.mean(Q)) if len(Q) else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MSSC and organized MSSC profiles of an image with a scrambled version."
    )

    parser.add_argument("input", type=Path)

    parser.add_argument(
        "--size",
        default="auto",
        help="'auto', integer size, or 'none'. Default: auto",
    )
    parser.add_argument("--mode", choices=["rgb", "grayscale"], default="rgb")
    parser.add_argument(
        "--value-range",
        choices=["0_1", "minus1_1"],
        default="minus1_1",
    )

    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=None)

    parser.add_argument(
        "--scramble",
        choices=["tile", "phase"],
        default="tile",
        help="Scrambling mode. Default: tile",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size. Required for --scramble tile.",
    )
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-plot", type=Path, default=None)
    parser.add_argument("--save-scrambled", type=Path, default=None)

    return parser.parse_args()


def to_display_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)

    if arr.ndim == 3:
        if arr.min() < 0:
            arr = 0.5 * (arr + 1.0)
        return np.clip(arr, 0.0, 1.0)

    vmin = arr.min()
    vmax = arr.max()

    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    return arr


def make_scrambled_image(
    image: np.ndarray,
    scramble: str,
    tile_size: int | None,
    seed: int | None,
) -> tuple[np.ndarray, str]:
    if scramble == "tile":
        if tile_size is None:
            raise ValueError("--tile-size is required for --scramble tile")

        scrambled = tile_shuffle(
            image,
            tile_size=tile_size,
            seed=seed,
        )
        label = f"Tile-shuffled, tile={tile_size}"
        return scrambled, label

    if scramble == "phase":
        scrambled = phase_scramble(
            image,
            seed=seed,
            preserve_dc=True,
        )
        label = "Phase-scrambled"
        return scrambled, label

    raise ValueError(f"unknown scramble mode: {scramble}")


def compute_all_profiles(
    image: np.ndarray,
    block_size: int = 2,
    n_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if block_size != 2:
        raise ValueError(
            "Orientation coherence is currently implemented only for block_size=2"
        )

    C = complexity_profile(
        image,
        block_size=block_size,
        n_steps=n_steps,
    )

    Q = local_orientation_coherence_profile(
        image,
        n_steps=n_steps,
    )

    O = organized_profile(C, Q)

    summary = summarize_profiles(C, Q, O)

    return C, Q, O, summary


def save_csv(
    path: Path,
    original_C: np.ndarray,
    original_Q: np.ndarray,
    original_O: np.ndarray,
    scrambled_C: np.ndarray,
    scrambled_Q: np.ndarray,
    scrambled_O: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "original_C",
                "original_Q",
                "original_O",
                "scrambled_C",
                "scrambled_Q",
                "scrambled_O",
            ]
        )

        for k in range(len(original_C)):
            writer.writerow(
                [
                    k,
                    original_C[k],
                    original_Q[k],
                    original_O[k],
                    scrambled_C[k],
                    scrambled_Q[k],
                    scrambled_O[k],
                ]
            )


def format_summary_for_title(summary: dict[str, float]) -> str:
    return (
        f"C={summary['C_total']:.4g}, "
        f"O={summary['O_total']:.4g}\n"
        f"S_C={summary['S_C']:.4g}, "
        f"S_O={summary['S_O']:.4g}"
    )


def save_comparison_plot(
    path: Path,
    original_image: np.ndarray,
    scrambled_image: np.ndarray,
    original_C: np.ndarray,
    original_Q: np.ndarray,
    original_O: np.ndarray,
    scrambled_C: np.ndarray,
    scrambled_Q: np.ndarray,
    scrambled_O: np.ndarray,
    original_summary: dict[str, float],
    scrambled_summary: dict[str, float],
    scramble_label: str,
) -> None:
    k = np.arange(len(original_C))

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 0.75, 0.75, 0.75])

    ax_img_1 = fig.add_subplot(gs[0, 0])
    ax_img_2 = fig.add_subplot(gs[0, 1])

    ax_C = fig.add_subplot(gs[1, :])
    ax_Q = fig.add_subplot(gs[2, :])
    ax_O = fig.add_subplot(gs[3, :])

    ax_img_1.imshow(to_display_image(original_image), interpolation="nearest")
    ax_img_1.set_title("Original\n" + format_summary_for_title(original_summary))
    ax_img_1.set_xticks([])
    ax_img_1.set_yticks([])

    ax_img_2.imshow(to_display_image(scrambled_image), interpolation="nearest")
    ax_img_2.set_title(scramble_label + "\n" + format_summary_for_title(scrambled_summary))
    ax_img_2.set_xticks([])
    ax_img_2.set_yticks([])

    ax_C.plot(k, original_C, marker="o", label="original")
    ax_C.plot(k, scrambled_C, marker="o", label=scramble_label)
    ax_C.set_ylabel("C_k")
    ax_C.set_title("Naive MSSC profile")
    ax_C.legend()

    ax_Q.plot(k, original_Q, marker="o", label="original")
    ax_Q.plot(k, scrambled_Q, marker="o", label=scramble_label)
    ax_Q.axhline(0.0, linewidth=1)
    ax_Q.set_ylabel("Q_k")
    ax_Q.set_title("Local orientation coherence")
    ax_Q.legend()

    ax_O.plot(k, original_O, marker="o", label="original")
    ax_O.plot(k, scrambled_O, marker="o", label=scramble_label)
    ax_O.set_xlabel("Scale index k")
    ax_O.set_ylabel("O_k")
    ax_O.set_title("Organized MSSC profile: O_k = C_k max(Q_k, 0)")
    ax_O.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(name: str, summary: dict[str, float]) -> None:
    print(name)
    print(f"  C_total = {summary['C_total']:.12g}")
    print(f"  O_total = {summary['O_total']:.12g}")
    print(f"  H_C     = {summary['H_C']:.12g}")
    print(f"  H_O     = {summary['H_O']:.12g}")
    print(f"  S_C     = {summary['S_C']:.12g}")
    print(f"  S_O     = {summary['S_O']:.12g}")
    print(f"  Q_mean  = {summary['Q_mean']:.12g}")


def main() -> None:
    args = parse_args()

    image = load_image(
        args.input,
        size=parse_size(args.size),
        mode=args.mode,
        value_range=args.value_range,
    )

    scrambled, scramble_label = make_scrambled_image(
        image,
        scramble=args.scramble,
        tile_size=args.tile_size,
        seed=args.seed,
    )

    original_C, original_Q, original_O, original_summary = compute_all_profiles(
        image,
        block_size=args.block_size,
        n_steps=args.n_steps,
    )

    scrambled_C, scrambled_Q, scrambled_O, scrambled_summary = compute_all_profiles(
        scrambled,
        block_size=args.block_size,
        n_steps=args.n_steps,
    )

    print(f"Input: {args.input}")
    print(f"Image shape after loading/resizing: {image.shape}")
    print(f"Scramble mode: {args.scramble}")
    print(f"Seed: {args.seed}")

    if args.scramble == "tile":
        print(f"Tile size: {args.tile_size}")

    print()
    print_summary("Original:", original_summary)
    print()
    print_summary("Scrambled:", scrambled_summary)
    print()

    print("k original_C original_Q original_O scrambled_C scrambled_Q scrambled_O")
    for k in range(len(original_C)):
        print(
            f"{k} "
            f"{original_C[k]:.12g} {original_Q[k]:.12g} {original_O[k]:.12g} "
            f"{scrambled_C[k]:.12g} {scrambled_Q[k]:.12g} {scrambled_O[k]:.12g}"
        )

    if args.out_csv is not None:
        save_csv(
            args.out_csv,
            original_C,
            original_Q,
            original_O,
            scrambled_C,
            scrambled_Q,
            scrambled_O,
        )
        print(f"Saved CSV: {args.out_csv}")

    if args.out_plot is not None:
        save_comparison_plot(
            args.out_plot,
            image,
            scrambled,
            original_C,
            original_Q,
            original_O,
            scrambled_C,
            scrambled_Q,
            scrambled_O,
            original_summary,
            scrambled_summary,
            scramble_label=scramble_label,
        )
        print(f"Saved plot: {args.out_plot}")

    if args.save_scrambled is not None:
        save_image(scrambled, args.save_scrambled)
        print(f"Saved scrambled display image: {args.save_scrambled}")


if __name__ == "__main__":
    main()
