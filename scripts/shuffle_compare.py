from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mssc.complexity import complexity_profile
from mssc.image_io import load_image, save_image
from mssc.orientation import (
    haar_channel_energy_profile,
    local_orientation_coherence_profile,
    orientation_diverse_organized_profile,
    orientation_entropy_profile,
    organized_profile,
    scale_orientation_entropy_profile,
)
from mssc.shuffle import phase_scramble, tile_shuffle


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    return int(value)


def profile_entropy(profile: np.ndarray, eps: float = 1e-15) -> float:
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))

    if total <= eps:
        return 0.0

    p = x / total
    p = p[p > eps]

    return float(-np.sum(p * np.log(p)))


def entropic_complexity(profile: np.ndarray, eps: float = 1e-15) -> float:
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))

    if total <= eps:
        return 0.0

    return total * profile_entropy(x, eps=eps)


def normalize_intensity(
    image: np.ndarray,
    mode: str = "none",
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Optional intensity normalization before computing MSSC.

    mode="none":
        Return the image unchanged.

    mode="minmax":
        Affinely map the image to the full input range.

        If the image contains negative values, assume the natural range is [-1, 1].
        Otherwise assume [0, 1].

        For RGB/vector images, the min and max are taken globally across
        all channels. This preserves relative color-channel amplitudes.
    """
    img = np.asarray(image, dtype=float)

    if mode == "none":
        return img

    if mode != "minmax":
        raise ValueError("normalize_intensity mode must be 'none' or 'minmax'")

    vmin = float(np.min(img))
    vmax = float(np.max(img))

    if vmax - vmin <= eps:
        return np.zeros_like(img)

    if vmin < 0:
        out_min, out_max = -1.0, 1.0
    else:
        out_min, out_max = 0.0, 1.0

    normalized = (img - vmin) / (vmax - vmin)
    normalized = out_min + normalized * (out_max - out_min)

    return normalized


def summarize_profiles(
    C: np.ndarray,
    Q: np.ndarray,
    D: np.ndarray,
    O: np.ndarray,
    Odiv: np.ndarray,
    J: np.ndarray,
) -> dict[str, float]:
    return {
        "C_total": float(np.sum(C)),
        "O_total": float(np.sum(O)),
        "Odiv_total": float(np.sum(Odiv)),
        "J_total": float(np.sum(J)),

        "H_C": profile_entropy(C),
        "H_O": profile_entropy(O),
        "H_Odiv": profile_entropy(Odiv),
        "H_J": profile_entropy(J),

        "S_C": entropic_complexity(C),
        "S_O": entropic_complexity(O),
        "S_Odiv": entropic_complexity(Odiv),
        "S_J": entropic_complexity(J),

        "Q_mean": float(np.mean(Q)) if len(Q) else 0.0,
        "D_mean": float(np.mean(D)) if len(D) else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MSSC and organized MSSC profiles of an image with a scrambled version."
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
    parser.add_argument("--mode", choices=["rgb", "grayscale"], default="grayscale")
    parser.add_argument(
        "--value-range",
        choices=["0_1", "minus1_1"],
        default="minus1_1",
    )

    parser.add_argument(
        "--normalize-intensity",
        choices=["none", "minmax"],
        default="none",
        help="Optional intensity normalization before MSSC. Default: none.",
    )
    parser.add_argument(
        "--compare-normalized",
        action="store_true",
        help="Run both raw and minmax-normalized analyses.",
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


def show_image(ax, image: np.ndarray) -> None:
    display = to_display_image(image)

    if display.ndim == 2:
        ax.imshow(display, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(display, interpolation="nearest")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if block_size != 2:
        raise ValueError(
            "Orientation observables are currently implemented only for block_size=2"
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

    D = orientation_entropy_profile(
        image,
        n_steps=n_steps,
    )
    E = haar_channel_energy_profile(
        image,
        n_steps=n_steps,
    )

    O = organized_profile(C, Q)
    Odiv = orientation_diverse_organized_profile(C, Q, D)
    J = scale_orientation_entropy_profile(E, Q)

    summary = summarize_profiles(C, Q, D, O, Odiv, J)

    return C, Q, D, O, Odiv, J, summary


def save_csv(
    path: Path,
    original_C: np.ndarray,
    original_Q: np.ndarray,
    original_D: np.ndarray,
    original_O: np.ndarray,
    original_Odiv: np.ndarray,
    original_J: np.ndarray,
    scrambled_C: np.ndarray,
    scrambled_Q: np.ndarray,
    scrambled_D: np.ndarray,
    scrambled_O: np.ndarray,
    scrambled_Odiv: np.ndarray,
    scrambled_J: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "original_C",
                "original_Q",
                "original_D",
                "original_O",
                "original_Odiv",
                "original_J",
                "scrambled_C",
                "scrambled_Q",
                "scrambled_D",
                "scrambled_O",
                "scrambled_Odiv",
                "scrambled_J",
            ]
        )

        for k in range(len(original_C)):
            writer.writerow(
                [
                    k,
                    original_C[k],
                    original_Q[k],
                    original_D[k],
                    original_O[k],
                    original_Odiv[k],
                    original_J[k],
                    scrambled_C[k],
                    scrambled_Q[k],
                    scrambled_D[k],
                    scrambled_O[k],
                    scrambled_Odiv[k],
                    scrambled_J[k],
                ]
            )


def format_summary_for_title(summary: dict[str, float]) -> str:
    return (
        f"C={summary['C_total']:.4g}, "
        f"O={summary['O_total']:.4g}\n"
        f"Odiv={summary['Odiv_total']:.4g}, "
        f"J={summary['J_total']:.4g}"
    )


def save_comparison_plot(
    path: Path,
    original_image: np.ndarray,
    scrambled_image: np.ndarray,
    original_C: np.ndarray,
    original_Q: np.ndarray,
    original_D: np.ndarray,
    original_O: np.ndarray,
    original_Odiv: np.ndarray,
    original_J: np.ndarray,
    scrambled_C: np.ndarray,
    scrambled_Q: np.ndarray,
    scrambled_D: np.ndarray,
    scrambled_O: np.ndarray,
    scrambled_Odiv: np.ndarray,
    scrambled_J: np.ndarray,
    original_summary: dict[str, float],
    scrambled_summary: dict[str, float],
    scramble_label: str,
    normalization_label: str,
) -> None:
    k = np.arange(len(original_C))

    fig = plt.figure(figsize=(11, 12))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.15, 0.7, 0.7, 0.7, 0.7])

    ax_img_1 = fig.add_subplot(gs[0, 0])
    ax_img_2 = fig.add_subplot(gs[0, 1])

    ax_C = fig.add_subplot(gs[1, :])
    ax_Q = fig.add_subplot(gs[2, :])
    ax_D = fig.add_subplot(gs[3, :])
    ax_O = fig.add_subplot(gs[4, :])

    show_image(ax_img_1, original_image)
    ax_img_1.set_title("Original\n" + format_summary_for_title(original_summary))
    ax_img_1.set_xticks([])
    ax_img_1.set_yticks([])

    show_image(ax_img_2, scrambled_image)
    ax_img_2.set_title(scramble_label + "\n" + format_summary_for_title(scrambled_summary))
    ax_img_2.set_xticks([])
    ax_img_2.set_yticks([])

    ax_C.plot(k, original_C, marker="o", label="original")
    ax_C.plot(k, scrambled_C, marker="o", label=scramble_label)
    ax_C.set_ylabel("C_k")
    ax_C.set_title(f"Naive MSSC profile ({normalization_label})")
    ax_C.legend()

    ax_Q.plot(k, original_Q, marker="o", label="original")
    ax_Q.plot(k, scrambled_Q, marker="o", label=scramble_label)
    ax_Q.axhline(0.0, linewidth=1)
    ax_Q.set_ylabel("Q_k")
    ax_Q.set_title("Local orientation coherence")
    ax_Q.legend()

    ax_D.plot(k, original_D, marker="o", label="original")
    ax_D.plot(k, scrambled_D, marker="o", label=scramble_label)
    ax_D.set_ylabel("D_k")
    ax_D.set_title("Orientation entropy / diversity")
    ax_D.legend()

    ax_O.plot(k, original_O, marker="o", label="original O = C Q")
    ax_O.plot(k, scrambled_O, marker="o", label=f"{scramble_label} O = C Q")
    ax_O.plot(k, original_Odiv, marker="s", label="original Odiv = C Q D")
    ax_O.plot(k, scrambled_Odiv, marker="s", label=f"{scramble_label} Odiv = C Q D")
    ax_O.plot(k, original_J, marker="^", label="original J")
    ax_O.plot(k, scrambled_J, marker="^", label=f"{scramble_label} J")
    ax_O.set_xlabel("Scale index k")
    ax_O.set_ylabel("Profile value")
    ax_O.set_title("Ordered, orientation-diverse, and scale-orientation profiles")
    ax_O.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(name: str, summary: dict[str, float]) -> None:
    print(name)
    print(f"  C_total    = {summary['C_total']:.12g}")
    print(f"  O_total    = {summary['O_total']:.12g}")
    print(f"  Odiv_total = {summary['Odiv_total']:.12g}")
    print(f"  J_total    = {summary['J_total']:.12g}")
    print(f"  H_C        = {summary['H_C']:.12g}")
    print(f"  H_O        = {summary['H_O']:.12g}")
    print(f"  H_Odiv     = {summary['H_Odiv']:.12g}")
    print(f"  H_J        = {summary['H_J']:.12g}")
    print(f"  S_C        = {summary['S_C']:.12g}")
    print(f"  S_O        = {summary['S_O']:.12g}")
    print(f"  S_Odiv     = {summary['S_Odiv']:.12g}")
    print(f"  S_J        = {summary['S_J']:.12g}")
    print(f"  Q_mean     = {summary['Q_mean']:.12g}")
    print(f"  D_mean     = {summary['D_mean']:.12g}")


def add_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def run_one_analysis(
    image_loaded: np.ndarray,
    args: argparse.Namespace,
    normalization_mode: str,
    out_plot: Path | None,
    out_csv: Path | None,
    save_scrambled: Path | None,
) -> None:
    image = normalize_intensity(
        image_loaded,
        mode=normalization_mode,
    )

    scrambled, scramble_label = make_scrambled_image(
        image,
        scramble=args.scramble,
        tile_size=args.tile_size,
        seed=args.seed,
    )

    (
        original_C,
        original_Q,
        original_D,
        original_O,
        original_Odiv,
        original_J,
        original_summary,
    ) = compute_all_profiles(
        image,
        block_size=args.block_size,
        n_steps=args.n_steps,
    )

    (
        scrambled_C,
        scrambled_Q,
        scrambled_D,
        scrambled_O,
        scrambled_Odiv,
        scrambled_J,
        scrambled_summary,
    ) = compute_all_profiles(
        scrambled,
        block_size=args.block_size,
        n_steps=args.n_steps,
    )

    print()
    print("=" * 72)
    print(f"Intensity normalization: {normalization_mode}")
    print("=" * 72)
    print_summary("Original:", original_summary)
    print()
    print_summary("Scrambled:", scrambled_summary)
    print()

    print(
        "k "
        "original_C original_Q original_D original_O original_Odiv original_J "
        "scrambled_C scrambled_Q scrambled_D scrambled_O scrambled_Odiv scrambled_J"
    )
    for k in range(len(original_C)):
        print(
            f"{k} "
            f"{original_C[k]:.12g} {original_Q[k]:.12g} "
            f"{original_D[k]:.12g} {original_O[k]:.12g} {original_Odiv[k]:.12g} "
            f"{original_J[k]:.12g} "
            f"{scrambled_C[k]:.12g} {scrambled_Q[k]:.12g} "
            f"{scrambled_D[k]:.12g} {scrambled_O[k]:.12g} {scrambled_Odiv[k]:.12g} "
            f"{scrambled_J[k]:.12g}"
        )

    if out_csv is not None:
        save_csv(
            out_csv,
            original_C,
            original_Q,
            original_D,
            original_O,
            original_Odiv,
            original_J,
            scrambled_C,
            scrambled_Q,
            scrambled_D,
            scrambled_O,
            scrambled_Odiv,
            scrambled_J,
        )
        print(f"Saved CSV: {out_csv}")

    if out_plot is not None:
        save_comparison_plot(
            out_plot,
            image,
            scrambled,
            original_C,
            original_Q,
            original_D,
            original_O,
            original_Odiv,
            original_J,
            scrambled_C,
            scrambled_Q,
            scrambled_D,
            scrambled_O,
            scrambled_Odiv,
            scrambled_J,
            original_summary,
            scrambled_summary,
            scramble_label=scramble_label,
            normalization_label=normalization_mode,
        )
        print(f"Saved plot: {out_plot}")

    if save_scrambled is not None:
        save_image(scrambled, save_scrambled)
        print(f"Saved scrambled display image: {save_scrambled}")


def main() -> None:
    args = parse_args()

    image_loaded = load_image(
        args.input,
        size=parse_size(args.size),
        mode=args.mode,
        value_range=args.value_range,
    )

    print(f"Input: {args.input}")
    print(f"Image shape after loading/resizing: {image_loaded.shape}")
    print(f"Mode: {args.mode}")
    print(f"Value range: {args.value_range}")
    print(f"Scramble mode: {args.scramble}")
    print(f"Seed: {args.seed}")

    if args.scramble == "tile":
        print(f"Tile size: {args.tile_size}")

    if args.compare_normalized:
        run_one_analysis(
            image_loaded=image_loaded,
            args=args,
            normalization_mode="none",
            out_plot=add_suffix(args.out_plot, "raw") if args.out_plot else None,
            out_csv=add_suffix(args.out_csv, "raw") if args.out_csv else None,
            save_scrambled=add_suffix(args.save_scrambled, "raw") if args.save_scrambled else None,
        )

        run_one_analysis(
            image_loaded=image_loaded,
            args=args,
            normalization_mode="minmax",
            out_plot=add_suffix(args.out_plot, "minmax") if args.out_plot else None,
            out_csv=add_suffix(args.out_csv, "minmax") if args.out_csv else None,
            save_scrambled=add_suffix(args.save_scrambled, "minmax") if args.save_scrambled else None,
        )

    else:
        run_one_analysis(
            image_loaded=image_loaded,
            args=args,
            normalization_mode=args.normalize_intensity,
            out_plot=args.out_plot,
            out_csv=args.out_csv,
            save_scrambled=args.save_scrambled,
        )


if __name__ == "__main__":
    main()
