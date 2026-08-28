from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.image_io import load_image
from scripts.benchmark_toy_panel import (
    make_checkerboard,
    make_nested_dyadic,
    make_noise,
    make_patchwork,
    make_spectral_fractal_binary,
    make_stripes,
)
from scripts.diagnose_jlocq_outlier import make_wavy_stripes
from scripts.make_binary_image import binarize_image, normalize_grayscale
from scripts.validate_mvp_complexity import EPS, compute_weighted_tree_profiles, require_matplotlib


SYNTHETIC_ORDER = [
    "stripes",
    "noise",
    "wavy_stripes",
    "fractal",
    "nested_dyadic",
    "patchwork",
]


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    size = int(value)
    if size <= 0:
        raise ValueError("--size must be positive")
    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale-resolved diagnostic for current nested structural complexity."
    )
    parser.add_argument("--size", default="512", help="Synthetic size as INT. Natural-image loader accepts 'auto' or 'none'.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for stochastic synthetic generators.")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional number of RG steps.")
    parser.add_argument(
        "--min-qenergy-pairs",
        type=int,
        default=32,
        help="Minimum number of native neighbor pairs required for Qenergy. Default: 32.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("ORIGINAL", "BINARY"),
        default=[],
        help="Optional original/binary pair for natural-image binarization panel. Repeat as needed.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def make_synthetic_images(size: int, seed: int) -> dict[str, np.ndarray]:
    return {
        "stripes": make_stripes(size, period=16, orientation="vertical"),
        "noise": make_noise(size, seed=seed),
        "wavy_stripes": make_wavy_stripes(
            size=size,
            stripe_period=64.0,
            wave_amplitude=24.0,
            wave_period=256.0,
            threshold=0.0,
            binary=True,
        ),
        "fractal": make_spectral_fractal_binary(size, beta=2.5, seed=seed),
        "nested_dyadic": make_nested_dyadic(size),
        "patchwork": make_patchwork(size),
        "checkerboard": make_checkerboard(size, cell_size=1),
    }


def load_grayscale_image(path: Path, size: int | str | None) -> np.ndarray:
    return load_image(
        path,
        size=size,
        mode="grayscale",
        value_range="minus1_1",
    )


def analyze_image(
    image_name: str,
    panel: str,
    variant: str,
    image: np.ndarray,
    n_steps: int | None,
    min_qenergy_pairs: int,
) -> dict[str, object]:
    result = compute_weighted_tree_profiles(
        image=image,
        n_steps=n_steps,
        min_qenergy_pairs=min_qenergy_pairs,
    )
    c_profile = np.asarray(result["C_profile"], dtype=float)
    qenergy = np.asarray(result["Qenergy_profile"], dtype=float)
    qdefined = np.asarray(result["Qenergy_defined"], dtype=bool)
    jnested_profile = np.asarray(result["Jnested_profile"], dtype=float)

    knested_profile = np.full_like(c_profile, np.nan, dtype=float)
    valid_c = c_profile > EPS
    knested_profile[valid_c] = jnested_profile[valid_c] / c_profile[valid_c]

    cdetail = float(result["Cdetail"])
    jnested = float(result["Jnested"])
    jhetero = float(result["Jhetero"])
    jstruct = float(result["Jstruct"])
    knested = float("nan") if cdetail <= EPS else jnested / cdetail
    khetero = float("nan") if cdetail <= EPS else jhetero / cdetail
    kstruct = float("nan") if cdetail <= EPS else jstruct / cdetail

    return {
        "image_name": image_name,
        "panel": panel,
        "variant": variant,
        "image": np.asarray(image, dtype=float),
        "C_profile": c_profile,
        "Qenergy_profile": qenergy,
        "Qenergy_defined": qdefined,
        "Knested_profile": knested_profile,
        "Jnested_profile": jnested_profile,
        "Cdetail": cdetail,
        "Knested": knested,
        "Khetero": khetero,
        "Kstruct": kstruct,
        "Jnested": jnested,
        "Jhetero": jhetero,
        "Jstruct": jstruct,
    }


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(result: dict[str, object]) -> None:
    name = str(result["image_name"])
    variant = str(result["variant"])
    cdetail = float(result["Cdetail"])
    knested = float(result["Knested"])
    khetero = float(result["Khetero"])
    kstruct = float(result["Kstruct"])
    jnested = float(result["Jnested"])
    jhetero = float(result["Jhetero"])
    jstruct = float(result["Jstruct"])
    c_profile = np.asarray(result["C_profile"], dtype=float)
    qenergy = np.asarray(result["Qenergy_profile"], dtype=float)
    qdefined = np.asarray(result["Qenergy_defined"], dtype=bool)
    knested_profile = np.asarray(result["Knested_profile"], dtype=float)
    jnested_profile = np.asarray(result["Jnested_profile"], dtype=float)

    print()
    print(f"{name} [{variant}]")
    print(
        f"  Cdetail={cdetail:.6g} "
        f"Knested={knested:.6g} "
        f"Khetero={khetero:.6g} "
        f"Kstruct={kstruct:.6g}"
    )
    print(
        f"  Jnested={jnested:.6g} "
        f"Jhetero={jhetero:.6g} "
        f"Jstruct={jstruct:.6g}"
    )
    print(
        f"  sum_k C_k={np.sum(c_profile):.6g} "
        f"sum_k Jnested_k={np.sum(jnested_profile):.6g} "
        f"existing Jnested={jnested:.6g} "
        f"diff={np.sum(jnested_profile) - jnested:.6g}"
    )
    print("  k    C_k         Qenergy_k   K_nested_k  J_nested_k")
    for k in range(len(c_profile)):
        qtext = f"{qenergy[k]:.6g}" if qdefined[k] and np.isfinite(qenergy[k]) else "undefined"
        ktext = f"{knested_profile[k]:.6g}" if np.isfinite(knested_profile[k]) else "nan"
        print(
            f"  {k:<2d} "
            f"{c_profile[k]:>10.6g} "
            f"{qtext:>11s} "
            f"{ktext:>11s} "
            f"{jnested_profile[k]:>11.6g}"
        )


def ratio_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return float("nan")
    return float(numerator / denominator)


def print_pair_ratios(original: dict[str, object], binary: dict[str, object]) -> None:
    name = str(original["image_name"])
    print()
    print(f"{name} binary/original ratios")
    print(f"  Cdetail = {ratio_or_nan(float(binary['Cdetail']), float(original['Cdetail'])):.6g}")
    print(f"  Knested = {ratio_or_nan(float(binary['Knested']), float(original['Knested'])):.6g}")
    print(f"  Jnested = {ratio_or_nan(float(binary['Jnested']), float(original['Jnested'])):.6g}")
    print(f"  Kstruct = {ratio_or_nan(float(binary['Kstruct']), float(original['Kstruct'])):.6g}")
    print(f"  Jstruct = {ratio_or_nan(float(binary['Jstruct']), float(original['Jstruct'])):.6g}")


def _plot_row(ax_image, ax_graph, result: dict[str, object], yleft_max: float | None = None, yright_max: float | None = None) -> None:
    image = np.asarray(result["image"], dtype=float)
    c_profile = np.asarray(result["C_profile"], dtype=float)
    qenergy = np.asarray(result["Qenergy_profile"], dtype=float)
    qdefined = np.asarray(result["Qenergy_defined"], dtype=bool)
    knested_profile = np.asarray(result["Knested_profile"], dtype=float)
    jnested_profile = np.asarray(result["Jnested_profile"], dtype=float)
    k = np.arange(len(c_profile))

    ax_image.imshow(image, cmap="gray", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_title(str(result["image_name"]).replace("_", " "), fontsize=11)

    ax_right = ax_graph.twinx()
    line_c, = ax_graph.plot(k, c_profile, color="#1d3557", marker="o", linewidth=2.0, markersize=4.5, label="C_k")
    line_j, = ax_graph.plot(k, jnested_profile, color="#d62828", marker="s", linewidth=2.0, markersize=4.5, label="Jnested_k")

    q_plot = qenergy.copy()
    q_plot[~qdefined] = np.nan
    line_q, = ax_right.plot(k, q_plot, color="#2a9d8f", marker="^", linewidth=2.0, markersize=4.5, label="Qenergy_k")
    undefined_k = k[~qdefined]
    if undefined_k.size:
        ax_right.scatter(undefined_k, np.zeros_like(undefined_k, dtype=float), marker="x", s=42, color="#2a9d8f")
    line_k, = ax_right.plot(k, knested_profile, color="#bc6c25", marker="D", linewidth=2.0, markersize=4.2, label="Knested_k")

    ax_graph.set_xlabel("RG scale k", fontsize=10)
    ax_graph.set_ylabel("C_k, Jnested_k", fontsize=10)
    ax_right.set_ylabel("Qenergy_k, Knested_k", fontsize=10)
    ax_graph.grid(alpha=0.25, linewidth=0.8)
    ax_graph.tick_params(labelsize=9)
    ax_right.tick_params(labelsize=9)

    ax_graph.set_xlim(-0.2, len(k) - 0.8 if len(k) else 0.8)
    if yleft_max is not None:
        ax_graph.set_ylim(0.0, yleft_max)
    if yright_max is not None:
        ax_right.set_ylim(0.0, yright_max)

    handles = [line_c, line_j, line_q, line_k]
    labels = [handle.get_label() for handle in handles]
    ax_graph.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=2)


def save_synthetic_figure(path: Path, results: list[dict[str, object]]) -> None:
    plt = require_matplotlib()
    nrows = len(results)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(10.5, max(2.9 * nrows, 8.0)),
        gridspec_kw={"width_ratios": [1.0, 2.7]},
        constrained_layout=True,
    )

    if nrows == 1:
        axes = np.asarray([axes], dtype=object)

    for row_axes, result in zip(axes, results):
        _plot_row(row_axes[0], row_axes[1], result)

    fig.suptitle("Scale-resolved synthetic diagnostics", fontsize=15)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_binarization_figure(path: Path, pair_results: list[tuple[dict[str, object], dict[str, object]]]) -> None:
    plt = require_matplotlib()
    nrows = 2 * len(pair_results)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(10.5, max(2.9 * nrows, 7.5)),
        gridspec_kw={"width_ratios": [1.0, 2.7]},
        constrained_layout=True,
    )

    if nrows == 1:
        axes = np.asarray([axes], dtype=object)

    for pair_index, (original, binary) in enumerate(pair_results):
        yleft_max = 1.05 * max(
            np.max(np.asarray(original["C_profile"], dtype=float)),
            np.max(np.asarray(original["Jnested_profile"], dtype=float)),
            np.max(np.asarray(binary["C_profile"], dtype=float)),
            np.max(np.asarray(binary["Jnested_profile"], dtype=float)),
            EPS,
        )
        right_candidates = [EPS]
        for data in (original, binary):
            q_vals = np.asarray(data["Qenergy_profile"], dtype=float)
            k_vals = np.asarray(data["Knested_profile"], dtype=float)
            if np.isfinite(q_vals).any():
                right_candidates.append(float(np.nanmax(q_vals)))
            if np.isfinite(k_vals).any():
                right_candidates.append(float(np.nanmax(k_vals)))
        yright_max = 1.05 * max(right_candidates)

        row0 = 2 * pair_index
        row1 = row0 + 1
        _plot_row(axes[row0, 0], axes[row0, 1], original, yleft_max=yleft_max, yright_max=yright_max)
        _plot_row(axes[row1, 0], axes[row1, 1], binary, yleft_max=yleft_max, yright_max=yright_max)
        axes[row0, 1].set_title(f"{original['image_name']} original", fontsize=11)
        axes[row1, 1].set_title(f"{binary['image_name']} binary", fontsize=11)

    fig.suptitle("Scale-resolved binarization diagnostics", fontsize=15)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    size_arg = parse_size(args.size)

    if args.min_qenergy_pairs < 0:
        raise ValueError("--min-qenergy-pairs must be nonnegative")

    synthetic_size = 512 if size_arg in {"auto", None} else int(size_arg)
    if not is_power_of_two(synthetic_size):
        raise ValueError("Synthetic --size must be a power of two")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_images = make_synthetic_images(synthetic_size, args.seed)
    synthetic_results: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []

    for label in SYNTHETIC_ORDER:
        result = analyze_image(
            image_name=label,
            panel="synthetic",
            variant="synthetic",
            image=synthetic_images[label],
            n_steps=args.n_steps,
            min_qenergy_pairs=args.min_qenergy_pairs,
        )
        synthetic_results.append(result)
        print_summary(result)

        for k in range(len(np.asarray(result["C_profile"], dtype=float))):
            csv_rows.append(
                {
                    "panel": "synthetic",
                    "image_name": label,
                    "variant": "synthetic",
                    "k": k,
                    "C": float(np.asarray(result["C_profile"], dtype=float)[k]),
                    "Qenergy": float(np.asarray(result["Qenergy_profile"], dtype=float)[k]),
                    "K_nested": float(np.asarray(result["Knested_profile"], dtype=float)[k]),
                    "J_nested": float(np.asarray(result["Jnested_profile"], dtype=float)[k]),
                }
            )

    pair_results: list[tuple[dict[str, object], dict[str, object]]] = []
    for original_path_str, binary_path_str in args.pair:
        original_path = Path(original_path_str)
        binary_path = Path(binary_path_str)
        pair_name = original_path.stem

        original_image = load_grayscale_image(original_path, size=size_arg)
        binary_image = load_grayscale_image(binary_path, size=size_arg)

        # Keep the current binarization convention available for user-side
        # cross-checks without changing the provided binary file path.
        _ = binarize_image(normalize_grayscale(original_image, mode="minmax"), threshold=0.0)

        original = analyze_image(
            image_name=pair_name,
            panel="binarization",
            variant="original",
            image=original_image,
            n_steps=args.n_steps,
            min_qenergy_pairs=args.min_qenergy_pairs,
        )
        binary = analyze_image(
            image_name=pair_name,
            panel="binarization",
            variant="binary",
            image=binary_image,
            n_steps=args.n_steps,
            min_qenergy_pairs=args.min_qenergy_pairs,
        )
        pair_results.append((original, binary))
        print_summary(original)
        print_summary(binary)
        print_pair_ratios(original, binary)

        for result in (original, binary):
            for k in range(len(np.asarray(result["C_profile"], dtype=float))):
                csv_rows.append(
                    {
                        "panel": "binarization",
                        "image_name": pair_name,
                        "variant": str(result["variant"]),
                        "k": k,
                        "C": float(np.asarray(result["C_profile"], dtype=float)[k]),
                        "Qenergy": float(np.asarray(result["Qenergy_profile"], dtype=float)[k]),
                        "K_nested": float(np.asarray(result["Knested_profile"], dtype=float)[k]),
                        "J_nested": float(np.asarray(result["Jnested_profile"], dtype=float)[k]),
                    }
                )

    save_csv_rows(
        args.out_dir / "scale_resolved_nested.csv",
        ["panel", "image_name", "variant", "k", "C", "Qenergy", "K_nested", "J_nested"],
        csv_rows,
    )

    try:
        save_synthetic_figure(args.out_dir / "scale_resolved_synthetic.png", synthetic_results)
        if pair_results:
            save_binarization_figure(args.out_dir / "scale_resolved_binarization.png", pair_results)
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped figure generation.")


if __name__ == "__main__":
    main()
