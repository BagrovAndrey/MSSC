from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.benchmark_toy_panel import (
    make_noise,
    make_patchwork,
    make_spectral_fractal_binary,
)
from scripts.diagnose_jlocq_outlier import make_wavy_stripes
from scripts.validate_mvp_complexity import EPS, compute_weighted_tree, require_matplotlib


DEFAULT_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_LABELS = ["fractal", "wavy_stripes", "patchwork", "noise"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose current CRH structural complexity under pure global amplitude scaling."
    )
    parser.add_argument("--size", type=int, default=512, help="Synthetic image size. Must be a power of two.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for stochastic generators.")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional number of RG steps.")
    parser.add_argument(
        "--min-qenergy-pairs",
        type=int,
        default=32,
        help="Minimum number of native neighbor pairs required for Qenergy. Default: 32.",
    )
    parser.add_argument(
        "--scale-factors",
        type=float,
        nargs="+",
        default=DEFAULT_FACTORS,
        help="Positive global amplitude scale factors. Default: 0.25 0.5 1.0 2.0 4.0.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        choices=["fractal", "wavy_stripes", "patchwork", "noise"],
        help="Synthetic benchmark images to include.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def built_in_images(size: int, seed: int) -> dict[str, np.ndarray]:
    return {
        "fractal": make_spectral_fractal_binary(size, beta=2.5, seed=seed),
        "wavy_stripes": make_wavy_stripes(
            size=size,
            stripe_period=64.0,
            wave_amplitude=24.0,
            wave_period=256.0,
            threshold=0.0,
            binary=True,
        ),
        "patchwork": make_patchwork(size),
        "noise": make_noise(size, seed=seed),
    }


def ratio_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return float("nan")
    return float(numerator / denominator)


def summarize_scaled_image(
    label: str,
    factor: float,
    image: np.ndarray,
    n_steps: int | None,
    min_qenergy_pairs: int,
) -> dict[str, float | str]:
    totals, _profile_rows = compute_weighted_tree(
        image=image,
        n_steps=n_steps,
        min_qenergy_pairs=min_qenergy_pairs,
    )

    cdetail = float(totals["Cdetail"])
    retained = float(totals["retained_energy_fraction"])
    hstruct = float(totals["Hstruct"])
    hnested = float(totals["Hnested"])
    ihetero = float(totals["Ihetero"])
    jstruct = float(totals["Jstruct"])
    jnested = float(totals["Jnested"])
    jhetero = float(totals["Jhetero"])

    ebar = cdetail
    wbar = ebar * retained if np.isfinite(retained) else float("nan")
    jspecific = float("nan") if ebar <= EPS else jstruct / ebar

    consistency_j = float("nan")
    if np.isfinite(ebar) and np.isfinite(retained) and np.isfinite(hstruct):
        consistency_j = jstruct - ebar * retained * hstruct

    return {
        "label": label,
        "a": factor,
        "a2_reference": factor * factor,
        "Cdetail": cdetail,
        "Wbar": wbar,
        "R": retained,
        "Hnested": hnested,
        "Ihetero": ihetero,
        "Hstruct": hstruct,
        "Jnested": jnested,
        "Jhetero": jhetero,
        "Jstruct": jstruct,
        "Jspecific": jspecific,
        "J_consistency_error": consistency_j,
    }


def add_baseline_ratios(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    baseline_by_label = {
        str(row["label"]): row
        for row in rows
        if abs(float(row["a"]) - 1.0) <= 1e-12
    }

    enriched: list[dict[str, float | str]] = []
    for row in rows:
        label = str(row["label"])
        baseline = baseline_by_label[label]

        row = dict(row)
        row["Cdetail_ratio"] = ratio_or_nan(float(row["Cdetail"]), float(baseline["Cdetail"]))
        row["R_ratio"] = ratio_or_nan(float(row["R"]), float(baseline["R"]))
        row["Hstruct_ratio"] = ratio_or_nan(float(row["Hstruct"]), float(baseline["Hstruct"]))
        row["Jstruct_ratio"] = ratio_or_nan(float(row["Jstruct"]), float(baseline["Jstruct"]))
        row["Jspecific_ratio"] = ratio_or_nan(float(row["Jspecific"]), float(baseline["Jspecific"]))
        enriched.append(row)

    return enriched


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: list[dict[str, float | str]], labels: list[str]) -> None:
    plt = require_matplotlib()
    metrics = [
        ("Cdetail_ratio", "Cdetail / Cdetail(a=1)"),
        ("Jstruct_ratio", "Jstruct / Jstruct(a=1)"),
        ("R_ratio", "R / R(a=1)"),
        ("Hstruct_ratio", "Hstruct / Hstruct(a=1)"),
        ("Jspecific_ratio", "Jspecific / Jspecific(a=1)"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8.5, 14), sharex=True, constrained_layout=True)
    colors = {
        "fractal": "#2a6f97",
        "wavy_stripes": "#d98f4e",
        "patchwork": "#4c956c",
        "noise": "#c1121f",
    }

    for ax, (key, ylabel) in zip(axes, metrics):
        for label in labels:
            subset = [row for row in rows if str(row["label"]) == label]
            x = np.asarray([float(row["a"]) for row in subset], dtype=float)
            y = np.asarray([float(row[key]) for row in subset], dtype=float)
            ax.plot(x, y, marker="o", label=label, color=colors[label])

        if key in {"Cdetail_ratio", "Jstruct_ratio"}:
            xref = np.asarray(sorted({float(row["a"]) for row in rows}), dtype=float)
            ax.plot(xref, xref * xref, linestyle="--", color="black", label="a^2 reference")

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title("Contrast-scaling diagnostic")
    axes[-1].set_xlabel("Amplitude scale factor a")
    axes[0].legend(frameon=False, ncol=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not is_power_of_two(args.size):
        raise ValueError("--size must be a power of two")
    if args.min_qenergy_pairs < 0:
        raise ValueError("--min-qenergy-pairs must be nonnegative")
    if any(f <= 0 for f in args.scale_factors):
        raise ValueError("--scale-factors must all be positive")
    if not any(abs(f - 1.0) <= 1e-12 for f in args.scale_factors):
        raise ValueError("--scale-factors must include 1.0 for baseline ratios")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    images = built_in_images(args.size, args.seed)
    rows: list[dict[str, float | str]] = []

    for label in args.labels:
        image = np.asarray(images[label], dtype=float)
        for factor in args.scale_factors:
            scaled = factor * image
            rows.append(
                summarize_scaled_image(
                    label=label,
                    factor=factor,
                    image=scaled,
                    n_steps=args.n_steps,
                    min_qenergy_pairs=args.min_qenergy_pairs,
                )
            )

    rows = add_baseline_ratios(rows)

    save_csv_rows(
        args.out_dir / "contrast_scaling_diagnostic.csv",
        [
            "label",
            "a",
            "a2_reference",
            "Cdetail",
            "Wbar",
            "R",
            "Hnested",
            "Ihetero",
            "Hstruct",
            "Jnested",
            "Jhetero",
            "Jstruct",
            "Jspecific",
            "Cdetail_ratio",
            "R_ratio",
            "Hstruct_ratio",
            "Jstruct_ratio",
            "Jspecific_ratio",
            "J_consistency_error",
        ],
        rows,
    )

    try:
        save_plot(
            args.out_dir / "contrast_scaling_diagnostic.png",
            rows,
            args.labels,
        )
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped plot generation.")

    print("Saved:")
    print(f"  {args.out_dir / 'contrast_scaling_diagnostic.csv'}")
    print(f"  {args.out_dir / 'contrast_scaling_diagnostic.png'}")


if __name__ == "__main__":
    main()
