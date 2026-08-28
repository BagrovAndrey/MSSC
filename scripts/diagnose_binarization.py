from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.image_io import load_image
from scripts.validate_mvp_complexity import EPS, compute_weighted_tree, require_matplotlib


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
        description="Diagnose why binarization changes energy-gated structural complexity."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("ORIGINAL", "BINARY"),
        required=True,
        help="Original/binary image pair. Repeat this option for multiple pairs.",
    )
    parser.add_argument(
        "--size",
        default="auto",
        help="'auto', 'none', or INT. Passed through to the image loader. Default: auto.",
    )
    parser.add_argument("--n-steps", type=int, default=None, help="Optional number of RG steps.")
    parser.add_argument(
        "--min-qenergy-pairs",
        type=int,
        default=32,
        help="Minimum number of native neighbor pairs required for Qenergy. Default: 32.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_grayscale_image(path: Path, size: int | str | None) -> np.ndarray:
    return load_image(
        path,
        size=size,
        mode="grayscale",
        value_range="minus1_1",
    )


def summarize_variant(
    pair_name: str,
    variant: str,
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
    jstruct = float(totals["Jstruct"])
    jnested = float(totals["Jnested"])
    jhetero = float(totals["Jhetero"])
    hstruct = float(totals["Hstruct"])
    hnested = float(totals["Hnested"])
    ihetero = float(totals["Ihetero"])
    retained = float(totals["retained_energy_fraction"])

    # Under the current lifting convention, Ebar is the mean total lifted
    # Haar-channel energy per original-space pixel, which should track Cdetail.
    ebar = cdetail
    wbar = ebar * retained if np.isfinite(retained) else float("nan")
    jspecific = float("nan") if ebar <= EPS else jstruct / ebar

    consistency_j = float("nan")
    if np.isfinite(ebar) and np.isfinite(retained) and np.isfinite(hstruct):
        consistency_j = jstruct - ebar * retained * hstruct

    return {
        "pair": pair_name,
        "variant": variant,
        "Cdetail": cdetail,
        "Ebar": ebar,
        "Wbar": wbar,
        "R": retained,
        "Hstruct": hstruct,
        "Hnested": hnested,
        "Ihetero": ihetero,
        "Jstruct": jstruct,
        "Jnested": jnested,
        "Jhetero": jhetero,
        "Jspecific": jspecific,
        "J_consistency_error": consistency_j,
    }


def ratio_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return float("nan")
    return float(numerator / denominator)


def diff_or_nan(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before):
        return float("nan")
    return float(after - before)


def build_ratio_row(
    pair_name: str,
    original: dict[str, float | str],
    binary: dict[str, float | str],
) -> dict[str, float | str]:
    c_orig = float(original["Cdetail"])
    c_bin = float(binary["Cdetail"])
    e_orig = float(original["Ebar"])
    e_bin = float(binary["Ebar"])
    r_orig = float(original["R"])
    r_bin = float(binary["R"])
    h_orig = float(original["Hstruct"])
    h_bin = float(binary["Hstruct"])
    j_orig = float(original["Jstruct"])
    j_bin = float(binary["Jstruct"])
    js_orig = float(original["Jspecific"])
    js_bin = float(binary["Jspecific"])

    return {
        "pair": pair_name,
        "Cdetail_ratio": ratio_or_nan(c_bin, c_orig),
        "Ebar_ratio": ratio_or_nan(e_bin, e_orig),
        "R_ratio": ratio_or_nan(r_bin, r_orig),
        "Hstruct_ratio": ratio_or_nan(h_bin, h_orig),
        "Jspecific_ratio": ratio_or_nan(js_bin, js_orig),
        "Jstruct_ratio": ratio_or_nan(j_bin, j_orig),
        "Jnested_ratio": ratio_or_nan(float(binary["Jnested"]), float(original["Jnested"])),
        "Jhetero_ratio": ratio_or_nan(float(binary["Jhetero"]), float(original["Jhetero"])),
        "Cdetail_delta": diff_or_nan(c_bin, c_orig),
        "R_delta": diff_or_nan(r_bin, r_orig),
        "Hstruct_delta": diff_or_nan(h_bin, h_orig),
        "Jspecific_delta": diff_or_nan(js_bin, js_orig),
        "Jstruct_delta": diff_or_nan(j_bin, j_orig),
        "Jfactor_from_Cdetail_R_Hstruct": ratio_or_nan(
            ratio_or_nan(c_bin, c_orig) * ratio_or_nan(r_bin, r_orig) * ratio_or_nan(h_bin, h_orig),
            1.0,
        ),
    }


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_panel(
    path: Path,
    pair_rows: list[tuple[str, dict[str, float | str], dict[str, float | str], np.ndarray, np.ndarray]],
) -> None:
    plt = require_matplotlib()
    nrows = len(pair_rows)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.0, max(3.2 * nrows, 4.0)),
        constrained_layout=True,
    )

    if nrows == 1:
        axes = np.asarray([axes], dtype=object)

    for row_index, (pair_name, original, binary, original_image, binary_image) in enumerate(pair_rows):
        for col_index, (variant_name, image, summary) in enumerate(
            (
                ("original", original_image, original),
                ("binary", binary_image, binary),
            )
        ):
            ax = axes[row_index, col_index]
            ax.imshow(image, cmap="gray", vmin=-1.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{pair_name}: {variant_name}", fontsize=11)
            ax.text(
                0.5,
                -0.18,
                "\n".join(
                    [
                        f"Cdetail = {float(summary['Cdetail']):.3f}",
                        f"R = {float(summary['R']):.3f}",
                        f"Hstruct = {float(summary['Hstruct']):.3f}",
                        f"Jstruct = {float(summary['Jstruct']):.3f}",
                        f"Jspecific = {float(summary['Jspecific']):.3f}",
                    ]
                ),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

    fig.suptitle("Binarization diagnostic", fontsize=14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    size = parse_size(args.size)

    if args.min_qenergy_pairs < 0:
        raise ValueError("--min-qenergy-pairs must be nonnegative")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    ratio_rows: list[dict[str, object]] = []
    panel_rows: list[tuple[str, dict[str, float | str], dict[str, float | str], np.ndarray, np.ndarray]] = []

    for original_path_str, binary_path_str in args.pair:
        original_path = Path(original_path_str)
        binary_path = Path(binary_path_str)
        pair_name = original_path.stem
        if pair_name.endswith("_binary"):
            pair_name = pair_name[: -len("_binary")]

        original_image = load_grayscale_image(original_path, size=size)
        binary_image = load_grayscale_image(binary_path, size=size)

        original = summarize_variant(
            pair_name=pair_name,
            variant="original",
            image=original_image,
            n_steps=args.n_steps,
            min_qenergy_pairs=args.min_qenergy_pairs,
        )
        binary = summarize_variant(
            pair_name=pair_name,
            variant="binary",
            image=binary_image,
            n_steps=args.n_steps,
            min_qenergy_pairs=args.min_qenergy_pairs,
        )

        summary_rows.extend([original, binary])
        ratio_rows.append(build_ratio_row(pair_name, original, binary))
        panel_rows.append((pair_name, original, binary, original_image, binary_image))

    save_csv_rows(
        args.out_dir / "binarization_summary.csv",
        [
            "pair",
            "variant",
            "Cdetail",
            "Ebar",
            "Wbar",
            "R",
            "Hstruct",
            "Hnested",
            "Ihetero",
            "Jstruct",
            "Jnested",
            "Jhetero",
            "Jspecific",
            "J_consistency_error",
        ],
        summary_rows,
    )
    save_csv_rows(
        args.out_dir / "binarization_ratios.csv",
        [
            "pair",
            "Cdetail_ratio",
            "Ebar_ratio",
            "R_ratio",
            "Hstruct_ratio",
            "Jspecific_ratio",
            "Jstruct_ratio",
            "Jnested_ratio",
            "Jhetero_ratio",
            "Cdetail_delta",
            "R_delta",
            "Hstruct_delta",
            "Jspecific_delta",
            "Jstruct_delta",
            "Jfactor_from_Cdetail_R_Hstruct",
        ],
        ratio_rows,
    )

    try:
        save_panel(args.out_dir / "binarization_panel.png", panel_rows)
        save_panel(args.out_dir / "binarization_panel.pdf", panel_rows)
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped figure generation.")

    print("Saved:")
    print(f"  {args.out_dir / 'binarization_summary.csv'}")
    print(f"  {args.out_dir / 'binarization_ratios.csv'}")
    print(f"  {args.out_dir / 'binarization_panel.png'}")


if __name__ == "__main__":
    main()
