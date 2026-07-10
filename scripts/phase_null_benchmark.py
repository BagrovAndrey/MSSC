from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.display import display_name, phase_name
from mssc.image_io import load_image, save_image
from mssc.shuffle import phase_scramble
from scripts.benchmark_toy_panel import (
    make_checkerboard,
    make_nested_dyadic,
    make_noise,
    make_patchwork,
    make_spectral_fractal_binary,
    make_stripes,
)
from scripts.diagnose_jlocq_outlier import (
    EPS,
    PROFILE_KEYS,
    TOTAL_KEYS,
    compute_profiles,
    make_offset_list,
    make_wavy_stripes,
    periodic_shift,
    profile_matrix_from_data,
    require_matplotlib,
    show_image,
    totals_vector_from_data,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    return int(value)


def parse_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MSSC-style diagnostics against a phase-scrambled null model."
    )
    parser.add_argument(
        "--preset",
        choices=["toy_plus_wavy"],
        default="toy_plus_wavy",
        help="Built-in generated image set. Default: toy_plus_wavy.",
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", default="auto")
    parser.add_argument("--mode", choices=["rgb", "grayscale"], default="grayscale")
    parser.add_argument("--value-range", choices=["0_1", "minus1_1"], default="minus1_1")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=4)
    parser.add_argument("--phase-null-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-panel-images", type=int, default=12)
    parser.add_argument(
        "--diagnostics-level",
        choices=["core", "full"],
        default="core",
        help="Default summary detail level. Default: core.",
    )
    parser.add_argument(
        "--profile-panel-labels",
        default="wavy_stripes,fractal,nested_dyadic,noise",
        help="Comma-separated labels for profile panel.",
    )
    parser.add_argument("--offset-average", action="store_true")
    parser.add_argument("--offset-mode", choices=["basic", "powers", "random"], default="powers")
    parser.add_argument("--max-power", type=int, default=None)
    parser.add_argument("--num-random-offsets", type=int, default=32)
    parser.add_argument("--offset-seed", type=int, default=0)
    return parser.parse_args()


def built_in_images(size: int, seed: int) -> dict[str, np.ndarray]:
    return {
        "stripes": make_stripes(size, period=16, orientation="vertical"),
        "checkerboard": make_checkerboard(size, cell_size=1),
        "patchwork": make_patchwork(size),
        "nested_dyadic": make_nested_dyadic(size),
        "fractal": make_spectral_fractal_binary(size, beta=2.5, seed=seed),
        "noise": make_noise(size, seed=seed),
        "wavy_stripes": make_wavy_stripes(
            size=size,
            stripe_period=64.0,
            wave_amplitude=24.0,
            wave_period=256.0,
            threshold=0.0,
            binary=True,
        ),
    }


def load_images_from_dir(
    image_dir: Path,
    size: int | str | None,
    mode: str,
    value_range: str,
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        label = path.stem
        images[label] = load_image(
            path,
            size=size,
            mode=mode,
            value_range=value_range,
        )
    return images


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_excess(excess: float, absolute: float) -> float:
    if abs(absolute) <= EPS:
        return float("nan")
    return excess / absolute


def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(values, axis=0), np.std(values, axis=0)


def compute_offset_phase_null(
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = make_offset_list(
        size=image.shape[0],
        mode=args.offset_mode,
        max_power=args.max_power,
        num_random_offsets=args.num_random_offsets,
        offset_seed=args.offset_seed,
    )

    original_values = []
    phase_values = []
    rows = []

    for dy, dx in offsets:
        shifted = periodic_shift(image, dy=dy, dx=dx)
        data = compute_profiles(shifted, n_steps=args.n_steps, connectivity=args.connectivity)
        vec = totals_vector_from_data(data)
        original_values.append(vec)
        row: dict[str, float | int | str] = {"label": "", "source": "", "kind": "original", "seed": -1, "dy": dy, "dx": dx}
        for idx, key in enumerate(TOTAL_KEYS):
            row[key] = float(vec[idx])
        rows.append(row)

    for seed in range(args.phase_null_seeds):
        scrambled = phase_scramble(image, seed=seed, preserve_dc=True)
        for dy, dx in offsets:
            shifted = periodic_shift(scrambled, dy=dy, dx=dx)
            data = compute_profiles(shifted, n_steps=args.n_steps, connectivity=args.connectivity)
            vec = totals_vector_from_data(data)
            phase_values.append(vec)
            row = {"label": "", "source": "", "kind": "phase", "seed": seed, "dy": dy, "dx": dx}
            for idx, key in enumerate(TOTAL_KEYS):
                row[key] = float(vec[idx])
            rows.append(row)

    return np.asarray(original_values, dtype=float), np.asarray(phase_values, dtype=float), np.array(offsets, dtype=int), np.asarray(rows, dtype=object)


def save_phase_null_panel(path: Path, display_rows: list[dict[str, object]]) -> None:
    plt = require_matplotlib()
    nrows = len(display_rows)
    fig, axes = plt.subplots(nrows, 5, figsize=(14, max(3, 2.8 * nrows)))
    if nrows == 1:
        axes = np.asarray([axes])

    for row_idx, item in enumerate(display_rows):
        label = item["label"]
        original = item["image"]
        phase_image = item["phase_image"]
        abs_vals = item["abs"]
        phase_mean = item["phase_mean"]

        show_image(axes[row_idx, 0], original)
        axes[row_idx, 0].set_title(str(label))

        show_image(axes[row_idx, 1], phase_image)
        axes[row_idx, 1].set_title("phase seed0")

        for col_idx, metric in enumerate(["Jglob", "Jloc", "JlocQ"], start=2):
            ax = axes[row_idx, col_idx]
            ax.bar([0, 1], [abs_vals[metric], phase_mean[metric]], color=["tab:blue", "tab:orange"])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["abs", "phase"])
            ax.set_title(display_name(metric))

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_phase_null_excess_panel(path: Path, labels: list[str], metric_rows: dict[str, list[float]], rel_rows: dict[str, list[float]]) -> None:
    plt = require_matplotlib()
    x = np.arange(len(labels))
    width = 0.25
    metrics = ["Jglob", "Jloc", "JlocQ"]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.8 * len(labels) + 4), 9), sharex=True)

    for idx, metric in enumerate(metrics):
        legend_name = phase_name(metric) if metric == "JlocQ" else f"{display_name(metric)} excess"
        axes[0].bar(x + (idx - 1) * width, metric_rows[metric], width=width, label=legend_name)
        axes[1].bar(x + (idx - 1) * width, rel_rows[metric], width=width, label=legend_name)

    axes[0].set_ylabel("excess")
    axes[0].set_title("Phase-null excess")
    axes[0].legend()

    axes[1].set_ylabel("relative excess")
    axes[1].set_title("Relative phase-null excess")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_profile_comparison_panel(path: Path, selected: list[dict[str, object]]) -> None:
    plt = require_matplotlib()
    nrows = len(selected)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, max(3, 3.2 * nrows)))
    if nrows == 1:
        axes = [axes]

    for ax, item in zip(axes, selected):
        label = item["label"]
        original = item["original_profile"]
        phase_mean = item["phase_mean_profile"]
        k = np.arange(len(original))
        ax.plot(k, original, marker="o", label=f"{display_name('JlocQ')} original")
        ax.plot(k, phase_mean, marker="s", label=f"{display_name('JlocQ')} phase mean")
        ax.plot(k, original - phase_mean, marker="^", label=phase_name("JlocQ"))
        ax.set_title(str(label))
        ax.set_xlabel("Scale index k")
        ax.set_ylabel(display_name("JlocQ"))
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.preset != "toy_plus_wavy":
        raise ValueError("only preset 'toy_plus_wavy' is currently supported")

    if args.size == "auto" or args.size == "none":
        generated_size = 512
    else:
        generated_size = int(args.size)

    generated = built_in_images(generated_size, seed=args.seed)
    loaded: dict[str, np.ndarray] = {}
    if args.image_dir is not None:
        loaded = load_images_from_dir(
            args.image_dir,
            size=parse_size(args.size),
            mode=args.mode,
            value_range=args.value_range,
        )

    all_items: list[tuple[str, str, np.ndarray]] = []
    all_items.extend((label, "generated", image) for label, image in generated.items())
    all_items.extend((label, "loaded", image) for label, image in loaded.items())

    summary_abs_rows: list[dict[str, float | str]] = []
    summary_phase_rows: list[dict[str, float | str]] = []
    profiles_original_rows: list[dict[str, float | str]] = []
    profiles_phase_mean_rows: list[dict[str, float | str]] = []
    profiles_phase_std_rows: list[dict[str, float | str]] = []
    phase_null_value_rows: list[dict[str, float | int | str]] = []
    summary_offset_phase_rows: list[dict[str, float | str]] = []
    offset_phase_value_rows: list[dict[str, float | int | str]] = []

    display_rows: list[dict[str, object]] = []
    profile_lookup: dict[str, dict[str, np.ndarray]] = {}

    for label, source, image in all_items:
        save_image(image, images_dir / f"{label}.png")

        original_data = compute_profiles(image, n_steps=args.n_steps, connectivity=args.connectivity)
        original_totals = totals_vector_from_data(original_data)
        original_profile = profile_matrix_from_data(original_data)

        summary_abs_rows.append(
            {
                "label": label,
                "source": source,
                "Cdetail_abs": float(original_totals[0]),
                "O_abs": float(original_totals[1]),
                "Odiv_abs": float(original_totals[2]),
                "Jglobal_abs": float(original_totals[3]),
                "Jloc_abs": float(original_totals[4]),
                "Jnested_abs": float(original_totals[5]),
            }
        )

        phase_totals = []
        phase_profiles = []
        representative_phase = None

        phase_null_value_rows.append(
            {
                "label": label,
                "source": source,
                "kind": "original",
                "seed": -1,
                "C": float(original_totals[0]),
                "O": float(original_totals[1]),
                "Odiv": float(original_totals[2]),
                "Jglob": float(original_totals[3]),
                "Jloc": float(original_totals[4]),
                "JlocQ": float(original_totals[5]),
            }
        )

        for seed in range(args.phase_null_seeds):
            scrambled = phase_scramble(image, seed=seed, preserve_dc=True)
            if representative_phase is None:
                representative_phase = scrambled
                save_image(scrambled, images_dir / f"{label}_phase_seed0.png")

            phase_data = compute_profiles(scrambled, n_steps=args.n_steps, connectivity=args.connectivity)
            phase_vec = totals_vector_from_data(phase_data)
            phase_mat = profile_matrix_from_data(phase_data)
            phase_totals.append(phase_vec)
            phase_profiles.append(phase_mat)

            row: dict[str, float | int | str] = {
                "label": label,
                "source": source,
                "kind": "phase",
                "seed": seed,
            }
            for idx, key in enumerate(TOTAL_KEYS):
                row[key] = float(phase_vec[idx])
            phase_null_value_rows.append(row)

        phase_totals_arr = np.asarray(phase_totals, dtype=float)
        phase_profiles_arr = np.asarray(phase_profiles, dtype=float)
        phase_mean, phase_std = mean_std(phase_totals_arr)
        profile_mean, profile_std = mean_std(phase_profiles_arr)

        for idx, metric in enumerate(TOTAL_KEYS):
            abs_val = float(original_totals[idx])
            mean_val = float(phase_mean[idx])
            std_val = float(phase_std[idx])
            excess = abs_val - mean_val
            excess_pos = max(excess, 0.0)
            excess_z = float("nan") if std_val <= EPS else excess / std_val
            summary_phase_rows.append(
                {
                    "label": label,
                    "source": source,
                    "metric": display_name(metric),
                    "abs": abs_val,
                    "phase_mean": mean_val,
                    "phase_std": std_val,
                    "excess": excess,
                    "excess_pos": excess_pos,
                    "excess_z": excess_z,
                    "relative_excess": relative_excess(excess, abs_val),
                    "Jphase": excess if metric == "JlocQ" else float("nan"),
                    "Jphase_pos": excess_pos if metric == "JlocQ" else float("nan"),
                    "Jphase_z": excess_z if metric == "JlocQ" else float("nan"),
                    "Jphase_relative": relative_excess(excess, abs_val) if metric == "JlocQ" else float("nan"),
                }
            )

        for k in range(original_profile.shape[0]):
            original_row: dict[str, float | str] = {"label": label, "source": source, "k": k}
            mean_row: dict[str, float | str] = {"label": label, "source": source, "k": k}
            std_row: dict[str, float | str] = {"label": label, "source": source, "k": k}
            for idx, key in enumerate(PROFILE_KEYS):
                original_row[key] = float(original_profile[k, idx])
                mean_row[key] = float(profile_mean[k, idx])
                std_row[key] = float(profile_std[k, idx])
            profiles_original_rows.append(original_row)
            profiles_phase_mean_rows.append(mean_row)
            profiles_phase_std_rows.append(std_row)

        profile_lookup[label] = {
            "original": original_profile[:, PROFILE_KEYS.index("JlocQ")],
            "phase_mean": profile_mean[:, PROFILE_KEYS.index("JlocQ")],
        }

        display_rows.append(
            {
                "label": label,
                "source": source,
                "image": image,
                "phase_image": representative_phase if representative_phase is not None else image,
                "abs": {
                    "Jglob": float(original_totals[3]),
                    "Jloc": float(original_totals[4]),
                    "JlocQ": float(original_totals[5]),
                },
                "phase_mean": {
                    "Jglob": float(phase_mean[3]),
                    "Jloc": float(phase_mean[4]),
                    "JlocQ": float(phase_mean[5]),
                },
            }
        )

        if args.offset_average:
            original_offset, phase_offset, offsets, _ = compute_offset_phase_null(image, args)
            original_offset_mean = np.mean(original_offset, axis=0)
            original_offset_std = np.std(original_offset, axis=0)
            phase_offset_mean = np.mean(phase_offset, axis=0)
            phase_offset_std = np.std(phase_offset, axis=0)

            for idx, metric in enumerate(TOTAL_KEYS):
                excess = float(original_offset_mean[idx] - phase_offset_mean[idx])
                excess_pos = max(excess, 0.0)
                z_val = float("nan") if phase_offset_std[idx] <= EPS else excess / float(phase_offset_std[idx])
                summary_offset_phase_rows.append(
                    {
                        "label": label,
                        "source": source,
                        "metric": metric,
                        "original_offset_mean": float(original_offset_mean[idx]),
                        "original_offset_std": float(original_offset_std[idx]),
                        "phase_offset_mean": float(phase_offset_mean[idx]),
                        "phase_offset_std": float(phase_offset_std[idx]),
                        "offset_phase_excess": excess,
                        "offset_phase_excess_pos": excess_pos,
                        "offset_phase_excess_z": z_val,
                    }
                )

            for offset_idx, (dy, dx) in enumerate(offsets):
                row: dict[str, float | int | str] = {
                    "label": label,
                    "source": source,
                    "kind": "original",
                    "seed": -1,
                    "dy": int(dy),
                    "dx": int(dx),
                }
                for idx, key in enumerate(TOTAL_KEYS):
                    row[key] = float(original_offset[offset_idx, idx])
                offset_phase_value_rows.append(row)

            n_offsets = len(offsets)
            for flat_idx in range(phase_offset.shape[0]):
                seed = flat_idx // n_offsets
                offset_idx = flat_idx % n_offsets
                dy, dx = offsets[offset_idx]
                row = {
                    "label": label,
                    "source": source,
                    "kind": "phase",
                    "seed": int(seed),
                    "dy": int(dy),
                    "dx": int(dx),
                }
                for idx, key in enumerate(TOTAL_KEYS):
                    row[key] = float(phase_offset[flat_idx, idx])
                offset_phase_value_rows.append(row)

    save_csv_rows(
        args.out_dir / "summary_abs.csv",
        ["label", "source", "Cdetail_abs", "O_abs", "Odiv_abs", "Jglobal_abs", "Jloc_abs", "Jnested_abs"],
        summary_abs_rows,
    )
    save_csv_rows(
        args.out_dir / "summary_phase_null.csv",
        ["label", "source", "metric", "abs", "phase_mean", "phase_std", "excess", "excess_pos", "excess_z", "relative_excess", "Jphase", "Jphase_pos", "Jphase_z", "Jphase_relative"],
        summary_phase_rows,
    )
    save_csv_rows(
        args.out_dir / "profiles_original.csv",
        ["label", "source", "k"] + PROFILE_KEYS,
        profiles_original_rows,
    )
    save_csv_rows(
        args.out_dir / "profiles_phase_mean.csv",
        ["label", "source", "k"] + PROFILE_KEYS,
        profiles_phase_mean_rows,
    )
    save_csv_rows(
        args.out_dir / "profiles_phase_std.csv",
        ["label", "source", "k"] + PROFILE_KEYS,
        profiles_phase_std_rows,
    )
    save_csv_rows(
        args.out_dir / "phase_null_values.csv",
        ["label", "source", "kind", "seed"] + TOTAL_KEYS,
        phase_null_value_rows,
    )

    if args.offset_average:
        save_csv_rows(
            args.out_dir / "summary_offset_phase_null.csv",
            ["label", "source", "metric", "original_offset_mean", "original_offset_std", "phase_offset_mean", "phase_offset_std", "offset_phase_excess", "offset_phase_excess_pos", "offset_phase_excess_z"],
            summary_offset_phase_rows,
        )
        save_csv_rows(
            args.out_dir / "offset_phase_null_values.csv",
            ["label", "source", "kind", "seed", "dy", "dx"] + TOTAL_KEYS,
            offset_phase_value_rows,
        )

    generated_labels = [label for label, source, _ in all_items if source == "generated"]
    loaded_labels = [label for label, source, _ in all_items if source == "loaded"]
    panel_labels = generated_labels + loaded_labels[: max(0, args.max_panel_images - len(generated_labels))]
    panel_rows = [item for item in display_rows if item["label"] in panel_labels]
    save_phase_null_panel(args.out_dir / "phase_null_panel.png", panel_rows)

    labels_for_excess = [row["label"] for row in summary_abs_rows]
    metric_rows = {"Jglob": [], "Jloc": [], "JlocQ": []}
    rel_rows = {"Jglob": [], "Jloc": [], "JlocQ": []}
    for label in labels_for_excess:
        for metric in ["Jglob", "Jloc", "JlocQ"]:
            row = next(r for r in summary_phase_rows if r["label"] == label and r["metric"] == display_name(metric))
            metric_rows[metric].append(float(row["excess"]))
            rel_rows[metric].append(float(row["relative_excess"]))
    save_phase_null_excess_panel(args.out_dir / "phase_null_excess_panel.png", labels_for_excess, metric_rows, rel_rows)

    selected_labels = [label for label in parse_csv_list(args.profile_panel_labels) if label in profile_lookup]
    if loaded_labels:
        first_loaded = loaded_labels[0]
        if first_loaded not in selected_labels:
            selected_labels.append(first_loaded)
    selected = [
        {
            "label": label,
            "original_profile": profile_lookup[label]["original"],
            "phase_mean_profile": profile_lookup[label]["phase_mean"],
        }
        for label in selected_labels
    ]
    if selected:
        save_profile_comparison_panel(args.out_dir / "profile_comparison_panel.png", selected)

    print("Phase-null benchmark summary")
    if args.diagnostics_level == "full":
        print("label                 Jglobal_abs Jglobal_exc Jloc_abs   Jloc_exc   Jnested_abs Jphase")
    else:
        print("label                 Jnested_abs phase_mean  Jphase      Jphase_rel  Jphase_z")
    for label in generated_labels:
        abs_row = next(r for r in summary_abs_rows if r["label"] == label)
        jglob = next(r for r in summary_phase_rows if r["label"] == label and r["metric"] == "Jglobal")
        jloc = next(r for r in summary_phase_rows if r["label"] == label and r["metric"] == "Jloc")
        jlocq = next(r for r in summary_phase_rows if r["label"] == label and r["metric"] == "Jnested")
        if args.diagnostics_level == "full":
            print(
                f"{label:<20s} "
                f"{float(abs_row['Jglobal_abs']):>11.4g} {float(jglob['excess']):>11.4g} "
                f"{float(abs_row['Jloc_abs']):>10.4g} {float(jloc['excess']):>10.4g} "
                f"{float(abs_row['Jnested_abs']):>11.4g} {float(jlocq['Jphase']):>10.4g}"
            )
        else:
            print(
                f"{label:<20s} "
                f"{float(abs_row['Jnested_abs']):>11.4g} {float(jlocq['phase_mean']):>10.4g} "
                f"{float(jlocq['Jphase']):>10.4g} {float(jlocq['Jphase_relative']):>11.4g} "
                f"{float(jlocq['Jphase_z']):>9.4g}"
            )


if __name__ == "__main__":
    main()
