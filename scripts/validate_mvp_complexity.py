from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mssc.complexity import complexity_profile, coarse_grain, max_steps
from mssc.image_io import load_image
from mssc.orientation import (
    detail_energy_coherence_from_map,
    detail_energy_map,
    heterogeneous_complexity_profile_from_weights,
    lifted_haar_channel_energy_profile,
    local_scale_orientation_entropy_profile_from_weights,
    structural_complexity_profile_from_weights,
)
from scripts.benchmark_toy_panel import (
    make_checkerboard,
    make_nested_dyadic,
    make_noise,
    make_patchwork,
    make_spectral_fractal_binary,
    make_stripes,
)
from scripts.diagnose_jlocq_outlier import EPS, make_wavy_stripes


CANONICAL_ORDER = [
    "stripes",
    "checkerboard",
    "patchwork",
    "nested_dyadic",
    "wavy_stripes",
    "fractal",
    "noise",
]


@dataclass
class ReplicateResult:
    label: str
    source: str
    replicate: int
    seed: int | None
    image: np.ndarray
    Cdetail: float
    Jstruct: float
    Jnested: float
    Jhetero: float
    Hstruct: float
    Hnested: float
    Ihetero: float
    retained_energy_fraction: float
    profile_rows: list[dict[str, float | int | str | bool]]


def require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def parse_size_arg(value: str) -> int | str:
    if value == "auto":
        return value

    size = int(value)
    if size <= 0:
        raise ValueError("--size must be positive")
    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the current energy-gated MVP structural complexity panel."
    )
    parser.add_argument(
        "--size",
        default="512",
        help="Synthetic image size as an integer, or 'auto' for optional natural-image inputs. Default: 512.",
    )
    parser.add_argument("--n-steps", type=int, default=None, help="Optional number of RG steps.")
    parser.add_argument("--noise-seeds", type=int, default=10, help="Number of noise realizations. Default: 10.")
    parser.add_argument(
        "--fractal-seeds",
        type=int,
        default=1,
        help="Number of fractal realizations. Use 1 for the canonical deterministic-looking panel. Default: 1.",
    )
    parser.add_argument(
        "--min-qenergy-pairs",
        type=int,
        default=32,
        help="Minimum number of native horizontal+vertical block-neighbor pairs required for Qenergy. Default: 32.",
    )
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional directory of extra natural grayscale images.")
    parser.add_argument(
        "--max-natural-panel",
        type=int,
        default=3,
        help="Maximum number of natural images to include in the main panel. Default: 3.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_image_builders(size: int) -> dict[str, callable]:
    return {
        "stripes": lambda seed: make_stripes(size, period=16, orientation="vertical"),
        "checkerboard": lambda seed: make_checkerboard(size, cell_size=1),
        "patchwork": lambda seed: make_patchwork(size),
        "nested_dyadic": lambda seed: make_nested_dyadic(size),
        "wavy_stripes": lambda seed: make_wavy_stripes(
            size=size,
            stripe_period=64.0,
            wave_amplitude=24.0,
            wave_period=256.0,
            threshold=0.0,
            binary=True,
        ),
        "fractal": lambda seed: make_spectral_fractal_binary(size, beta=2.5, seed=seed),
        "noise": lambda seed: make_noise(size, seed=seed),
    }


def collect_natural_images(image_dir: Path | None, size: int | str) -> list[tuple[str, np.ndarray]]:
    if image_dir is None:
        return []

    if not image_dir.exists():
        raise ValueError(f"--image-dir does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"--image-dir is not a directory: {image_dir}")

    rows: list[tuple[str, np.ndarray]] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            image = load_image(
                path,
                size=size,
                mode="grayscale",
                value_range="minus1_1",
            )
        except Exception:
            continue
        rows.append((path.stem, image))
    return rows


def native_neighbor_pair_count(grid_shape: tuple[int, int]) -> int:
    nrows, ncols = grid_shape
    horizontal = nrows * max(ncols - 1, 0)
    vertical = max(nrows - 1, 0) * ncols
    return int(horizontal + vertical)


def compute_qenergy_profile(
    image: np.ndarray,
    n_steps: int | None,
    min_qenergy_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    ebar = []
    qenergy = []
    variance = []
    defined = []
    pair_counts = []

    for _ in range(n_steps):
        e_map = detail_energy_map(current)
        pair_count = native_neighbor_pair_count(e_map.shape)
        q_val, var_val, raw_defined = detail_energy_coherence_from_map(e_map, eps=EPS)
        is_defined = bool(raw_defined and pair_count >= min_qenergy_pairs)

        ebar.append(float(np.mean(e_map)) if e_map.size else 0.0)
        qenergy.append(float(q_val) if is_defined else float("nan"))
        variance.append(float(var_val))
        defined.append(is_defined)
        pair_counts.append(pair_count)
        current = coarse_grain(current, block_size=2)

    return (
        np.asarray(ebar, dtype=float),
        np.asarray(qenergy, dtype=float),
        np.asarray(variance, dtype=float),
        np.asarray(defined, dtype=bool),
        np.asarray(pair_counts, dtype=int),
    )


def compute_weighted_tree(
    image: np.ndarray,
    n_steps: int | None,
    min_qenergy_pairs: int,
) -> tuple[dict[str, np.ndarray | float], list[dict[str, float | int | str | bool]]]:
    C = complexity_profile(image, block_size=2, n_steps=n_steps)
    lifted_E = lifted_haar_channel_energy_profile(image, n_steps=n_steps)
    ebar_k, qenergy_k, variance_k, defined_k, pair_count_k = compute_qenergy_profile(
        image,
        n_steps=n_steps,
        min_qenergy_pairs=min_qenergy_pairs,
    )

    qgate = np.zeros_like(qenergy_k, dtype=float)
    valid_q = defined_k & np.isfinite(qenergy_k)
    qgate[valid_q] = np.maximum(qenergy_k[valid_q], 0.0)
    W = lifted_E * qgate[:, None, None, None]

    Jnested_profile = local_scale_orientation_entropy_profile_from_weights(W)
    Jstruct_profile = structural_complexity_profile_from_weights(W)
    Jhetero_profile = heterogeneous_complexity_profile_from_weights(W)

    total_E = float(np.sum(lifted_E))
    total_W = float(np.sum(W))
    Wbar = float(np.mean(np.sum(W, axis=(0, 3))))

    Jstruct = float(np.sum(Jstruct_profile))
    Jnested = float(np.sum(Jnested_profile))
    Jhetero = float(np.sum(Jhetero_profile))

    if abs(Jstruct - Jnested - Jhetero) > 1e-11:
        raise ValueError(
            "Jstruct decomposition failed: "
            f"Jstruct={Jstruct:.12g}, Jnested={Jnested:.12g}, "
            f"Jhetero={Jhetero:.12g}"
        )

    retained = float("nan") if total_E <= EPS else total_W / total_E
    Hstruct = float("nan") if Wbar <= EPS else Jstruct / Wbar
    Hnested = float("nan") if Wbar <= EPS else Jnested / Wbar
    Ihetero = float("nan") if Wbar <= EPS else Jhetero / Wbar

    if Wbar > EPS and abs(Hstruct - Hnested - Ihetero) > 1e-11:
        raise ValueError(
            "Entropy decomposition failed: "
            f"Hstruct={Hstruct:.12g}, Hnested={Hnested:.12g}, Ihetero={Ihetero:.12g}"
        )

    Wbar_k = np.mean(np.sum(W, axis=-1), axis=(1, 2))
    profile_rows: list[dict[str, float | int | str | bool]] = []
    for k in range(len(ebar_k)):
        profile_rows.append(
            {
                "k": k,
                "Ebar_k": float(ebar_k[k]),
                "Qenergy_k": float(qenergy_k[k]),
                "Qenergy_defined": bool(defined_k[k]),
                "energy_variance_k": float(variance_k[k]),
                "neighbor_pair_count": int(pair_count_k[k]),
                "Wbar_k": float(Wbar_k[k]),
            }
        )

    return (
        {
            "Cdetail": float(np.sum(C)),
            "Jstruct": Jstruct,
            "Jnested": Jnested,
            "Jhetero": Jhetero,
            "Hstruct": Hstruct,
            "Hnested": Hnested,
            "Ihetero": Ihetero,
            "retained_energy_fraction": retained,
        },
        profile_rows,
    )


def compute_weighted_tree_profiles(
    image: np.ndarray,
    n_steps: int | None,
    min_qenergy_pairs: int,
) -> dict[str, np.ndarray | float]:
    C = complexity_profile(image, block_size=2, n_steps=n_steps)
    lifted_E = lifted_haar_channel_energy_profile(image, n_steps=n_steps)
    ebar_k, qenergy_k, variance_k, defined_k, pair_count_k = compute_qenergy_profile(
        image,
        n_steps=n_steps,
        min_qenergy_pairs=min_qenergy_pairs,
    )

    qgate = np.zeros_like(qenergy_k, dtype=float)
    valid_q = defined_k & np.isfinite(qenergy_k)
    qgate[valid_q] = np.maximum(qenergy_k[valid_q], 0.0)
    W = lifted_E * qgate[:, None, None, None]

    Jnested_profile = local_scale_orientation_entropy_profile_from_weights(W)
    Jstruct_profile = structural_complexity_profile_from_weights(W)
    Jhetero_profile = heterogeneous_complexity_profile_from_weights(W)

    total_E = float(np.sum(lifted_E))
    total_W = float(np.sum(W))
    Wbar = float(np.mean(np.sum(W, axis=(0, 3))))

    Jstruct = float(np.sum(Jstruct_profile))
    Jnested = float(np.sum(Jnested_profile))
    Jhetero = float(np.sum(Jhetero_profile))

    if abs(Jstruct - Jnested - Jhetero) > 1e-11:
        raise ValueError(
            "Jstruct decomposition failed: "
            f"Jstruct={Jstruct:.12g}, Jnested={Jnested:.12g}, "
            f"Jhetero={Jhetero:.12g}"
        )

    retained = float("nan") if total_E <= EPS else total_W / total_E
    Hstruct = float("nan") if Wbar <= EPS else Jstruct / Wbar
    Hnested = float("nan") if Wbar <= EPS else Jnested / Wbar
    Ihetero = float("nan") if Wbar <= EPS else Jhetero / Wbar

    if Wbar > EPS and abs(Hstruct - Hnested - Ihetero) > 1e-11:
        raise ValueError(
            "Entropy decomposition failed: "
            f"Hstruct={Hstruct:.12g}, Hnested={Hnested:.12g}, Ihetero={Ihetero:.12g}"
        )

    Wbar_k = np.mean(np.sum(W, axis=-1), axis=(1, 2))
    return {
        "C_profile": C,
        "Qenergy_profile": qenergy_k,
        "Qenergy_defined": defined_k,
        "energy_variance_profile": variance_k,
        "neighbor_pair_count_profile": pair_count_k,
        "Wbar_profile": Wbar_k,
        "Jnested_profile": Jnested_profile,
        "Jstruct_profile": Jstruct_profile,
        "Jhetero_profile": Jhetero_profile,
        "Cdetail": float(np.sum(C)),
        "Jstruct": Jstruct,
        "Jnested": Jnested,
        "Jhetero": Jhetero,
        "Hstruct": Hstruct,
        "Hnested": Hnested,
        "Ihetero": Ihetero,
        "retained_energy_fraction": retained,
    }


def analyze_replicate(
    label: str,
    source: str,
    replicate: int,
    seed: int | None,
    image: np.ndarray,
    n_steps: int | None,
    min_qenergy_pairs: int,
) -> ReplicateResult:
    summary = compute_weighted_tree_profiles(
        image=image,
        n_steps=n_steps,
        min_qenergy_pairs=min_qenergy_pairs,
    )
    profile_rows: list[dict[str, float | int | str | bool]] = []
    ebar_k, qenergy_k, variance_k, defined_k, pair_count_k = compute_qenergy_profile(
        image,
        n_steps=n_steps,
        min_qenergy_pairs=min_qenergy_pairs,
    )
    wbar_k = np.asarray(summary["Wbar_profile"], dtype=float)
    for k in range(len(ebar_k)):
        profile_rows.append(
            {
                "k": k,
                "Ebar_k": float(ebar_k[k]),
                "Qenergy_k": float(qenergy_k[k]),
                "Qenergy_defined": bool(defined_k[k]),
                "energy_variance_k": float(variance_k[k]),
                "neighbor_pair_count": int(pair_count_k[k]),
                "Wbar_k": float(wbar_k[k]),
            }
        )
    return ReplicateResult(
        label=label,
        source=source,
        replicate=replicate,
        seed=seed,
        image=np.asarray(image, dtype=float),
        Cdetail=float(summary["Cdetail"]),
        Jstruct=float(summary["Jstruct"]),
        Jnested=float(summary["Jnested"]),
        Jhetero=float(summary["Jhetero"]),
        Hstruct=float(summary["Hstruct"]),
        Hnested=float(summary["Hnested"]),
        Ihetero=float(summary["Ihetero"]),
        retained_energy_fraction=float(summary["retained_energy_fraction"]),
        profile_rows=profile_rows,
    )


def aggregate_results(
    label: str,
    source: str,
    replicates: list[ReplicateResult],
) -> dict[str, float | int | str]:
    def values(name: str) -> np.ndarray:
        return np.asarray([getattr(item, name) for item in replicates], dtype=float)

    return {
        "label": label,
        "source": source,
        "Cdetail": float(np.mean(values("Cdetail"))),
        "Jstruct": float(np.mean(values("Jstruct"))),
        "Jnested": float(np.mean(values("Jnested"))),
        "Jhetero": float(np.mean(values("Jhetero"))),
        "Hstruct": float(np.mean(values("Hstruct"))),
        "Hnested": float(np.mean(values("Hnested"))),
        "Ihetero": float(np.mean(values("Ihetero"))),
        "retained_energy_fraction": float(np.mean(values("retained_energy_fraction"))),
        "Jstruct_std": float(np.std(values("Jstruct"))),
        "Jnested_std": float(np.std(values("Jnested"))),
        "Jhetero_std": float(np.std(values("Jhetero"))),
        "retained_energy_fraction_std": float(np.std(values("retained_energy_fraction"))),
        "n_replicates": len(replicates),
    }


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_thumbnail_panel(
    path: Path,
    rows: list[dict[str, object]],
    thumbnail_lookup: dict[str, np.ndarray],
    title: str,
) -> None:
    plt = require_matplotlib()
    display_rows = [row for row in rows if str(row["label"]) in thumbnail_lookup]
    if not display_rows:
        return

    ncols = len(display_rows)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(2.4 * ncols, 8.0), 4.2),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]

    for ax, row in zip(axes, display_rows):
        label = str(row["label"])
        ax.imshow(thumbnail_lookup[label], cmap="gray", vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label.replace("_", " "), fontsize=12)

        total = float(row["Jstruct"])
        hetero = float(row["Jhetero"])
        nested = float(row["Jnested"])
        total_std = float(row.get("Jstruct_std", 0.0))

        if label == "noise" and total_std > 0.0:
            jstruct_text = f"Jstruct = {total:.3f} +/- {total_std:.3f}"
        else:
            jstruct_text = f"Jstruct = {total:.3f}"

        numbers = "\n".join(
            [
                jstruct_text,
                f"Jhetero = {hetero:.3f}",
                f"Jnested = {nested:.3f}",
            ]
        )
        ax.text(
            0.5,
            -0.18,
            numbers,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_gate_retention_summary(path: Path, rows: list[dict[str, object]], canonical_labels: list[str]) -> None:
    plt = require_matplotlib()
    subset = [row for row in rows if str(row["label"]) in canonical_labels]
    labels = [str(row["label"]).replace("_", " ") for row in subset]
    values = [float(row["retained_energy_fraction"]) for row in subset]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(labels)), values, color="#4c956c")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("retained_energy_fraction")
    ax.set_title("Energy gate retention")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_entropy_decomposition(path: Path, rows: list[dict[str, object]], canonical_labels: list[str]) -> None:
    plt = require_matplotlib()
    subset = [row for row in rows if str(row["label"]) in canonical_labels]
    labels = [str(row["label"]).replace("_", " ") for row in subset]
    hnested = np.asarray([float(row["Hnested"]) for row in subset], dtype=float)
    ihetero = np.asarray([float(row["Ihetero"]) for row in subset], dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, hnested, color="#2a6f97", label="Hnested")
    ax.bar(x, ihetero, bottom=hnested, color="#d98f4e", label="Ihetero")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Hstruct")
    ax.set_title("Normalized entropy decomposition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_qenergy_profiles(path: Path, profile_rows: list[dict[str, object]]) -> None:
    plt = require_matplotlib()
    labels = ["patchwork", "fractal", "wavy_stripes", "noise"]
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.8, 3.1 * len(labels)), sharex=True)

    for ax, label in zip(axes, labels):
        subset = [
            row for row in profile_rows
            if str(row["label"]) == label and int(row["replicate"]) == 0
        ]
        k = np.asarray([int(row["k"]) for row in subset], dtype=int)
        defined = np.asarray([bool(row["Qenergy_defined"]) for row in subset], dtype=bool)
        qenergy = np.asarray(
            [float(row["Qenergy_k"]) if bool(row["Qenergy_defined"]) else np.nan for row in subset],
            dtype=float,
        )
        ax.plot(k, qenergy, marker="o", color="#2a6f97")
        undefined_k = k[~defined]
        if undefined_k.size:
            ax.scatter(undefined_k, np.zeros_like(undefined_k, dtype=float), marker="x", s=50, color="#c1121f", label="undefined")
        ax.set_ylim(bottom=-0.05)
        ax.set_ylabel(label)
        if undefined_k.size:
            ax.legend(frameon=False, loc="upper right")

    axes[0].set_title("Qenergy profiles")
    axes[-1].set_xlabel("Scale index k")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_ensemble_stats(label: str, replicates: list[ReplicateResult]) -> None:
    if len(replicates) <= 1:
        return

    def stats(name: str) -> tuple[float, float, float, float]:
        values = np.asarray([getattr(item, name) for item in replicates], dtype=float)
        return (
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
        )

    print()
    print(f"{label} ensemble:")
    for metric in ("Jstruct", "Jnested", "Jhetero", "retained_energy_fraction"):
        mean, std, vmin, vmax = stats(metric)
        print(
            f"  {metric:<24s} "
            f"mean={mean:.6g} std={std:.6g} min={vmin:.6g} max={vmax:.6g}"
        )


def main() -> None:
    args = parse_args()
    size_arg = parse_size_arg(args.size)

    if args.noise_seeds <= 0:
        raise ValueError("--noise-seeds must be positive")
    if args.fractal_seeds <= 0:
        raise ValueError("--fractal-seeds must be positive")
    if args.min_qenergy_pairs < 0:
        raise ValueError("--min-qenergy-pairs must be nonnegative")

    synthetic_size = 512 if size_arg == "auto" else int(size_arg)
    if not is_power_of_two(synthetic_size):
        raise ValueError("Synthetic --size must be a power of two")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    builders = canonical_image_builders(synthetic_size)
    replicate_results: list[ReplicateResult] = []

    for label in CANONICAL_ORDER:
        if label == "noise":
            seeds = list(range(args.noise_seeds))
        elif label == "fractal":
            seeds = list(range(args.fractal_seeds))
        else:
            seeds = [0]

        for replicate, seed in enumerate(seeds):
            image = builders[label](seed)
            replicate_results.append(
                analyze_replicate(
                    label=label,
                    source="synthetic",
                    replicate=replicate,
                    seed=seed,
                    image=image,
                    n_steps=args.n_steps,
                    min_qenergy_pairs=args.min_qenergy_pairs,
                )
            )

    natural_images = collect_natural_images(args.image_dir, size=size_arg)
    for label, image in natural_images:
        replicate_results.append(
            analyze_replicate(
                label=label,
                source="natural",
                replicate=0,
                seed=None,
                image=image,
                n_steps=args.n_steps,
                min_qenergy_pairs=args.min_qenergy_pairs,
            )
        )

    grouped: dict[tuple[str, str], list[ReplicateResult]] = {}
    for result in replicate_results:
        grouped.setdefault((result.label, result.source), []).append(result)

    summary_rows: list[dict[str, object]] = []
    for label in CANONICAL_ORDER:
        summary_rows.append(aggregate_results(label, "synthetic", grouped[(label, "synthetic")]))
    for label, _image in natural_images:
        summary_rows.append(aggregate_results(label, "natural", grouped[(label, "natural")]))

    print_ensemble_stats("noise", grouped[("noise", "synthetic")])
    print_ensemble_stats("fractal", grouped[("fractal", "synthetic")])

    replicate_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    for result in replicate_results:
        replicate_rows.append(
            {
                "label": result.label,
                "replicate": result.replicate,
                "seed": "" if result.seed is None else result.seed,
                "Cdetail": result.Cdetail,
                "Jstruct": result.Jstruct,
                "Jnested": result.Jnested,
                "Jhetero": result.Jhetero,
                "Hstruct": result.Hstruct,
                "Hnested": result.Hnested,
                "Ihetero": result.Ihetero,
                "retained_energy_fraction": result.retained_energy_fraction,
            }
        )
        for row in result.profile_rows:
            profile_row = {
                "label": result.label,
                "replicate": result.replicate,
            }
            profile_row.update(row)
            profile_rows.append(profile_row)

    save_csv_rows(
        args.out_dir / "mvp_validation_summary.csv",
        [
            "label",
            "source",
            "Cdetail",
            "Jstruct",
            "Jnested",
            "Jhetero",
            "Hstruct",
            "Hnested",
            "Ihetero",
            "retained_energy_fraction",
            "Jstruct_std",
            "Jnested_std",
            "Jhetero_std",
            "retained_energy_fraction_std",
            "n_replicates",
        ],
        summary_rows,
    )
    save_csv_rows(
        args.out_dir / "mvp_validation_replicates.csv",
        [
            "label",
            "replicate",
            "seed",
            "Cdetail",
            "Jstruct",
            "Jnested",
            "Jhetero",
            "Hstruct",
            "Hnested",
            "Ihetero",
            "retained_energy_fraction",
        ],
        replicate_rows,
    )
    save_csv_rows(
        args.out_dir / "mvp_validation_profiles.csv",
        [
            "label",
            "replicate",
            "k",
            "Ebar_k",
            "Qenergy_k",
            "Qenergy_defined",
            "energy_variance_k",
            "neighbor_pair_count",
            "Wbar_k",
        ],
        profile_rows,
    )

    try:
        synthetic_thumbnails = {
            label: grouped[(label, "synthetic")][0].image
            for label in CANONICAL_ORDER
        }
        synthetic_rows = [row for row in summary_rows if str(row["label"]) in CANONICAL_ORDER]
        main_png = args.out_dir / "mvp_validation_panel.png"
        save_thumbnail_panel(
            main_png,
            synthetic_rows,
            synthetic_thumbnails,
            title="Synthetic MVP validation set",
        )
        save_thumbnail_panel(
            main_png.with_suffix(".pdf"),
            synthetic_rows,
            synthetic_thumbnails,
            title="Synthetic MVP validation set",
        )

        natural_panel_items = natural_images[: args.max_natural_panel]
        if natural_panel_items:
            natural_thumbnails = {
                label: grouped[(label, "natural")][0].image
                for label, _image in natural_panel_items
            }
            natural_rows = [
                row for row in summary_rows
                if str(row["source"]) == "natural" and str(row["label"]) in natural_thumbnails
            ]
            natural_png = args.out_dir / "natural_validation_panel.png"
            save_thumbnail_panel(
                natural_png,
                natural_rows,
                natural_thumbnails,
                title="Natural-image validation set",
            )
            save_thumbnail_panel(
                natural_png.with_suffix(".pdf"),
                natural_rows,
                natural_thumbnails,
                title="Natural-image validation set",
            )

        save_gate_retention_summary(
            args.out_dir / "gate_retention_summary.png",
            summary_rows,
            CANONICAL_ORDER,
        )
        save_entropy_decomposition(
            args.out_dir / "entropy_decomposition.png",
            summary_rows,
            CANONICAL_ORDER,
        )
        save_qenergy_profiles(
            args.out_dir / "qenergy_profiles.png",
            profile_rows,
        )
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped figure generation.")


if __name__ == "__main__":
    main()
