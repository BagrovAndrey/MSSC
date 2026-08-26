from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.complexity import complexity_profile
from mssc.display import display_name, phase_name
from mssc.image_io import save_image
from mssc.orientation import (
    heterogeneous_complexity_profile_from_weights,
    lifted_haar_channel_energy_profile,
    lifted_local_orientation_coherence_profile,
    local_scale_orientation_entropy_profile_from_weights,
    local_scale_orientation_weights,
    structural_complexity_profile_from_weights,
)
from mssc.shuffle import phase_scramble
from scripts.benchmark_toy_panel import (
    make_checkerboard,
    make_nested_dyadic,
    make_noise,
    make_patchwork,
    make_spectral_fractal_binary,
    make_stripes,
)
from scripts.diagnose_jlocq_outlier import EPS, make_wavy_stripes


def require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the structural complexity tree decomposition on canonical toy patterns."
    )
    parser.add_argument("--size", type=int, default=512, help="Image size. Must be a power of two.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for stochastic generators.")
    parser.add_argument("--phase-null-seeds", type=int, default=20, help="Number of phase-scrambled surrogates.")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional number of RG steps.")
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=4, help="Connectivity for local q maps.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--save-images", action="store_true", help="Save PNG exports of the generated arrays.")
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


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_tree_profiles(
    image: np.ndarray,
    n_steps: int | None,
    connectivity: int,
) -> dict[str, np.ndarray | float]:
    C = complexity_profile(image, block_size=2, n_steps=n_steps)
    lifted_E = lifted_haar_channel_energy_profile(image, n_steps=n_steps)
    lifted_q = lifted_local_orientation_coherence_profile(
        image,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    W = local_scale_orientation_weights(lifted_E, lifted_q)

    Jnested_profile = local_scale_orientation_entropy_profile_from_weights(W)
    Jstruct_profile = structural_complexity_profile_from_weights(W)
    Jhetero_profile = heterogeneous_complexity_profile_from_weights(W)

    Jnested = float(np.sum(Jnested_profile))
    Jstruct = float(np.sum(Jstruct_profile))
    Jhetero = float(np.sum(Jhetero_profile))

    decomposition_error = Jstruct - Jnested - Jhetero
    if decomposition_error < -1e-12 or decomposition_error > 1e-12:
        raise ValueError(
            "Jstruct decomposition failed: "
            f"Jstruct={Jstruct:.12g}, Jnested={Jnested:.12g}, "
            f"Jhetero={Jhetero:.12g}, diff={decomposition_error:.12g}"
        )

    return {
        "C": C,
        "Jnested": Jnested_profile,
        "Jstruct": Jstruct_profile,
        "Jhetero": Jhetero_profile,
        "C_total": float(np.sum(C)),
        "Jnested_total": Jnested,
        "Jstruct_total": Jstruct,
        "Jhetero_total": Jhetero,
        "decomposition_error": decomposition_error,
    }


def compute_phase_nested_stats(
    image: np.ndarray,
    n_steps: int | None,
    connectivity: int,
    n_seeds: int,
) -> dict[str, np.ndarray | float]:
    phase_profiles = []

    for seed in range(n_seeds):
        scrambled = phase_scramble(image, seed=seed, preserve_dc=True)
        lifted_E = lifted_haar_channel_energy_profile(scrambled, n_steps=n_steps)
        lifted_q = lifted_local_orientation_coherence_profile(
            scrambled,
            n_steps=n_steps,
            connectivity=connectivity,
        )
        W = local_scale_orientation_weights(lifted_E, lifted_q)
        phase_profiles.append(local_scale_orientation_entropy_profile_from_weights(W))

    phase_profiles_arr = np.asarray(phase_profiles, dtype=float)
    phase_mean_profile = np.mean(phase_profiles_arr, axis=0)
    phase_std_profile = np.std(phase_profiles_arr, axis=0)

    Jspectral = float(np.sum(phase_mean_profile))
    phase_std = float(np.std(np.sum(phase_profiles_arr, axis=1)))

    return {
        "Jspectral_profile": phase_mean_profile,
        "phase_profile_std": phase_std_profile,
        "Jspectral_total": Jspectral,
        "phase_std": phase_std,
    }


def save_complexity_tree_bars(path: Path, summary_rows: list[dict[str, float | str]]) -> None:
    plt = require_matplotlib()
    labels = [str(row["label"]) for row in summary_rows]
    nested = np.asarray([float(row["Jnested"]) for row in summary_rows], dtype=float)
    hetero = np.asarray([float(row["Jhetero"]) for row in summary_rows], dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(labels)), 6))
    ax.bar(x, nested, label=display_name("JlocQ"))
    ax.bar(x, hetero, bottom=nested, label=display_name("Jhetero"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(display_name("Jstruct"))
    ax.set_title("Structural complexity decomposition")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_nested_phase_decomposition(path: Path, summary_rows: list[dict[str, float | str]]) -> None:
    plt = require_matplotlib()
    labels = [str(row["label"]) for row in summary_rows]
    nested = np.asarray([float(row["Jnested"]) for row in summary_rows], dtype=float)
    spectral = np.asarray([float(row["Jspectral"]) for row in summary_rows], dtype=float)
    phase = np.asarray([float(row["Jphase"]) for row in summary_rows], dtype=float)
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.2 * len(labels)), 9), sharex=True)
    axes[0].bar(x - width / 2, nested, width=width, label=display_name("JlocQ"))
    axes[0].bar(x + width / 2, spectral, width=width, label=display_name("Jspectral"))
    axes[0].set_ylabel("nested total")
    axes[0].set_title("Nested branch and phase-null spectral baseline")
    axes[0].legend()

    colors = ["tab:red" if value < 0 else "tab:green" for value in phase]
    axes[1].bar(x, phase, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel(phase_name("JlocQ"))
    axes[1].set_title("Phase-specific nested correction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_nested_vs_heterogeneous(path: Path, summary_rows: list[dict[str, float | str]]) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in summary_rows:
        x = float(row["Jnested"])
        y = float(row["Jhetero"])
        label = str(row["label"])
        ax.scatter([x], [y], s=60)
        ax.text(x, y, f" {label}", va="center")

    ax.set_xlabel(display_name("JlocQ"))
    ax.set_ylabel(display_name("Jhetero"))
    ax.set_title("Nested vs heterogeneous complexity")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not is_power_of_two(args.size):
        raise ValueError("--size must be a power of two")
    if args.phase_null_seeds <= 0:
        raise ValueError("--phase-null-seeds must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    images = built_in_images(args.size, args.seed)
    summary_rows: list[dict[str, float | str]] = []
    profile_rows: list[dict[str, float | str]] = []

    for label, image in images.items():
        if args.save_images:
            save_image(image, args.out_dir / f"{label}.png")

        tree = compute_tree_profiles(
            image=image,
            n_steps=args.n_steps,
            connectivity=args.connectivity,
        )
        phase = compute_phase_nested_stats(
            image=image,
            n_steps=args.n_steps,
            connectivity=args.connectivity,
            n_seeds=args.phase_null_seeds,
        )

        Jnested = float(tree["Jnested_total"])
        Jstruct = float(tree["Jstruct_total"])
        Jhetero = float(tree["Jhetero_total"])
        Jspectral = float(phase["Jspectral_total"])
        Jphase = Jnested - Jspectral
        phase_std = float(phase["phase_std"])
        phase_z = float("nan") if phase_std <= EPS else Jphase / phase_std
        phase_decomposition_error = Jnested - Jspectral - Jphase

        if phase_decomposition_error < -1e-12 or phase_decomposition_error > 1e-12:
            raise ValueError(
                "Nested phase decomposition failed: "
                f"Jnested={Jnested:.12g}, Jspectral={Jspectral:.12g}, "
                f"Jphase={Jphase:.12g}, diff={phase_decomposition_error:.12g}"
            )

        summary_rows.append(
            {
                "label": label,
                "Cdetail": float(tree["C_total"]),
                "Jstruct": Jstruct,
                "Jnested": Jnested,
                "Jhetero": Jhetero,
                "Jspectral": Jspectral,
                "Jphase": Jphase,
                "phase_std": phase_std,
                "phase_z": phase_z,
                "nested_fraction": float("nan") if Jstruct <= EPS else Jnested / Jstruct,
                "hetero_fraction": float("nan") if Jstruct <= EPS else Jhetero / Jstruct,
                "Jphase_relative": float("nan") if Jnested <= EPS else Jphase / Jnested,
                "decomposition_error": float(tree["decomposition_error"]),
                "phase_decomposition_error": phase_decomposition_error,
            }
        )

        Jnested_profile = np.asarray(tree["Jnested"], dtype=float)
        Jstruct_profile = np.asarray(tree["Jstruct"], dtype=float)
        Jhetero_profile = np.asarray(tree["Jhetero"], dtype=float)
        Jspectral_profile = np.asarray(phase["Jspectral_profile"], dtype=float)
        phase_std_profile = np.asarray(phase["phase_profile_std"], dtype=float)
        Jphase_profile = Jnested_profile - Jspectral_profile

        for k in range(len(Jnested_profile)):
            profile_rows.append(
                {
                    "label": label,
                    "k": k,
                    "Cdetail": float(np.asarray(tree["C"], dtype=float)[k]),
                    "Jstruct": float(Jstruct_profile[k]),
                    "Jnested": float(Jnested_profile[k]),
                    "Jhetero": float(Jhetero_profile[k]),
                    "Jspectral": float(Jspectral_profile[k]),
                    "Jphase": float(Jphase_profile[k]),
                    "phase_std": float(phase_std_profile[k]),
                    "decomposition_error": float(Jstruct_profile[k] - Jnested_profile[k] - Jhetero_profile[k]),
                    "phase_decomposition_error": float(Jnested_profile[k] - Jspectral_profile[k] - Jphase_profile[k]),
                }
            )

    save_csv_rows(
        args.out_dir / "complexity_tree_summary.csv",
        [
            "label",
            "Cdetail",
            "Jstruct",
            "Jnested",
            "Jhetero",
            "Jspectral",
            "Jphase",
            "phase_std",
            "phase_z",
            "nested_fraction",
            "hetero_fraction",
            "Jphase_relative",
            "decomposition_error",
            "phase_decomposition_error",
        ],
        summary_rows,
    )
    save_csv_rows(
        args.out_dir / "complexity_tree_profiles.csv",
        [
            "label",
            "k",
            "Cdetail",
            "Jstruct",
            "Jnested",
            "Jhetero",
            "Jspectral",
            "Jphase",
            "phase_std",
            "decomposition_error",
            "phase_decomposition_error",
        ],
        profile_rows,
    )

    try:
        save_complexity_tree_bars(args.out_dir / "complexity_tree_bars.png", summary_rows)
        save_nested_phase_decomposition(args.out_dir / "nested_phase_decomposition.png", summary_rows)
        save_nested_vs_heterogeneous(args.out_dir / "nested_vs_heterogeneous.png", summary_rows)
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped plot generation.")

    print("Complexity tree summary")
    print("label                 Cdetail    Jstruct    Jnested    Jhetero    Jspectral Jphase     phase_z")
    for row in summary_rows:
        print(
            f"{str(row['label']):<20s} "
            f"{float(row['Cdetail']):>10.4g} "
            f"{float(row['Jstruct']):>10.4g} "
            f"{float(row['Jnested']):>10.4g} "
            f"{float(row['Jhetero']):>10.4g} "
            f"{float(row['Jspectral']):>10.4g} "
            f"{float(row['Jphase']):>10.4g} "
            f"{float(row['phase_z']):>10.4g}"
        )


if __name__ == "__main__":
    main()
