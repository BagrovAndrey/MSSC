from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.complexity import complexity_profile
from mssc.display import display_name
from mssc.image_io import save_image
from mssc.orientation import (
    detail_energy_coherence_profile,
    heterogeneous_complexity_profile_from_weights,
    lifted_haar_channel_energy_profile,
    lifted_local_orientation_coherence_profile,
    local_orientation_coherence_profile,
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


def require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def parse_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose q-weighting and multiscale support using the current MSSC local weight construction."
    )
    parser.add_argument("--size", type=int, default=512, help="Image size. Must be a power of two.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for stochastic generators.")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional number of RG steps.")
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=4, help="Connectivity for local q maps.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        default="patchwork,fractal,wavy_stripes,noise,checkerboard",
        help="Comma-separated image labels to analyze.",
    )
    parser.add_argument("--eps-abs", type=float, default=1e-15, help="Absolute threshold for active support maps.")
    parser.add_argument("--eps-rel", type=float, default=1e-6, help="Relative threshold for active support maps.")
    parser.add_argument("--save-images", action="store_true", help="Save PNG exports of the generated arrays.")
    return parser.parse_args()


def built_in_images(size: int, seed: int) -> dict[str, np.ndarray]:
    return {
        "patchwork": make_patchwork(size),
        "fractal": make_spectral_fractal_binary(size, beta=2.5, seed=seed),
        "wavy_stripes": make_wavy_stripes(
            size=size,
            stripe_period=64.0,
            wave_amplitude=24.0,
            wave_period=256.0,
            threshold=0.0,
            binary=True,
        ),
        "noise": make_noise(size, seed=seed),
        "checkerboard": make_checkerboard(size, cell_size=1),
        "nested_dyadic": make_nested_dyadic(size),
        "straight_stripes": make_stripes(size, period=16, orientation="vertical"),
    }


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_weight_tensor(lifted_E: np.ndarray, lifted_q: np.ndarray, weighting: str) -> np.ndarray:
    qpos = np.maximum(lifted_q, 0.0)

    if weighting == "q0":
        factor = np.ones_like(qpos)
    elif weighting == "qsqrt":
        factor = np.sqrt(qpos)
    elif weighting == "q1":
        factor = qpos
    else:
        raise ValueError(f"unknown weighting: {weighting}")

    return lifted_E * factor[..., None]


def make_gate_weight_tensor(
    lifted_E: np.ndarray,
    lifted_q: np.ndarray,
    qenergy_profile: np.ndarray,
    qenergy_defined: np.ndarray,
    gate: str,
) -> np.ndarray:
    if gate == "no_gate":
        return lifted_E

    if gate == "energy_gate":
        gate_profile = np.zeros_like(qenergy_profile, dtype=float)
        valid = qenergy_defined & np.isfinite(qenergy_profile)
        gate_profile[valid] = np.maximum(qenergy_profile[valid], 0.0)
        return lifted_E * gate_profile[:, None, None, None]

    if gate == "orientation_gate":
        return lifted_E * np.maximum(lifted_q, 0.0)[..., None]

    raise ValueError(f"unknown gate: {gate}")


def decompose_weights(
    label: str,
    weighting: str,
    weights: np.ndarray,
    total_cdetail: float,
    lifted_E: np.ndarray,
    eps: float = EPS,
) -> dict[str, float | str]:
    Jstruct_profile = structural_complexity_profile_from_weights(weights)
    Jnested_profile = local_scale_orientation_entropy_profile_from_weights(weights)
    Jhetero_profile = heterogeneous_complexity_profile_from_weights(weights)

    Jstruct = float(np.sum(Jstruct_profile))
    Jnested = float(np.sum(Jnested_profile))
    Jhetero = float(np.sum(Jhetero_profile))

    Wbar = float(np.mean(np.sum(weights, axis=(0, 3))))
    Ebar = float(np.mean(np.sum(lifted_E, axis=(0, 3))))
    total_E = float(np.sum(lifted_E))
    total_W = float(np.sum(weights))
    qE_fraction = float("nan") if total_E <= eps else total_W / total_E

    Hstruct = float("nan") if Wbar <= eps else Jstruct / Wbar
    Hnested = float("nan") if Wbar <= eps else Jnested / Wbar
    Ihetero = float("nan") if Wbar <= eps else Jhetero / Wbar

    if Wbar > eps:
        h_error = Hstruct - Hnested - Ihetero
        j_error = Jstruct - Jnested - Jhetero
        if abs(h_error) > 1e-11:
            raise ValueError(
                "Entropy decomposition failed: "
                f"label={label}, weighting={weighting}, Hstruct={Hstruct:.12g}, "
                f"Hnested={Hnested:.12g}, Ihetero={Ihetero:.12g}, diff={h_error:.12g}"
            )
        if abs(j_error) > 1e-11:
            raise ValueError(
                "Weight decomposition failed: "
                f"label={label}, weighting={weighting}, Jstruct={Jstruct:.12g}, "
                f"Jnested={Jnested:.12g}, Jhetero={Jhetero:.12g}, diff={j_error:.12g}"
            )

    return {
        "label": label,
        "weighting": weighting,
        "gate": weighting,
        "Cdetail": total_cdetail,
        "Ebar": Ebar,
        "Wbar": Wbar,
        "qE_fraction": qE_fraction,
        "retained_energy_fraction": qE_fraction,
        "Hstruct": Hstruct,
        "Hnested": Hnested,
        "Ihetero": Ihetero,
        "Jstruct": Jstruct,
        "Jnested": Jnested,
        "Jhetero": Jhetero,
    }


def active_fraction_map(value: np.ndarray, eps_abs: float, eps_rel: float) -> np.ndarray:
    vmax = float(np.max(value))
    threshold = max(eps_abs, eps_rel * vmax)
    return value > threshold


def effective_support_fraction(value: np.ndarray, eps: float = EPS) -> float:
    total = float(np.sum(value))
    sq_total = float(np.sum(value * value))
    n_pix = value.shape[0] * value.shape[1]

    if sq_total <= eps or n_pix <= 0:
        return 0.0

    return (total * total) / (n_pix * sq_total)


def summarize_active_history(counts: np.ndarray, prefix: str) -> dict[str, float]:
    flat = counts.ravel().astype(float)
    return {
        f"mean_{prefix}": float(np.mean(flat)),
        f"median_{prefix}": float(np.median(flat)),
        f"std_{prefix}": float(np.std(flat)),
        f"max_{prefix}": float(np.max(flat)),
        f"p25_{prefix}": float(np.percentile(flat, 25)),
        f"p75_{prefix}": float(np.percentile(flat, 75)),
        f"p90_{prefix}": float(np.percentile(flat, 90)),
    }


def save_count_map(path: Path, counts: np.ndarray, vmin: float, vmax: float) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(counts, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_active_history_plot(path: Path, histogram_rows: list[dict[str, float | str]]) -> None:
    plt = require_matplotlib()
    labels = ["fractal", "wavy_stripes"]
    fig, ax = plt.subplots(figsize=(8, 5))

    max_bin = 0
    for row in histogram_rows:
        if row["label"] in labels:
            max_bin = max(max_bin, int(row["n_active_scales"]))

    x = np.arange(max_bin + 1)
    width = 0.38

    for idx, label in enumerate(labels):
        subset = [row for row in histogram_rows if row["label"] == label]
        fractions = np.zeros(max_bin + 1, dtype=float)
        for row in subset:
            fractions[int(row["n_active_scales"])] = float(row["pixel_fraction"])
        ax.bar(x + (idx - 0.5) * width, fractions, width=width, label=label)

    ax.set_xlabel("n_active_scales")
    ax.set_ylabel("pixel_fraction")
    ax.set_title("Active RG-history length from lifted E")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_q_ablation_plot(path: Path, rows: list[dict[str, float | str]], labels: list[str]) -> None:
    plt = require_matplotlib()
    weightings = ["q0", "qsqrt", "q1"]
    x = np.arange(len(labels))
    width = 0.24

    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.1 * len(labels)), 9), sharex=True)

    for idx, weighting in enumerate(weightings):
        subset = [row for row in rows if row["weighting"] == weighting]
        jnested = [float(next(row["Jnested"] for row in subset if row["label"] == label)) for label in labels]
        jstruct = [float(next(row["Jstruct"] for row in subset if row["label"] == label)) for label in labels]
        axes[0].bar(x + (idx - 1) * width, jnested, width=width, label=weighting)
        axes[1].bar(x + (idx - 1) * width, jstruct, width=width, label=weighting)

    axes[0].set_ylabel(display_name("JlocQ"))
    axes[0].set_title("q-weighting ablation: Jnested")
    axes[0].legend()
    axes[1].set_ylabel(display_name("Jstruct"))
    axes[1].set_title("q-weighting ablation: Jstruct")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_energy_gate_ablation_plot(path: Path, rows: list[dict[str, float | str]], labels: list[str]) -> None:
    plt = require_matplotlib()
    gates = ["no_gate", "energy_gate", "orientation_gate"]
    x = np.arange(len(labels))
    width = 0.24

    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.1 * len(labels)), 9), sharex=True)

    for idx, gate in enumerate(gates):
        subset = [row for row in rows if row["gate"] == gate]
        jstruct = [float(next(row["Jstruct"] for row in subset if row["label"] == label)) for label in labels]
        jnested = [float(next(row["Jnested"] for row in subset if row["label"] == label)) for label in labels]
        axes[0].bar(x + (idx - 1) * width, jstruct, width=width, label=gate)
        axes[1].bar(x + (idx - 1) * width, jnested, width=width, label=gate)

    axes[0].set_ylabel(display_name("Jstruct"))
    axes[0].set_title("Organization-gate ablation: Jstruct")
    axes[0].legend()
    axes[1].set_ylabel(display_name("JlocQ"))
    axes[1].set_title("Organization-gate ablation: Jnested")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_weight_vs_entropy_plot(path: Path, rows: list[dict[str, float | str]]) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in rows:
        if row["weighting"] != "q1":
            continue
        x = float(row["Wbar"])
        y = float(row["Hnested"])
        label = str(row["label"])
        ax.scatter([x], [y], s=60)
        ax.text(x, y, f" {label}", va="center")

    ax.set_xlabel("Wbar")
    ax.set_ylabel("Hnested")
    ax.set_title("Canonical q1: organized weight vs normalized nested entropy")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_coherence_profiles_comparison(
    path: Path,
    energy_rows: list[dict[str, float | str]],
    orientation_rows: list[dict[str, float | str]],
) -> None:
    plt = require_matplotlib()
    labels = ["fractal", "wavy_stripes", "noise", "patchwork"]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 3.2 * len(labels)), sharex=True)

    for ax, label in zip(axes, labels):
        e_rows = [row for row in energy_rows if row["label"] == label]
        o_rows = [row for row in orientation_rows if row["label"] == label]
        if e_rows:
            k = np.asarray([int(row["k"]) for row in e_rows], dtype=int)
            qenergy = np.asarray(
                [float(row["Qenergy_k"]) if str(row["Qenergy_defined"]) == "True" else np.nan for row in e_rows],
                dtype=float,
            )
            ax.plot(k, qenergy, marker="o", label="Qenergy_k")
        if o_rows:
            k = np.asarray([int(row["k"]) for row in o_rows], dtype=int)
            qorient = np.asarray([float(row["Qorient_k"]) for row in o_rows], dtype=float)
            ax.plot(k, qorient, marker="s", label="Qorient_k")
        ax.set_ylabel(label)
        ax.legend()

    axes[0].set_title("Energy coherence vs orientation coherence")
    axes[-1].set_xlabel("Scale index k")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_scale_support_plot(path: Path, rows: list[dict[str, float | str]]) -> None:
    plt = require_matplotlib()
    compare_labels = ["fractal", "wavy_stripes"]
    metrics = [
        ("Ebar_k", "Ebar_k"),
        ("E_active_fraction", "E_active_fraction"),
        ("effective_support_fraction", "effective_support_fraction"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 9), sharex=True)
    for ax, (key, title) in zip(axes, metrics):
        for label in compare_labels:
            subset = [row for row in rows if row["label"] == label]
            k = np.asarray([int(row["k"]) for row in subset], dtype=int)
            values = np.asarray([float(row[key]) for row in subset], dtype=float)
            ax.plot(k, values, marker="o", label=label)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()

    axes[-1].set_xlabel("Scale index k")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_active_history_histogram(path: Path, counts_by_label: dict[str, dict[str, np.ndarray]]) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    for label in ("fractal", "wavy_stripes"):
        if label not in counts_by_label:
            continue
        ax.hist(
            counts_by_label[label]["n_E"].ravel(),
            bins=np.arange(np.max(counts_by_label[label]["n_E"]) + 2) - 0.5,
            alpha=0.5,
            label=label,
        )
    ax.set_title("Active RG-history length from lifted E")
    ax.set_ylabel("count")
    ax.set_xlabel("number of active scales")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not is_power_of_two(args.size):
        raise ValueError("--size must be a power of two")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_csv_list(args.labels)
    images = built_in_images(args.size, args.seed)

    for label in labels:
        if label not in images:
            raise ValueError(f"unknown label: {label}")

    if args.save_images:
        for label in labels:
            save_image(images[label], args.out_dir / f"{label}.png")

    weighting_rows: list[dict[str, float | str]] = []
    gate_rows: list[dict[str, float | str]] = []
    energy_coherence_rows: list[dict[str, float | str]] = []
    orientation_coherence_rows: list[dict[str, float | str]] = []
    energy_gate_scale_rows: list[dict[str, float | str]] = []
    scale_weight_rows: list[dict[str, float | str]] = []
    support_rows: list[dict[str, float | str]] = []
    active_history_rows: list[dict[str, float | str]] = []
    active_history_hist_rows: list[dict[str, float | str]] = []
    count_maps: dict[str, dict[str, np.ndarray]] = {}

    for label in labels:
        image = images[label]
        C = complexity_profile(image, block_size=2, n_steps=args.n_steps)
        lifted_E = lifted_haar_channel_energy_profile(image, n_steps=args.n_steps)
        lifted_q = lifted_local_orientation_coherence_profile(
            image,
            n_steps=args.n_steps,
            connectivity=args.connectivity,
        )
        native_Ebar, qenergy_profile, energy_variance, qenergy_defined = detail_energy_coherence_profile(
            image,
            n_steps=args.n_steps,
        )
        qorient_profile = local_orientation_coherence_profile(
            image,
            n_steps=args.n_steps,
        )

        for weighting in ("q0", "qsqrt", "q1"):
            W = make_weight_tensor(lifted_E, lifted_q, weighting)
            weighting_rows.append(
                decompose_weights(
                    label=label,
                    weighting=weighting,
                    weights=W,
                    total_cdetail=float(np.sum(C)),
                    lifted_E=lifted_E,
                )
            )

        for gate in ("no_gate", "energy_gate", "orientation_gate"):
            W = make_gate_weight_tensor(
                lifted_E=lifted_E,
                lifted_q=lifted_q,
                qenergy_profile=qenergy_profile,
                qenergy_defined=qenergy_defined,
                gate=gate,
            )
            row = decompose_weights(
                label=label,
                weighting=gate,
                weights=W,
                total_cdetail=float(np.sum(C)),
                lifted_E=lifted_E,
            )
            row["gate"] = gate
            gate_rows.append(row)

        for k in range(len(native_Ebar)):
            energy_coherence_rows.append(
                {
                    "label": label,
                    "k": k,
                    "Ebar_k": float(native_Ebar[k]),
                    "energy_variance_k": float(energy_variance[k]),
                    "Qenergy_k": float(qenergy_profile[k]),
                    "Qenergy_defined": bool(qenergy_defined[k]),
                }
            )
            orientation_coherence_rows.append(
                {
                    "label": label,
                    "k": k,
                    "Qorient_k": float(qorient_profile[k]),
                }
            )
            energy_gate_scale_rows.append(
                {
                    "label": label,
                    "k": k,
                    "Ebar_k": float(np.mean(np.sum(lifted_E[k], axis=-1))),
                    "Qenergy_k": float(qenergy_profile[k]),
                    "Qenergy_defined": bool(qenergy_defined[k]),
                    "Wbar_energy_gate_k": (
                        float(np.mean(np.sum(lifted_E[k], axis=-1))) * float(qenergy_profile[k])
                        if bool(qenergy_defined[k]) and np.isfinite(qenergy_profile[k])
                        else 0.0
                    ),
                }
            )

        E_k = np.sum(lifted_E, axis=-1)
        W_q1 = make_weight_tensor(lifted_E, lifted_q, "q1")
        W_k = np.sum(W_q1, axis=-1)
        Jnested_k = local_scale_orientation_entropy_profile_from_weights(W_q1)

        E_active_stack = []

        for k in range(E_k.shape[0]):
            Ebar_k = float(np.mean(E_k[k]))
            Wbar_k = float(np.mean(W_k[k]))
            qE_fraction_k = float("nan") if Ebar_k <= EPS else Wbar_k / Ebar_k

            scale_weight_rows.append(
                {
                    "label": label,
                    "k": k,
                    "Ebar_k": Ebar_k,
                    "Wbar_k": Wbar_k,
                    "qE_fraction_k": qE_fraction_k,
                    "Jnested_k": float(Jnested_k[k]),
                }
            )

            E_active = active_fraction_map(E_k[k], eps_abs=args.eps_abs, eps_rel=args.eps_rel)
            E_active_stack.append(E_active)

            support_rows.append(
                {
                    "label": label,
                    "k": k,
                    "E_active_fraction": float(np.mean(E_active)),
                    "Ebar_k": Ebar_k,
                    "effective_support_fraction": effective_support_fraction(E_k[k]),
                }
            )

        n_E = np.sum(np.asarray(E_active_stack, dtype=int), axis=0)
        count_maps[label] = {"n_E": n_E}

        row: dict[str, float | str] = {"label": label}
        row.update(summarize_active_history(n_E, "nE"))
        active_history_rows.append(row)

        hist = np.bincount(n_E.ravel())
        total_pixels = int(n_E.size)
        for n_active_scales, pixel_count in enumerate(hist):
            active_history_hist_rows.append(
                {
                    "label": label,
                    "n_active_scales": n_active_scales,
                    "pixel_count": int(pixel_count),
                    "pixel_fraction": float(pixel_count / total_pixels),
                }
            )

    save_csv_rows(
        args.out_dir / "q_ablation_summary.csv",
        [
            "label",
            "weighting",
            "Cdetail",
            "Ebar",
            "Wbar",
            "qE_fraction",
            "Hstruct",
            "Hnested",
            "Ihetero",
            "Jstruct",
            "Jnested",
            "Jhetero",
        ],
        weighting_rows,
    )
    save_csv_rows(
        args.out_dir / "energy_gate_ablation_summary.csv",
        [
            "label",
            "gate",
            "Cdetail",
            "Ebar",
            "Wbar",
            "retained_energy_fraction",
            "Hstruct",
            "Hnested",
            "Ihetero",
            "Jstruct",
            "Jnested",
            "Jhetero",
        ],
        gate_rows,
    )
    save_csv_rows(
        args.out_dir / "energy_coherence_profiles.csv",
        ["label", "k", "Ebar_k", "energy_variance_k", "Qenergy_k", "Qenergy_defined"],
        energy_coherence_rows,
    )
    save_csv_rows(
        args.out_dir / "energy_gate_scale_profiles.csv",
        ["label", "k", "Ebar_k", "Qenergy_k", "Qenergy_defined", "Wbar_energy_gate_k"],
        energy_gate_scale_rows,
    )
    save_csv_rows(
        args.out_dir / "scale_weight_profiles.csv",
        ["label", "k", "Ebar_k", "Wbar_k", "qE_fraction_k", "Jnested_k"],
        scale_weight_rows,
    )
    save_csv_rows(
        args.out_dir / "spatial_support_profiles.csv",
        ["label", "k", "Ebar_k", "E_active_fraction", "effective_support_fraction"],
        support_rows,
    )
    save_csv_rows(
        args.out_dir / "active_history_summary.csv",
        [
            "label",
            "mean_nE",
            "median_nE",
            "std_nE",
            "max_nE",
            "p25_nE",
            "p75_nE",
            "p90_nE",
        ],
        active_history_rows,
    )
    save_csv_rows(
        args.out_dir / "active_history_histogram.csv",
        ["label", "n_active_scales", "pixel_count", "pixel_fraction"],
        active_history_hist_rows,
    )

    if "fractal" in count_maps and "wavy_stripes" in count_maps:
        vmax_E = float(max(np.max(count_maps["fractal"]["n_E"]), np.max(count_maps["wavy_stripes"]["n_E"])))
        try:
            save_count_map(args.out_dir / "fractal_active_scale_count.png", count_maps["fractal"]["n_E"], vmin=0.0, vmax=vmax_E)
            save_count_map(args.out_dir / "wavy_stripes_active_scale_count.png", count_maps["wavy_stripes"]["n_E"], vmin=0.0, vmax=vmax_E)
        except ModuleNotFoundError:
            pass

    try:
        save_q_ablation_plot(args.out_dir / "q_ablation.png", weighting_rows, labels)
        save_weight_vs_entropy_plot(args.out_dir / "weight_vs_entropy.png", weighting_rows)
        save_coherence_profiles_comparison(
            args.out_dir / "coherence_profiles_comparison.png",
            energy_coherence_rows,
            orientation_coherence_rows,
        )
        save_energy_gate_ablation_plot(
            args.out_dir / "energy_gate_complexity_ablation.png",
            gate_rows,
            labels,
        )
        save_scale_support_plot(args.out_dir / "spatial_replication_fractal_vs_wavy.png", support_rows)
        save_active_history_plot(args.out_dir / "active_history_fractal_vs_wavy.png", active_history_hist_rows)
        save_active_history_histogram(args.out_dir / "active_history_histogram.png", count_maps)
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped plot generation.")

    print("q-weighting ablation summary")
    print("label          weighting   Wbar      Hnested   Jnested   Ihetero   Jhetero   Jstruct")
    for label in ("patchwork", "fractal", "wavy_stripes", "noise"):
        for weighting in ("q0", "qsqrt", "q1"):
            matches = [row for row in weighting_rows if row["label"] == label and row["weighting"] == weighting]
            if not matches:
                continue
            row = matches[0]
            print(
                f"{label:<14s} "
                f"{weighting:<10s} "
                f"{float(row['Wbar']):>9.4g} "
                f"{float(row['Hnested']):>9.4g} "
                f"{float(row['Jnested']):>9.4g} "
                f"{float(row['Ihetero']):>9.4g} "
                f"{float(row['Jhetero']):>9.4g} "
                f"{float(row['Jstruct']):>9.4g}"
            )

    print()
    print("Diagnostic ratios")
    for weighting in ("q0", "qsqrt", "q1"):
        fractal = next(row for row in weighting_rows if row["label"] == "fractal" and row["weighting"] == weighting)
        wavy = next(row for row in weighting_rows if row["label"] == "wavy_stripes" and row["weighting"] == weighting)
        noise = next(row for row in weighting_rows if row["label"] == "noise" and row["weighting"] == weighting)

        fractal_jnested = float(fractal["Jnested"])
        fractal_jstruct = float(fractal["Jstruct"])

        print(weighting)
        print(f"  Jnested(wavy) / Jnested(fractal) = {float(wavy['Jnested']) / fractal_jnested:.12g}")
        print(f"  Jstruct(wavy) / Jstruct(fractal) = {float(wavy['Jstruct']) / fractal_jstruct:.12g}")
        print(f"  Jnested(noise) / Jnested(fractal) = {float(noise['Jnested']) / fractal_jnested:.12g}")
        print(f"  Jstruct(noise) / Jstruct(fractal) = {float(noise['Jstruct']) / fractal_jstruct:.12g}")

    print()
    print("Energy-coherence gate ablation")
    print("label          gate               Wbar    Hnested   Jnested   Ihetero   Jhetero   Jstruct")
    for label in ("patchwork", "fractal", "wavy_stripes", "noise", "checkerboard", "straight_stripes", "nested_dyadic"):
        for gate in ("no_gate", "energy_gate", "orientation_gate"):
            matches = [row for row in gate_rows if row["label"] == label and row["gate"] == gate]
            if not matches:
                continue
            row = matches[0]
            print(
                f"{label:<14s} "
                f"{gate:<18s} "
                f"{float(row['Wbar']):>8.4g} "
                f"{float(row['Hnested']):>9.4g} "
                f"{float(row['Jnested']):>9.4g} "
                f"{float(row['Ihetero']):>9.4g} "
                f"{float(row['Jhetero']):>9.4g} "
                f"{float(row['Jstruct']):>9.4g}"
            )

    print()
    print("Energy-gate ratios")
    for gate in ("no_gate", "energy_gate", "orientation_gate"):
        fractal = next(row for row in gate_rows if row["label"] == "fractal" and row["gate"] == gate)
        wavy = next(row for row in gate_rows if row["label"] == "wavy_stripes" and row["gate"] == gate)
        noise = next(row for row in gate_rows if row["label"] == "noise" and row["gate"] == gate)
        patchwork = next(row for row in gate_rows if row["label"] == "patchwork" and row["gate"] == gate)

        fractal_jstruct = float(fractal["Jstruct"])

        print(gate)
        print(f"  Jstruct(fractal) = {float(fractal['Jstruct']):.12g}")
        print(f"  Jstruct(wavy) = {float(wavy['Jstruct']):.12g}")
        print(f"  Jstruct(noise) = {float(noise['Jstruct']):.12g}")
        print(f"  Jstruct(patchwork) = {float(patchwork['Jstruct']):.12g}")
        print(f"  Jstruct(wavy) / Jstruct(fractal) = {float(wavy['Jstruct']) / fractal_jstruct:.12g}")
        print(f"  Jstruct(noise) / Jstruct(fractal) = {float(noise['Jstruct']) / fractal_jstruct:.12g}")
        print(f"  Jstruct(noise) / Jstruct(wavy) = {float(noise['Jstruct']) / float(wavy['Jstruct']):.12g}")

    print()
    print("Retained energy fractions")
    for gate in ("energy_gate", "orientation_gate"):
        print(gate)
        for label in ("fractal", "wavy_stripes", "noise", "patchwork"):
            row = next(row for row in gate_rows if row["label"] == label and row["gate"] == gate)
            print(f"  {label}: {float(row['retained_energy_fraction']):.12g}")

    print()
    print("Multiscale spatial-support diagnostic")
    print("label          mean_nE   median_nE   p90_nE")
    for label in ("fractal", "wavy_stripes"):
        matches = [row for row in active_history_rows if row["label"] == label]
        if not matches:
            continue
        row = matches[0]
        print(
            f"{label:<14s} "
            f"{float(row['mean_nE']):>8.4g} "
            f"{float(row['median_nE']):>11.4g} "
            f"{float(row['p90_nE']):>8.4g}"
        )

    print()
    print("Scale support")
    print("label          k   Ebar_k   active_fraction   effective_support")
    for label in ("fractal", "wavy_stripes"):
        subset = [row for row in support_rows if row["label"] == label]
        for row in subset:
            print(
                f"{label:<14s} "
                f"{int(row['k']):>2d} "
                f"{float(row['Ebar_k']):>8.4g} "
                f"{float(row['E_active_fraction']):>17.4g} "
                f"{float(row['effective_support_fraction']):>19.4g}"
            )


if __name__ == "__main__":
    main()
