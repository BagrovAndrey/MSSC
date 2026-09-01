from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.complexity import coarse_grain, complexity_profile, max_steps
from mssc.orientation import (
    detail_energy_correlation_stats_from_map,
    detail_energy_map,
    heterogeneous_complexity_profile_from_weights,
    lifted_haar_channel_energy_profile,
    local_detail_energy_correlation_map,
    local_scale_orientation_entropy_profile_from_weights,
    native_neighbor_pair_count,
    structural_complexity_profile_from_weights,
)
from scripts.benchmark_toy_panel import (
    make_nested_dyadic,
    make_noise,
    make_patchwork,
    make_spectral_fractal_binary,
    make_stripes,
)
from scripts.diagnose_jlocq_outlier import EPS, make_wavy_stripes
from scripts.validate_mvp_complexity import require_matplotlib


GATE_VARIANTS = [
    "global_pos_current",
    "global_pos",
    "global_abs",
    "global_sq",
    "local_pos",
    "local_abs",
    "local_sq",
]

IMAGE_ORDER = [
    "stripes",
    "noise",
    "wavy_stripes",
    "fractal",
    "nested_dyadic",
    "patchwork",
    "half_fractal_half_noise",
]

GATE_TITLES = {
    "global_pos_current": "Energy-gate ablation: global max(rho, 0), current undefined->0",
    "global_pos": "Energy-gate ablation: global max(rho, 0), constant-energy regular",
    "global_abs": "Energy-gate ablation: global |rho|",
    "global_sq": "Energy-gate ablation: global rho^2",
    "local_pos": "Energy-gate ablation: local max(rho, 0)",
    "local_abs": "Energy-gate ablation: local |rho|",
    "local_sq": "Energy-gate ablation: local rho^2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study of alternative energy-coherence gates."
    )
    parser.add_argument("--size", type=int, default=512, help="Synthetic image size. Must be a power of two.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for stochastic synthetic generators.")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional number of RG steps.")
    parser.add_argument(
        "--min-qenergy-pairs",
        type=int,
        default=32,
        help="Minimum global horizontal+vertical native-block pair count. Default: 32.",
    )
    parser.add_argument(
        "--local-window-blocks",
        type=int,
        default=5,
        help="Odd local window size on the native block grid. Default: 5.",
    )
    parser.add_argument(
        "--min-local-pairs",
        type=int,
        default=12,
        help="Minimum local horizontal+vertical native-block pair count. Default: 12.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def make_synthetic_images(size: int, seed: int) -> dict[str, np.ndarray]:
    fractal = make_spectral_fractal_binary(size, beta=2.5, seed=seed)
    noise = make_noise(size, seed=seed)
    half = size // 2
    half_fractal_half_noise = np.empty((size, size), dtype=np.float64)
    half_fractal_half_noise[:, :half] = fractal[:, :half]
    half_fractal_half_noise[:, half:] = noise[:, half:]

    return {
        "stripes": make_stripes(size, period=16, orientation="vertical"),
        "noise": noise,
        "wavy_stripes": make_wavy_stripes(
            size=size,
            stripe_period=64.0,
            wave_amplitude=24.0,
            wave_period=256.0,
            threshold=0.0,
            binary=True,
        ),
        "fractal": fractal,
        "nested_dyadic": make_nested_dyadic(size),
        "patchwork": make_patchwork(size),
        "half_fractal_half_noise": half_fractal_half_noise,
    }


def transform_global_rho(rho: float, transform: str) -> float:
    if transform == "pos":
        return max(rho, 0.0)
    if transform == "abs":
        return abs(rho)
    if transform == "sq":
        return rho * rho
    raise ValueError(transform)


def gate_variant_map_for_scale(
    energy_map: np.ndarray,
    gate_variant: str,
    min_qenergy_pairs: int,
    local_window_blocks: int,
    min_local_pairs: int,
) -> tuple[np.ndarray, dict[str, object]]:
    e_map = np.asarray(energy_map, dtype=float)
    stats = detail_energy_correlation_stats_from_map(e_map, eps=EPS)
    pair_count = int(stats["pair_count"])
    zero_energy = bool(stats["zero_energy"])
    nonzero_constant = bool(stats["nonzero_constant"])
    rho_all = float(stats["rho_all"])

    diagnostics: dict[str, object] = {
        "rho_all": float(stats["rho_all"]),
        "rho_horizontal": float(stats["rho_horizontal"]),
        "rho_vertical": float(stats["rho_vertical"]),
        "q_mean": 0.0,
        "q_defined_fraction": 0.0,
        "pair_count": pair_count,
        "gate_is_local": gate_variant.startswith("local_"),
    }

    if gate_variant == "global_pos_current":
        q_scalar = 0.0
        if bool(stats["defined_all"]) and pair_count >= min_qenergy_pairs and np.isfinite(rho_all):
            q_scalar = max(rho_all, 0.0)
            diagnostics["q_defined_fraction"] = 1.0
        q_map = np.full_like(e_map, q_scalar, dtype=float)
        diagnostics["q_mean"] = float(q_scalar)
        return q_map, diagnostics

    if gate_variant.startswith("global_"):
        transform = gate_variant.split("_", 1)[1]
        if zero_energy:
            q_scalar = 0.0
        elif nonzero_constant:
            q_scalar = 1.0
        elif bool(stats["defined_all"]) and pair_count >= min_qenergy_pairs and np.isfinite(rho_all):
            q_scalar = transform_global_rho(rho_all, transform=transform)
            diagnostics["q_defined_fraction"] = 1.0
        else:
            q_scalar = 0.0

        q_map = np.full_like(e_map, q_scalar, dtype=float)
        diagnostics["q_mean"] = float(q_scalar)
        return q_map, diagnostics

    transform = gate_variant.split("_", 1)[1]
    q_map, defined_map, local_pair_count_map = local_detail_energy_correlation_map(
        e_map,
        window_size=local_window_blocks,
        min_local_pairs=min_local_pairs,
        transform=transform,
        eps=EPS,
    )
    diagnostics["q_mean"] = float(np.mean(q_map)) if q_map.size else 0.0
    diagnostics["q_defined_fraction"] = float(np.mean(defined_map)) if defined_map.size else 0.0
    diagnostics["local_pair_count_mean"] = float(np.mean(local_pair_count_map)) if local_pair_count_map.size else 0.0
    return q_map, diagnostics


def lifted_gate_weight_profile(
    image: np.ndarray,
    gate_variant: str,
    n_steps: int | None,
    min_qenergy_pairs: int,
    local_window_blocks: int,
    min_local_pairs: int,
) -> tuple[np.ndarray, list[dict[str, object]], list[np.ndarray]]:
    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    lifted_E = lifted_haar_channel_energy_profile(image, n_steps=n_steps)
    current = np.asarray(image, dtype=float)
    q_diagnostics: list[dict[str, object]] = []
    native_q_maps: list[np.ndarray] = []
    lifted_q_maps: list[np.ndarray] = []

    for k in range(n_steps):
        e_map = detail_energy_map(current)
        q_map_native, diag = gate_variant_map_for_scale(
            e_map,
            gate_variant=gate_variant,
            min_qenergy_pairs=min_qenergy_pairs,
            local_window_blocks=local_window_blocks,
            min_local_pairs=min_local_pairs,
        )
        native_q_maps.append(q_map_native)
        q_diagnostics.append(diag)

        factor = 2 ** (k + 1)
        lifted_q = np.repeat(np.repeat(q_map_native, factor, axis=0), factor, axis=1)
        lifted_q_maps.append(lifted_q)
        current = coarse_grain(current, block_size=2)

    lifted_q_profile = np.asarray(lifted_q_maps, dtype=float)
    W = lifted_E * lifted_q_profile[..., None]
    return W, q_diagnostics, native_q_maps


def analyze_gate_variant(
    image: np.ndarray,
    gate_variant: str,
    n_steps: int | None,
    min_qenergy_pairs: int,
    local_window_blocks: int,
    min_local_pairs: int,
) -> dict[str, object]:
    c_profile = complexity_profile(image, block_size=2, n_steps=n_steps)
    lifted_E = lifted_haar_channel_energy_profile(image, n_steps=n_steps)
    W, q_diagnostics, native_q_maps = lifted_gate_weight_profile(
        image=image,
        gate_variant=gate_variant,
        n_steps=n_steps,
        min_qenergy_pairs=min_qenergy_pairs,
        local_window_blocks=local_window_blocks,
        min_local_pairs=min_local_pairs,
    )

    jnested_k = local_scale_orientation_entropy_profile_from_weights(W)
    jstruct_k = structural_complexity_profile_from_weights(W)
    jhetero_k = jstruct_k - jnested_k

    jnested = float(np.sum(jnested_k))
    jstruct = float(np.sum(jstruct_k))
    jhetero = float(np.sum(jhetero_k))
    cdetail = float(np.sum(c_profile))
    total_E = float(np.sum(lifted_E))
    total_W = float(np.sum(W))
    wbar = float(np.mean(np.sum(W, axis=(0, 3))))

    r = float("nan") if total_E <= EPS else total_W / total_E
    hnested = float("nan") if wbar <= EPS else jnested / wbar
    ihetero = float("nan") if wbar <= EPS else jhetero / wbar
    hstruct = float("nan") if wbar <= EPS else jstruct / wbar
    knested = float("nan") if cdetail <= EPS else jnested / cdetail
    khetero = float("nan") if cdetail <= EPS else jhetero / cdetail
    kstruct = float("nan") if cdetail <= EPS else jstruct / cdetail

    return {
        "C_profile": c_profile,
        "W_profile": W,
        "J_nested_k": jnested_k,
        "J_struct_k": jstruct_k,
        "J_hetero_k": jhetero_k,
        "J_nested": jnested,
        "J_struct": jstruct,
        "J_hetero": jhetero,
        "K_nested": knested,
        "K_hetero": khetero,
        "K_struct": kstruct,
        "R": r,
        "H_nested": hnested,
        "I_hetero": ihetero,
        "H_struct": hstruct,
        "q_diagnostics": q_diagnostics,
        "native_q_maps": native_q_maps,
    }


def half_diagnostics(
    image_name: str,
    W_profile: np.ndarray,
    lifted_E: np.ndarray,
    q_diagnostics: list[dict[str, object]],
) -> dict[str, float]:
    if image_name != "half_fractal_half_noise":
        return {}

    half = W_profile.shape[2] // 2
    left_W = W_profile[:, :, :half, :]
    right_W = W_profile[:, :, half:, :]
    left_E = lifted_E[:, :, :half, :]
    right_E = lifted_E[:, :, half:, :]

    left_wbar = float(np.mean(np.sum(left_W, axis=(0, 3))))
    right_wbar = float(np.mean(np.sum(right_W, axis=(0, 3))))
    left_ebar = float(np.mean(np.sum(left_E, axis=(0, 3))))
    right_ebar = float(np.mean(np.sum(right_E, axis=(0, 3))))

    left_q_mean = 0.0
    right_q_mean = 0.0
    if q_diagnostics:
        left_vals = []
        right_vals = []
        for diag in q_diagnostics:
            left_vals.append(float(diag.get("left_q_mean", diag["q_mean"])))
            right_vals.append(float(diag.get("right_q_mean", diag["q_mean"])))
        left_q_mean = float(np.mean(left_vals))
        right_q_mean = float(np.mean(right_vals))

    return {
        "left_q_mean": left_q_mean,
        "right_q_mean": right_q_mean,
        "left_E_mean": left_ebar,
        "right_E_mean": right_ebar,
        "left_W_mean": left_wbar,
        "right_W_mean": right_wbar,
        "left_retained_fraction": float("nan") if left_ebar <= EPS else left_wbar / left_ebar,
        "right_retained_fraction": float("nan") if right_ebar <= EPS else right_wbar / right_ebar,
    }


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def print_gate_block(
    gate_variant: str,
    image_name: str,
    result: dict[str, object],
    extra: dict[str, float],
) -> None:
    jnested_k = np.asarray(result["J_nested_k"], dtype=float)
    jstruct_k = np.asarray(result["J_struct_k"], dtype=float)
    jhetero_k = np.asarray(result["J_hetero_k"], dtype=float)

    print()
    print(f"{gate_variant} :: {image_name}")
    print(
        f"  Jnested={format_float(float(result['J_nested']))} "
        f"Jhetero={format_float(float(result['J_hetero']))} "
        f"Jstruct={format_float(float(result['J_struct']))}"
    )
    print(
        f"  d(sum Jnested_k)={format_float(float(np.sum(jnested_k) - float(result['J_nested'])))} "
        f"d(sum Jstruct_k)={format_float(float(np.sum(jstruct_k) - float(result['J_struct'])))} "
        f"d(sum Jhetero_k)={format_float(float(np.sum(jhetero_k) - float(result['J_hetero'])))} "
        f"d(Jnested+Jhetero-Jstruct)={format_float(float(result['J_nested']) + float(result['J_hetero']) - float(result['J_struct']))}"
    )

    if image_name == "nested_dyadic":
        q_means = [float(diag["q_mean"]) for diag in result["q_diagnostics"]]
        print(f"  nested_dyadic q_mean_k: {' '.join(format_float(v) for v in q_means)}")

    if extra:
        print(
            f"  half stats: "
            f"left_q={format_float(extra['left_q_mean'])} "
            f"right_q={format_float(extra['right_q_mean'])} "
            f"left_R={format_float(extra['left_retained_fraction'])} "
            f"right_R={format_float(extra['right_retained_fraction'])}"
        )


def compute_shared_ylim(
    results_by_gate: dict[str, dict[str, dict[str, object]]],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for image_name in IMAGE_ORDER:
        vals = []
        for gate_variant in GATE_VARIANTS:
            result = results_by_gate[gate_variant][image_name]
            for key in ("J_nested_k", "J_struct_k", "J_hetero_k"):
                arr = np.asarray(result[key], dtype=float)
                if arr.size:
                    vals.append(float(np.min(arr)))
                    vals.append(float(np.max(arr)))

        ymin = min(vals) if vals else -0.1
        ymax = max(vals) if vals else 0.1
        if abs(ymax - ymin) <= EPS:
            pad = max(0.05, 0.1 * max(abs(ymin), 1.0))
        else:
            pad = 0.08 * (ymax - ymin)
        limits[image_name] = (ymin - pad, ymax + pad)
    return limits


def save_gate_figure(
    path: Path,
    gate_variant: str,
    images: dict[str, np.ndarray],
    results: dict[str, dict[str, object]],
    ylimits: dict[str, tuple[float, float]],
) -> None:
    plt = require_matplotlib()
    nrows = len(IMAGE_ORDER)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(10.8, max(2.6 * nrows, 10.0)),
        gridspec_kw={"width_ratios": [1.0, 2.7]},
        constrained_layout=True,
    )

    if nrows == 1:
        axes = np.asarray([axes], dtype=object)

    for row_index, image_name in enumerate(IMAGE_ORDER):
        image_ax = axes[row_index, 0]
        graph_ax = axes[row_index, 1]
        result = results[image_name]
        k = np.arange(len(np.asarray(result["J_nested_k"], dtype=float)))

        image_ax.imshow(images[image_name], cmap="gray", vmin=-1.0, vmax=1.0, interpolation="nearest")
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_ax.set_title(image_name.replace("_", " "), fontsize=11)

        graph_ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)
        graph_ax.plot(k, result["J_nested_k"], color="#1d3557", marker="o", linewidth=2.0, markersize=4.2, label="Jnested_k")
        graph_ax.plot(k, result["J_struct_k"], color="#d62828", marker="s", linewidth=2.0, markersize=4.2, label="Jstruct_k")
        graph_ax.plot(k, result["J_hetero_k"], color="#2a9d8f", marker="^", linewidth=2.0, markersize=4.2, label="Jhetero_k")
        graph_ax.set_xlim(-0.2, len(k) - 0.8 if len(k) else 0.8)
        graph_ax.set_ylim(*ylimits[image_name])
        graph_ax.grid(alpha=0.22, linewidth=0.8)
        graph_ax.tick_params(labelsize=9)
        graph_ax.set_xlabel("RG scale k", fontsize=10)
        graph_ax.set_ylabel("J contribution", fontsize=10)
        graph_ax.text(
            0.01,
            0.95,
            (
                f"Jn={float(result['J_nested']):.3f}   "
                f"Jh={float(result['J_hetero']):.3f}   "
                f"Js={float(result['J_struct']):.3f}"
            ),
            transform=graph_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        )

        if row_index == 0:
            graph_ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

    fig.suptitle(GATE_TITLES[gate_variant], fontsize=15)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not is_power_of_two(args.size):
        raise ValueError("--size must be a power of two")
    if args.min_qenergy_pairs < 0:
        raise ValueError("--min-qenergy-pairs must be nonnegative")
    if args.local_window_blocks <= 0 or args.local_window_blocks % 2 == 0:
        raise ValueError("--local-window-blocks must be a positive odd integer")
    if args.min_local_pairs < 0:
        raise ValueError("--min-local-pairs must be nonnegative")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    images = make_synthetic_images(args.size, args.seed)
    results_by_gate: dict[str, dict[str, dict[str, object]]] = {}
    summary_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    for gate_variant in GATE_VARIANTS:
        gate_results: dict[str, dict[str, object]] = {}
        for image_name in IMAGE_ORDER:
            result = analyze_gate_variant(
                image=images[image_name],
                gate_variant=gate_variant,
                n_steps=args.n_steps,
                min_qenergy_pairs=args.min_qenergy_pairs,
                local_window_blocks=args.local_window_blocks,
                min_local_pairs=args.min_local_pairs,
            )
            gate_results[image_name] = result

        results_by_gate[gate_variant] = gate_results

    shared_ylimits = compute_shared_ylim(results_by_gate)

    for gate_variant in GATE_VARIANTS:
        for image_name in IMAGE_ORDER:
            result = results_by_gate[gate_variant][image_name]
            lifted_E = lifted_haar_channel_energy_profile(images[image_name], n_steps=args.n_steps)
            extra = half_diagnostics(
                image_name=image_name,
                W_profile=np.asarray(result["W_profile"], dtype=float),
                lifted_E=lifted_E,
                q_diagnostics=result["q_diagnostics"],
            )

            if image_name == "half_fractal_half_noise":
                q_means_left = []
                q_means_right = []
                for q_map in result["native_q_maps"]:
                    half = q_map.shape[1] // 2
                    q_means_left.append(float(np.mean(q_map[:, :half])) if half > 0 else 0.0)
                    q_means_right.append(float(np.mean(q_map[:, half:])) if q_map.shape[1] - half > 0 else 0.0)
                if q_means_left:
                    extra["left_q_mean"] = float(np.mean(q_means_left))
                    extra["right_q_mean"] = float(np.mean(q_means_right))

            print_gate_block(gate_variant, image_name, result, extra)

            summary_rows.append(
                {
                    "gate_variant": gate_variant,
                    "image": image_name,
                    "J_nested": float(result["J_nested"]),
                    "J_hetero": float(result["J_hetero"]),
                    "J_struct": float(result["J_struct"]),
                    "K_nested": float(result["K_nested"]),
                    "K_hetero": float(result["K_hetero"]),
                    "K_struct": float(result["K_struct"]),
                    "R": float(result["R"]),
                    "H_nested": float(result["H_nested"]),
                    "I_hetero": float(result["I_hetero"]),
                    "H_struct": float(result["H_struct"]),
                    "left_q_mean": extra.get("left_q_mean", float("nan")),
                    "right_q_mean": extra.get("right_q_mean", float("nan")),
                    "left_retained_fraction": extra.get("left_retained_fraction", float("nan")),
                    "right_retained_fraction": extra.get("right_retained_fraction", float("nan")),
                }
            )

            q_diags = result["q_diagnostics"]
            for k in range(len(np.asarray(result["J_nested_k"], dtype=float))):
                q_mean_k = float(q_diags[k]["q_mean"])
                row = {
                    "gate_variant": gate_variant,
                    "image": image_name,
                    "k": k,
                    "J_nested_k": float(np.asarray(result["J_nested_k"], dtype=float)[k]),
                    "J_struct_k": float(np.asarray(result["J_struct_k"], dtype=float)[k]),
                    "J_hetero_k": float(np.asarray(result["J_hetero_k"], dtype=float)[k]),
                    "q_mean_k": q_mean_k,
                    "rho_all": float(q_diags[k]["rho_all"]),
                    "rho_horizontal": float(q_diags[k]["rho_horizontal"]),
                    "rho_vertical": float(q_diags[k]["rho_vertical"]),
                }
                profile_rows.append(row)

    save_csv_rows(
        args.out_dir / "gate_ablation_summary.csv",
        [
            "gate_variant",
            "image",
            "J_nested",
            "J_hetero",
            "J_struct",
            "K_nested",
            "K_hetero",
            "K_struct",
            "R",
            "H_nested",
            "I_hetero",
            "H_struct",
            "left_q_mean",
            "right_q_mean",
            "left_retained_fraction",
            "right_retained_fraction",
        ],
        summary_rows,
    )
    save_csv_rows(
        args.out_dir / "gate_ablation_profiles.csv",
        [
            "gate_variant",
            "image",
            "k",
            "J_nested_k",
            "J_struct_k",
            "J_hetero_k",
            "q_mean_k",
            "rho_all",
            "rho_horizontal",
            "rho_vertical",
        ],
        profile_rows,
    )

    try:
        for gate_variant in GATE_VARIANTS:
            save_gate_figure(
                args.out_dir / f"gate_ablation_{gate_variant}.png",
                gate_variant=gate_variant,
                images=images,
                results=results_by_gate[gate_variant],
                ylimits=shared_ylimits,
            )
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped figure generation.")


if __name__ == "__main__":
    main()
