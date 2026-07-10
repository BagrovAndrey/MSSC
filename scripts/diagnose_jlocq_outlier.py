from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.complexity import complexity_profile
from mssc.display import display_name, phase_name
from mssc.image_io import load_image, save_image
from mssc.orientation import (
    haar_channel_energy_profile,
    lifted_haar_channel_energy_profile,
    lifted_local_orientation_coherence_profile,
    local_orientation_coherence_profile,
    local_scale_orientation_entropy_profile,
    local_scale_orientation_entropy_profile_with_local_q,
    orientation_diverse_organized_profile,
    orientation_entropy_profile,
    organized_profile,
    scale_orientation_entropy_profile,
)
from mssc.shuffle import phase_scramble


PROFILE_KEYS = ["C", "Q", "D", "O", "Odiv", "Jglob", "Jloc", "JlocQ"]
TOTAL_KEYS = ["C", "O", "Odiv", "Jglob", "Jloc", "JlocQ"]
EPS = 1e-15


def require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def parse_size(value: str) -> int | str | None:
    if value == "auto":
        return "auto"
    if value == "none":
        return None
    return int(value)


def parse_csv_numbers(value: str, cast=float) -> list[int] | list[float]:
    if not value:
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_generate_kind(value: str) -> str:
    if value != "wavy_stripes":
        raise argparse.ArgumentTypeError("only 'wavy_stripes' is currently supported")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why an image has high JlocQ by decomposing the current metric ingredients."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Analyze an existing image file.")
    source.add_argument(
        "--generate",
        type=parse_generate_kind,
        help="Generate a synthetic diagnostic image.",
    )

    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument(
        "--size",
        default="auto",
        help=(
            "For --image: 'auto', 'none', or an integer side length. "
            "For generated images: integer side length."
        ),
    )
    parser.add_argument("--mode", choices=["rgb", "grayscale"], default="grayscale")
    parser.add_argument(
        "--value-range",
        choices=["0_1", "minus1_1"],
        default="minus1_1",
    )
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=4)

    parser.add_argument("--stripe-period", type=float, default=64.0)
    parser.add_argument("--wave-amplitude", type=float, default=24.0)
    parser.add_argument("--wave-period", type=float, default=256.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--binary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the generated wavy stripes are thresholded to +/-1. Default: true.",
    )

    parser.add_argument(
        "--map-scales",
        default=None,
        help="Comma-separated scale indices to export. Default: top 3 by JlocQ_k.",
    )
    parser.add_argument(
        "--phase-scramble-seeds",
        type=int,
        default=0,
        help="Legacy phase-scramble repeat summary. If > 0, analyze seeds 0..N-1.",
    )
    parser.add_argument("--sweep-wave-amplitude", default=None)
    parser.add_argument("--sweep-stripe-period", default=None)
    parser.add_argument("--sweep-wave-period", default=None)
    parser.add_argument(
        "--diagnostics-level",
        choices=["core", "full"],
        default="core",
        help="Default summary detail level. Default: core.",
    )

    parser.add_argument(
        "--offset-average",
        action="store_true",
        help="Run periodic dyadic-offset averaging diagnostics.",
    )
    parser.add_argument(
        "--offset-mode",
        choices=["basic", "powers", "random"],
        default="powers",
        help="Offset set used by --offset-average. Default: powers.",
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=None,
        help="Optional maximum dyadic power for offset generation.",
    )
    parser.add_argument(
        "--num-random-offsets",
        type=int,
        default=32,
        help="Number of offsets for --offset-mode random. Default: 32.",
    )
    parser.add_argument(
        "--offset-seed",
        type=int,
        default=0,
        help="Seed for random offset generation. Default: 0.",
    )
    parser.add_argument(
        "--phase-null-seeds",
        type=int,
        default=0,
        help="If > 0, build a phase-scramble null from seeds 0..N-1.",
    )

    return parser.parse_args()


def make_wavy_stripes(
    size: int = 512,
    stripe_period: float = 64.0,
    wave_amplitude: float = 24.0,
    wave_period: float = 256.0,
    threshold: float = 0.0,
    binary: bool = True,
) -> np.ndarray:
    y, x = np.indices((size, size), dtype=float)
    y_eff = y + wave_amplitude * np.sin(2.0 * np.pi * x / wave_period)
    raw = np.sin(2.0 * np.pi * y_eff / stripe_period)

    if binary:
        return np.where(raw > threshold, 1.0, -1.0).astype(np.float64)

    raw = raw.astype(np.float64)
    vmin = float(np.min(raw))
    vmax = float(np.max(raw))
    if vmax > vmin:
        raw = 2.0 * (raw - vmin) / (vmax - vmin) - 1.0
    else:
        raw = np.zeros_like(raw)
    return raw


def periodic_shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.roll(image, dy, axis=0)
    shifted = np.roll(shifted, dx, axis=1)
    return shifted


def make_offset_list(
    size: int,
    mode: str = "powers",
    max_power: int | None = None,
    num_random_offsets: int = 32,
    offset_seed: int = 0,
) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []

    if mode == "basic":
        candidates = [
            (0, 0),
            (1, 0), (0, 1), (1, 1),
            (2, 0), (0, 2), (2, 2),
            (4, 0), (0, 4), (4, 4),
            (8, 0), (0, 8), (8, 8),
        ]
        offsets = [(dy, dx) for dy, dx in candidates if dy < size and dx < size]

    elif mode == "powers":
        max_valid_power = int(np.floor(np.log2(max(1, size - 1)))) if size > 1 else 0
        if max_power is not None:
            max_valid_power = min(max_valid_power, max_power)

        scales = [0]
        scales.extend(2 ** p for p in range(max_valid_power + 1))

        candidates: list[tuple[int, int]] = [(0, 0)]
        for s in scales:
            candidates.extend([(s, 0), (0, s), (s, s)])
            if s > 0:
                candidates.extend([(s, 2 * s), (2 * s, s)])

        offsets = [(dy, dx) for dy, dx in candidates if dy < size and dx < size]

    elif mode == "random":
        rng = np.random.default_rng(offset_seed)
        offsets = [(0, 0)]
        for _ in range(num_random_offsets):
            offsets.append((int(rng.integers(0, size)), int(rng.integers(0, size))))

    else:
        raise ValueError(f"unknown offset mode: {mode}")

    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in offsets:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def to_display_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)

    if arr.ndim == 3:
        if arr.min() < 0:
            arr = 0.5 * (arr + 1.0)
        return np.clip(arr, 0.0, 1.0)

    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    return np.zeros_like(arr)


def show_image(ax, image: np.ndarray) -> None:
    display = to_display_image(image)
    if display.ndim == 2:
        ax.imshow(display, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(display, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def channel_labels(d: int) -> list[str]:
    if d == 3:
        return ["hx", "hy", "hxy"]
    return [f"channel_{i}" for i in range(d)]


def profile_entropy(profile: np.ndarray, eps: float = EPS) -> float:
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))
    if total <= eps:
        return 0.0

    p = x / total
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def entropic_complexity(profile: np.ndarray, eps: float = EPS) -> float:
    total = float(np.sum(profile))
    if total <= eps:
        return 0.0
    return total * profile_entropy(profile, eps=eps)


def total_metrics_from_summary(summary: dict[str, float]) -> dict[str, float]:
    return {
        "C": summary["C_total"],
        "O": summary["O_total"],
        "Odiv": summary["Odiv_total"],
        "Jglob": summary["Jglob_total"],
        "Jloc": summary["Jloc_total"],
        "JlocQ": summary["JlocQ_total"],
    }


def compute_profiles(
    image: np.ndarray,
    n_steps: int | None,
    connectivity: int,
) -> dict[str, np.ndarray | dict[str, float]]:
    C = complexity_profile(image, block_size=2, n_steps=n_steps)
    Q = local_orientation_coherence_profile(image, n_steps=n_steps)
    D = orientation_entropy_profile(image, n_steps=n_steps)
    O = organized_profile(C, Q)
    Odiv = orientation_diverse_organized_profile(C, Q, D)

    E_channel = haar_channel_energy_profile(image, n_steps=n_steps)
    E_lift = lifted_haar_channel_energy_profile(image, n_steps=n_steps)
    q_lift = lifted_local_orientation_coherence_profile(
        image,
        n_steps=n_steps,
        connectivity=connectivity,
    )

    Jglob = scale_orientation_entropy_profile(E_channel, Q)
    Jloc = local_scale_orientation_entropy_profile(E_lift, Q)
    JlocQ = local_scale_orientation_entropy_profile_with_local_q(E_lift, q_lift)

    W = E_lift * np.maximum(q_lift, 0.0)[..., None]
    Wsum = np.mean(np.sum(W, axis=-1), axis=(1, 2))

    Hloc_factor = np.zeros_like(JlocQ)
    valid = Wsum > EPS
    Hloc_factor[valid] = JlocQ[valid] / Wsum[valid]

    W_channel = np.mean(W, axis=(1, 2))
    E_channel_from_lift = np.mean(E_lift, axis=(1, 2))
    Wtot = np.sum(W, axis=(0, 3))

    summary = {
        "C_total": float(np.sum(C)),
        "O_total": float(np.sum(O)),
        "Odiv_total": float(np.sum(Odiv)),
        "Jglob_total": float(np.sum(Jglob)),
        "Jloc_total": float(np.sum(Jloc)),
        "JlocQ_total": float(np.sum(JlocQ)),
        "H_C": profile_entropy(C),
        "H_O": profile_entropy(O),
        "H_Odiv": profile_entropy(Odiv),
        "H_Jglob": profile_entropy(Jglob),
        "H_Jloc": profile_entropy(Jloc),
        "H_JlocQ": profile_entropy(JlocQ),
        "S_C": entropic_complexity(C),
        "S_O": entropic_complexity(O),
        "S_Odiv": entropic_complexity(Odiv),
        "S_Jglob": entropic_complexity(Jglob),
        "S_Jloc": entropic_complexity(Jloc),
        "S_JlocQ": entropic_complexity(JlocQ),
    }

    return {
        "C": C,
        "Q": Q,
        "D": D,
        "O": O,
        "Odiv": Odiv,
        "Jglob": Jglob,
        "Jloc": Jloc,
        "JlocQ": JlocQ,
        "E_channel": E_channel_from_lift,
        "W_channel": W_channel,
        "E_lift": E_lift,
        "q_lift": q_lift,
        "W": W,
        "Wtot": Wtot,
        "Wsum": Wsum,
        "Hloc_factor": Hloc_factor,
        "summary": summary,
    }


def jlocq_map_for_scale(
    W_scale: np.ndarray,
    Wtot: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    ratio = np.ones_like(W_scale)
    valid = Wtot > eps
    np.divide(W_scale, Wtot[..., None], out=ratio, where=valid[..., None])

    terms = np.zeros_like(W_scale)
    mask = (W_scale > eps) & valid[..., None]
    terms[mask] = -W_scale[mask] * np.log(ratio[mask])
    return np.sum(terms, axis=-1)


def save_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "label",
        "C_total",
        "O_total",
        "Odiv_total",
        "Jglob_total",
        "Jloc_total",
        "JlocQ_total",
    ]
    save_csv_rows(path, fieldnames, rows)


def save_profiles_csv(path: Path, data: dict[str, np.ndarray | dict[str, float]]) -> None:
    rows = []
    n = len(data["C"])  # type: ignore[arg-type]
    for k in range(n):
        rows.append(
            {
                "k": k,
                "C": float(data["C"][k]),  # type: ignore[index]
                "Q": float(data["Q"][k]),  # type: ignore[index]
                "D": float(data["D"][k]),  # type: ignore[index]
                "O": float(data["O"][k]),  # type: ignore[index]
                "Odiv": float(data["Odiv"][k]),  # type: ignore[index]
                "Jglob": float(data["Jglob"][k]),  # type: ignore[index]
                "Jloc": float(data["Jloc"][k]),  # type: ignore[index]
                "JlocQ": float(data["JlocQ"][k]),  # type: ignore[index]
                "Wsum": float(data["Wsum"][k]),  # type: ignore[index]
                "Hloc_factor": float(data["Hloc_factor"][k]),  # type: ignore[index]
            }
        )

    fieldnames = ["k", "C", "Q", "D", "O", "Odiv", "Jglob", "Jloc", "JlocQ", "Wsum", "Hloc_factor"]
    save_csv_rows(path, fieldnames, rows)


def save_channel_csv(path: Path, prefix: str, matrix: np.ndarray) -> list[str]:
    labels = channel_labels(matrix.shape[1])
    fieldnames = ["k"] + [f"{prefix}_{label}" for label in labels]
    rows = []
    for k in range(matrix.shape[0]):
        row: dict[str, float | int] = {"k": k}
        for idx, label in enumerate(labels):
            row[f"{prefix}_{label}"] = float(matrix[k, idx])
        rows.append(row)
    save_csv_rows(path, fieldnames, rows)
    return labels


def save_profile_matrix_csv(path: Path, mean_matrix: np.ndarray, keys: list[str]) -> None:
    rows = []
    for k in range(mean_matrix.shape[0]):
        row: dict[str, float | int] = {"k": k}
        for idx, key in enumerate(keys):
            row[key] = float(mean_matrix[k, idx])
        rows.append(row)
    save_csv_rows(path, ["k"] + keys, rows)


def save_map_png(path: Path, array: np.ndarray, cmap: str = "viridis") -> None:
    plt = require_matplotlib()
    arr = np.asarray(array, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        display = np.zeros_like(arr)
    else:
        display = (arr - vmin) / (vmax - vmin)
    plt.imsave(path, display, cmap=cmap)


def save_selected_scale_maps(
    out_dir: Path,
    data: dict[str, np.ndarray | dict[str, float]],
    scales: list[int],
) -> None:
    maps_dir = out_dir / "selected_scale_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    q_lift = data["q_lift"]  # type: ignore[assignment]
    E_lift = data["E_lift"]  # type: ignore[assignment]
    W = data["W"]  # type: ignore[assignment]
    Wtot = data["Wtot"]  # type: ignore[assignment]

    for k in scales:
        q_map = q_lift[k]
        E_sum = np.sum(E_lift[k], axis=-1)
        W_sum = np.sum(W[k], axis=-1)
        J_map = jlocq_map_for_scale(W[k], Wtot)

        np.save(maps_dir / f"q_lift_k{k}.npy", q_map)
        np.save(maps_dir / f"E_sum_k{k}.npy", E_sum)
        np.save(maps_dir / f"W_sum_k{k}.npy", W_sum)
        np.save(maps_dir / f"JlocQ_map_k{k}.npy", J_map)

        save_map_png(maps_dir / f"q_lift_k{k}.png", q_map, cmap="magma")
        save_map_png(maps_dir / f"E_sum_k{k}.png", E_sum, cmap="viridis")
        save_map_png(maps_dir / f"W_sum_k{k}.png", W_sum, cmap="viridis")
        save_map_png(maps_dir / f"JlocQ_map_k{k}.png", J_map, cmap="viridis")


def save_diagnostic_profiles_plot(path: Path, image: np.ndarray, data: dict[str, np.ndarray | dict[str, float]]) -> None:
    plt = require_matplotlib()
    k = np.arange(len(data["C"]))  # type: ignore[arg-type]

    fig, axes = plt.subplots(7, 1, figsize=(9, 18))

    show_image(axes[0], image)
    axes[0].set_title("Input image")

    axes[1].plot(k, data["C"], marker="o")  # type: ignore[arg-type]
    axes[1].set_ylabel("Cdetail")
    axes[1].set_title(f"{display_name('C')}_k")

    axes[2].plot(k, data["Q"], marker="o")  # type: ignore[arg-type]
    axes[2].set_ylabel("Q")
    axes[2].set_title("Local orientation coherence")

    axes[3].plot(k, data["D"], marker="o")  # type: ignore[arg-type]
    axes[3].set_ylabel("D")
    axes[3].set_title("Orientation entropy / diversity")

    axes[4].plot(k, data["Jglob"], marker="o", label=display_name("Jglob"))  # type: ignore[arg-type]
    axes[4].plot(k, data["Jloc"], marker="o", label="Jloc")  # type: ignore[arg-type]
    axes[4].plot(k, data["JlocQ"], marker="o", label=display_name("JlocQ"))  # type: ignore[arg-type]
    axes[4].set_ylabel("J")
    axes[4].set_title("Scale-orientation entropy profiles")
    axes[4].legend()

    axes[5].plot(k, data["Wsum"], marker="o")  # type: ignore[arg-type]
    axes[5].set_ylabel("Wsum")
    axes[5].set_title("q-weighted organized energy")

    axes[6].plot(k, data["Hloc_factor"], marker="o")  # type: ignore[arg-type]
    axes[6].set_ylabel("Hloc factor")
    axes[6].set_xlabel("Scale index k")
    axes[6].set_title("Local entropy factor = JlocQ_k / Wsum_k")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_channel_profiles_plot(path: Path, data: dict[str, np.ndarray | dict[str, float]]) -> None:
    plt = require_matplotlib()
    E_channel = data["E_channel"]  # type: ignore[assignment]
    W_channel = data["W_channel"]  # type: ignore[assignment]
    labels = channel_labels(E_channel.shape[1])
    k = np.arange(E_channel.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for idx, label in enumerate(labels):
        axes[0].plot(k, E_channel[:, idx], marker="o", label=label)
        axes[1].plot(k, W_channel[:, idx], marker="o", label=label)

    axes[0].set_ylabel("E")
    axes[0].set_title("Unweighted channel energies")
    axes[0].legend()

    axes[1].set_ylabel("W")
    axes[1].set_xlabel("Scale index k")
    axes[1].set_title("q-weighted channel energies")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def totals_row(label: str, summary: dict[str, float]) -> dict[str, float | str]:
    return {
        "label": label,
        "C_total": summary["C_total"],
        "O_total": summary["O_total"],
        "Odiv_total": summary["Odiv_total"],
        "Jglob_total": summary["Jglob_total"],
        "Jloc_total": summary["Jloc_total"],
        "JlocQ_total": summary["JlocQ_total"],
    }


def profile_matrix_from_data(data: dict[str, np.ndarray | dict[str, float]]) -> np.ndarray:
    return np.column_stack([np.asarray(data[key], dtype=float) for key in PROFILE_KEYS])


def totals_vector_from_data(data: dict[str, np.ndarray | dict[str, float]]) -> np.ndarray:
    summary = data["summary"]  # type: ignore[assignment]
    totals = total_metrics_from_summary(summary)
    return np.array([totals[key] for key in TOTAL_KEYS], dtype=float)


def save_sweep_csv_plot(
    out_dir: Path,
    filename_base: str,
    variable_name: str,
    values: list[float],
    rows: list[dict[str, float]],
) -> None:
    plt = require_matplotlib()
    fieldnames = [variable_name, "C", "O", "Odiv", "Jglob", "Jloc", "JlocQ"]
    save_csv_rows(out_dir / f"{filename_base}.csv", fieldnames, rows)

    C = [row["C"] for row in rows]
    JlocQ = [row["JlocQ"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(values, JlocQ, marker="o", label="JlocQ", color="tab:blue")
    ax1.set_xlabel(variable_name)
    ax1.set_ylabel("JlocQ", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(values, C, marker="s", label="C", color="tab:orange")
    ax2.set_ylabel("C", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(out_dir / f"{filename_base}.png", dpi=200)
    plt.close(fig)


def run_sweeps(args: argparse.Namespace, out_dir: Path) -> None:
    sweep_specs = [
        ("sweep_wave_amplitude", "wave_amplitude", args.sweep_wave_amplitude),
        ("sweep_stripe_period", "stripe_period", args.sweep_stripe_period),
        ("sweep_wave_period", "wave_period", args.sweep_wave_period),
    ]

    for filename_base, variable_name, raw_values in sweep_specs:
        if raw_values is None:
            continue

        values = parse_csv_numbers(raw_values, cast=float)
        rows: list[dict[str, float]] = []

        for value in values:
            image = make_wavy_stripes(
                size=int(args.size),
                stripe_period=value if variable_name == "stripe_period" else args.stripe_period,
                wave_amplitude=value if variable_name == "wave_amplitude" else args.wave_amplitude,
                wave_period=value if variable_name == "wave_period" else args.wave_period,
                threshold=args.threshold,
                binary=args.binary,
            )
            data = compute_profiles(image, n_steps=args.n_steps, connectivity=args.connectivity)
            summary = data["summary"]  # type: ignore[assignment]
            rows.append(
                {
                    variable_name: float(value),
                    "C": summary["C_total"],
                    "O": summary["O_total"],
                    "Odiv": summary["Odiv_total"],
                    "Jglob": summary["Jglob_total"],
                    "Jloc": summary["Jloc_total"],
                    "JlocQ": summary["JlocQ_total"],
                }
            )

        save_sweep_csv_plot(out_dir, filename_base, variable_name, list(values), rows)


def top_scales_for_maps(JlocQ: np.ndarray, requested: str | None) -> list[int]:
    if requested:
        return [int(x) for x in parse_csv_numbers(requested, cast=int)]

    order = np.argsort(JlocQ)[::-1]
    top = order[: min(3, len(order))]
    return [int(k) for k in top]


def mean_std_min_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.mean(values, axis=0),
        np.std(values, axis=0),
        np.min(values, axis=0),
        np.max(values, axis=0),
    )


def excess_stats(original: np.ndarray, null_mean: np.ndarray, null_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    excess = original - null_mean
    excess_pos = np.maximum(excess, 0.0)
    excess_z = np.full_like(excess, np.nan, dtype=float)
    valid = null_std > EPS
    excess_z[valid] = excess[valid] / null_std[valid]
    return excess, excess_pos, excess_z


def run_phase_scramble_summary(
    image: np.ndarray,
    n_seeds: int,
    n_steps: int | None,
    connectivity: int,
    out_dir: Path,
) -> tuple[list[dict[str, float | int]], dict[str, float], dict[str, float]] | None:
    if n_seeds <= 0:
        return None

    rows: list[dict[str, float | int]] = []
    totals = {key: [] for key in TOTAL_KEYS}

    for seed in range(n_seeds):
        scrambled = phase_scramble(image, seed=seed, preserve_dc=True)
        data = compute_profiles(scrambled, n_steps=n_steps, connectivity=connectivity)
        summary = data["summary"]  # type: ignore[assignment]
        metric_totals = total_metrics_from_summary(summary)

        row = {"seed": seed, **metric_totals}
        rows.append(row)

        for key in TOTAL_KEYS:
            totals[key].append(float(metric_totals[key]))

    save_csv_rows(
        out_dir / "phase_scramble_summary.csv",
        ["seed"] + TOTAL_KEYS,
        rows,
    )

    mean_row = {f"{key}_total": float(np.mean(totals[key])) for key in TOTAL_KEYS}
    std_row = {f"{key}_total": float(np.std(totals[key])) for key in TOTAL_KEYS}

    return rows, mean_row, std_row


def run_offset_average(
    image: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, np.ndarray | list[tuple[int, int]]]:
    offsets = make_offset_list(
        size=image.shape[0],
        mode=args.offset_mode,
        max_power=args.max_power,
        num_random_offsets=args.num_random_offsets,
        offset_seed=args.offset_seed,
    )

    values = []
    profiles = []

    for dy, dx in offsets:
        shifted = periodic_shift(image, dy=dy, dx=dx)
        data = compute_profiles(shifted, n_steps=args.n_steps, connectivity=args.connectivity)
        values.append(totals_vector_from_data(data))
        profiles.append(profile_matrix_from_data(data))

    values_arr = np.asarray(values, dtype=float)
    profiles_arr = np.asarray(profiles, dtype=float)

    total_mean, total_std, total_min, total_max = mean_std_min_max(values_arr)
    profile_mean, profile_std, _, _ = mean_std_min_max(profiles_arr)

    rows = []
    for (dy, dx), row_values in zip(offsets, values_arr):
        row: dict[str, float | int] = {"dy": dy, "dx": dx}
        for idx, key in enumerate(TOTAL_KEYS):
            row[key] = float(row_values[idx])
        rows.append(row)

    save_csv_rows(out_dir / "offset_values.csv", ["dy", "dx"] + TOTAL_KEYS, rows)

    original_data = compute_profiles(image, n_steps=args.n_steps, connectivity=args.connectivity)
    original_totals = totals_vector_from_data(original_data)

    summary_rows = []
    for idx, key in enumerate(TOTAL_KEYS):
        summary_rows.append(
            {
                "metric": key,
                "original_unshifted": float(original_totals[idx]),
                "offset_mean": float(total_mean[idx]),
                "offset_std": float(total_std[idx]),
                "offset_min": float(total_min[idx]),
                "offset_max": float(total_max[idx]),
            }
        )
    save_csv_rows(
        out_dir / "offset_summary.csv",
        ["metric", "original_unshifted", "offset_mean", "offset_std", "offset_min", "offset_max"],
        summary_rows,
    )

    save_profile_matrix_csv(out_dir / "offset_profiles_mean.csv", profile_mean, PROFILE_KEYS)
    save_profile_matrix_csv(out_dir / "offset_profiles_std.csv", profile_std, PROFILE_KEYS)

    return {
        "offsets": offsets,
        "values": values_arr,
        "profile_mean": profile_mean,
        "profile_std": profile_std,
        "total_mean": total_mean,
        "total_std": total_std,
        "total_min": total_min,
        "total_max": total_max,
        "original_data": original_data,
    }


def save_offset_profiles_plot(
    path: Path,
    image: np.ndarray,
    original_data: dict[str, np.ndarray | dict[str, float]],
    result: dict[str, np.ndarray | list[tuple[int, int]]],
) -> None:
    plt = require_matplotlib()
    profile_mean = result["profile_mean"]  # type: ignore[assignment]
    total_values = result["values"]  # type: ignore[assignment]

    k = np.arange(profile_mean.shape[0])

    fig, axes = plt.subplots(5, 1, figsize=(9, 15))
    show_image(axes[0], image)
    axes[0].set_title("Input image")

    axes[1].plot(k, original_data["C"], marker="o", label="original")  # type: ignore[arg-type]
    axes[1].plot(k, profile_mean[:, 0], marker="s", label="offset mean")
    axes[1].set_ylabel("C")
    axes[1].set_title("C_k: unshifted vs offset mean")
    axes[1].legend()

    axes[2].plot(k, original_data["Q"], marker="o", label="original")  # type: ignore[arg-type]
    axes[2].plot(k, profile_mean[:, 1], marker="s", label="offset mean")
    axes[2].set_ylabel("Q")
    axes[2].set_title("Q_k: unshifted vs offset mean")
    axes[2].legend()

    axes[3].plot(k, original_data["Jglob"], marker="o", label="Jglob original")  # type: ignore[arg-type]
    axes[3].plot(k, profile_mean[:, 5], marker="s", label="Jglob mean")
    axes[3].plot(k, original_data["Jloc"], marker="o", label="Jloc original")  # type: ignore[arg-type]
    axes[3].plot(k, profile_mean[:, 6], marker="s", label="Jloc mean")
    axes[3].plot(k, original_data["JlocQ"], marker="o", label="JlocQ original")  # type: ignore[arg-type]
    axes[3].plot(k, profile_mean[:, 7], marker="s", label="JlocQ mean")
    axes[3].set_ylabel("J")
    axes[3].set_title("Entropy profiles: unshifted vs offset mean")
    axes[3].legend(ncol=2)

    axes[4].bar(np.arange(total_values.shape[0]), total_values[:, TOTAL_KEYS.index("JlocQ")])
    axes[4].set_ylabel("JlocQ")
    axes[4].set_xlabel("Offset index")
    axes[4].set_title("Total JlocQ across offsets")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_phase_null(
    image: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, np.ndarray | np.ndarray | dict[str, np.ndarray]] | None:
    if args.phase_null_seeds <= 0:
        return None

    values = []
    profiles = []
    representative = None
    rows = []

    for seed in range(args.phase_null_seeds):
        scrambled = phase_scramble(image, seed=seed, preserve_dc=True)
        if representative is None:
            representative = scrambled
        data = compute_profiles(scrambled, n_steps=args.n_steps, connectivity=args.connectivity)
        total_vec = totals_vector_from_data(data)
        profile_mat = profile_matrix_from_data(data)

        values.append(total_vec)
        profiles.append(profile_mat)

        row: dict[str, float | int] = {"seed": seed}
        for idx, key in enumerate(TOTAL_KEYS):
            row[key] = float(total_vec[idx])
        rows.append(row)

    values_arr = np.asarray(values, dtype=float)
    profiles_arr = np.asarray(profiles, dtype=float)
    total_mean, total_std, _, _ = mean_std_min_max(values_arr)
    profile_mean, profile_std, _, _ = mean_std_min_max(profiles_arr)

    save_csv_rows(out_dir / "phase_null_values.csv", ["seed"] + TOTAL_KEYS, rows)
    save_profile_matrix_csv(out_dir / "phase_null_profiles_mean.csv", profile_mean, PROFILE_KEYS)
    save_profile_matrix_csv(out_dir / "phase_null_profiles_std.csv", profile_std, PROFILE_KEYS)

    original_data = compute_profiles(image, n_steps=args.n_steps, connectivity=args.connectivity)
    original_totals = totals_vector_from_data(original_data)
    excess, excess_pos, excess_z = excess_stats(original_totals, total_mean, total_std)

    summary_rows = []
    for idx, key in enumerate(TOTAL_KEYS):
        summary_rows.append(
            {
                "metric": display_name(key),
                "original": float(original_totals[idx]),
                "phase_mean": float(total_mean[idx]),
                "phase_std": float(total_std[idx]),
                "excess": float(excess[idx]),
                "excess_pos": float(excess_pos[idx]),
                "excess_z": float(excess_z[idx]),
                "Jphase": float(excess[idx]) if key == "JlocQ" else float("nan"),
                "Jphase_pos": float(excess_pos[idx]) if key == "JlocQ" else float("nan"),
                "Jphase_z": float(excess_z[idx]) if key == "JlocQ" else float("nan"),
            }
        )
    save_csv_rows(
        out_dir / "phase_null_summary.csv",
        ["metric", "original", "phase_mean", "phase_std", "excess", "excess_pos", "excess_z", "Jphase", "Jphase_pos", "Jphase_z"],
        summary_rows,
    )

    return {
        "values": values_arr,
        "profile_mean": profile_mean,
        "profile_std": profile_std,
        "total_mean": total_mean,
        "total_std": total_std,
        "excess": excess,
        "excess_pos": excess_pos,
        "excess_z": excess_z,
        "original_data": original_data,
        "representative": representative,
    }


def save_phase_null_profiles_plot(
    path: Path,
    image: np.ndarray,
    result: dict[str, np.ndarray | dict[str, np.ndarray]],
) -> None:
    plt = require_matplotlib()
    original_data = result["original_data"]  # type: ignore[assignment]
    representative = result["representative"]  # type: ignore[assignment]
    profile_mean = result["profile_mean"]  # type: ignore[assignment]
    profile_std = result["profile_std"]  # type: ignore[assignment]

    k = np.arange(profile_mean.shape[0])

    fig, axes = plt.subplots(6, 1, figsize=(9, 18))
    show_image(axes[0], image)
    axes[0].set_title("Original image")

    show_image(axes[1], representative)
    axes[1].set_title("Representative phase-scrambled image")

    axes[2].plot(k, original_data["C"], marker="o", label="original")  # type: ignore[arg-type]
    axes[2].plot(k, profile_mean[:, 0], marker="s", label="phase mean")
    axes[2].fill_between(k, profile_mean[:, 0] - profile_std[:, 0], profile_mean[:, 0] + profile_std[:, 0], alpha=0.2)
    axes[2].set_ylabel("C")
    axes[2].set_title("C_k vs phase-null mean")
    axes[2].legend()

    axes[3].plot(k, original_data["Q"], marker="o", label="original")  # type: ignore[arg-type]
    axes[3].plot(k, profile_mean[:, 1], marker="s", label="phase mean")
    axes[3].fill_between(k, profile_mean[:, 1] - profile_std[:, 1], profile_mean[:, 1] + profile_std[:, 1], alpha=0.2)
    axes[3].set_ylabel("Q")
    axes[3].set_title("Q_k vs phase-null mean")
    axes[3].legend()

    axes[4].plot(k, original_data["JlocQ"], marker="o", label="original")  # type: ignore[arg-type]
    axes[4].plot(k, profile_mean[:, 7], marker="s", label="phase mean")
    axes[4].fill_between(k, profile_mean[:, 7] - profile_std[:, 7], profile_mean[:, 7] + profile_std[:, 7], alpha=0.2)
    axes[4].set_ylabel(display_name("JlocQ"))
    axes[4].set_title(f"{display_name('JlocQ')}_k vs phase-null mean")
    axes[4].legend()

    axes[5].plot(k, np.asarray(original_data["JlocQ"]) - profile_mean[:, 7], marker="o")
    axes[5].set_ylabel("excess")
    axes[5].set_xlabel("Scale index k")
    axes[5].set_title(f"{phase_name('JlocQ')}_k")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_offset_phase_null(
    image: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, np.ndarray] | None:
    if not args.offset_average or args.phase_null_seeds <= 0:
        return None

    offsets = make_offset_list(
        size=image.shape[0],
        mode=args.offset_mode,
        max_power=args.max_power,
        num_random_offsets=args.num_random_offsets,
        offset_seed=args.offset_seed,
    )

    original_values = []
    original_profiles = []
    rows = []

    for dy, dx in offsets:
        shifted = periodic_shift(image, dy=dy, dx=dx)
        data = compute_profiles(shifted, n_steps=args.n_steps, connectivity=args.connectivity)
        total_vec = totals_vector_from_data(data)
        profile_mat = profile_matrix_from_data(data)
        original_values.append(total_vec)
        original_profiles.append(profile_mat)

        row: dict[str, float | int | str] = {"kind": "original_offset", "seed": -1, "dy": dy, "dx": dx}
        for idx, key in enumerate(TOTAL_KEYS):
            row[key] = float(total_vec[idx])
        rows.append(row)

    phase_values = []
    phase_profiles = []
    for seed in range(args.phase_null_seeds):
        scrambled = phase_scramble(image, seed=seed, preserve_dc=True)
        for dy, dx in offsets:
            shifted = periodic_shift(scrambled, dy=dy, dx=dx)
            data = compute_profiles(shifted, n_steps=args.n_steps, connectivity=args.connectivity)
            total_vec = totals_vector_from_data(data)
            profile_mat = profile_matrix_from_data(data)
            phase_values.append(total_vec)
            phase_profiles.append(profile_mat)

            row = {"kind": "phase_offset", "seed": seed, "dy": dy, "dx": dx}
            for idx, key in enumerate(TOTAL_KEYS):
                row[key] = float(total_vec[idx])
            rows.append(row)

    original_values_arr = np.asarray(original_values, dtype=float)
    original_profiles_arr = np.asarray(original_profiles, dtype=float)
    phase_values_arr = np.asarray(phase_values, dtype=float)
    phase_profiles_arr = np.asarray(phase_profiles, dtype=float)

    original_mean, original_std, _, _ = mean_std_min_max(original_values_arr)
    phase_mean, phase_std, _, _ = mean_std_min_max(phase_values_arr)
    phase_profile_mean, phase_profile_std, _, _ = mean_std_min_max(phase_profiles_arr)
    excess, excess_pos, excess_z = excess_stats(original_mean, phase_mean, phase_std)

    save_csv_rows(
        out_dir / "offset_phase_null_values.csv",
        ["kind", "seed", "dy", "dx"] + TOTAL_KEYS,
        rows,
    )

    summary_rows = []
    original_unshifted = totals_vector_from_data(
        compute_profiles(image, n_steps=args.n_steps, connectivity=args.connectivity)
    )
    for idx, key in enumerate(TOTAL_KEYS):
        summary_rows.append(
            {
                "metric": key,
                "original_unshifted": float(original_unshifted[idx]),
                "original_offset_mean": float(original_mean[idx]),
                "original_offset_std": float(original_std[idx]),
                "phase_offset_mean": float(phase_mean[idx]),
                "phase_offset_std": float(phase_std[idx]),
                "offset_phase_excess": float(excess[idx]),
                "offset_phase_excess_pos": float(excess_pos[idx]),
                "offset_phase_excess_z": float(excess_z[idx]),
            }
        )
    save_csv_rows(
        out_dir / "offset_phase_null_summary.csv",
        [
            "metric",
            "original_unshifted",
            "original_offset_mean",
            "original_offset_std",
            "phase_offset_mean",
            "phase_offset_std",
            "offset_phase_excess",
            "offset_phase_excess_pos",
            "offset_phase_excess_z",
        ],
        summary_rows,
    )

    save_profile_matrix_csv(out_dir / "offset_phase_null_profiles_mean.csv", phase_profile_mean, PROFILE_KEYS)
    save_profile_matrix_csv(out_dir / "offset_phase_null_profiles_std.csv", phase_profile_std, PROFILE_KEYS)

    return {
        "original_mean": original_mean,
        "original_std": original_std,
        "phase_mean": phase_mean,
        "phase_std": phase_std,
        "phase_profile_mean": phase_profile_mean,
        "phase_profile_std": phase_profile_std,
        "excess": excess,
        "excess_z": excess_z,
    }


def save_offset_phase_null_profiles_plot(
    path: Path,
    image: np.ndarray,
    original_data: dict[str, np.ndarray | dict[str, float]],
    result: dict[str, np.ndarray],
) -> None:
    plt = require_matplotlib()
    k = np.arange(result["phase_profile_mean"].shape[0])

    fig, axes = plt.subplots(5, 1, figsize=(9, 15))
    show_image(axes[0], image)
    axes[0].set_title("Original image")

    axes[1].plot(k, original_data["C"], marker="o", label="unshifted original")  # type: ignore[arg-type]
    axes[1].plot(k, result["phase_profile_mean"][:, 0], marker="s", label="phase+offset mean")
    axes[1].fill_between(k, result["phase_profile_mean"][:, 0] - result["phase_profile_std"][:, 0], result["phase_profile_mean"][:, 0] + result["phase_profile_std"][:, 0], alpha=0.2)
    axes[1].set_ylabel("C")
    axes[1].set_title("C_k vs phase+offset null mean")
    axes[1].legend()

    axes[2].plot(k, original_data["Q"], marker="o", label="unshifted original")  # type: ignore[arg-type]
    axes[2].plot(k, result["phase_profile_mean"][:, 1], marker="s", label="phase+offset mean")
    axes[2].fill_between(k, result["phase_profile_mean"][:, 1] - result["phase_profile_std"][:, 1], result["phase_profile_mean"][:, 1] + result["phase_profile_std"][:, 1], alpha=0.2)
    axes[2].set_ylabel("Q")
    axes[2].set_title("Q_k vs phase+offset null mean")
    axes[2].legend()

    axes[3].plot(k, original_data["JlocQ"], marker="o", label="unshifted original")  # type: ignore[arg-type]
    axes[3].plot(k, result["phase_profile_mean"][:, 7], marker="s", label="phase+offset mean")
    axes[3].fill_between(k, result["phase_profile_mean"][:, 7] - result["phase_profile_std"][:, 7], result["phase_profile_mean"][:, 7] + result["phase_profile_std"][:, 7], alpha=0.2)
    axes[3].set_ylabel("JlocQ")
    axes[3].set_title("JlocQ_k vs phase+offset null mean")
    axes[3].legend()

    axes[4].bar(np.arange(len(TOTAL_KEYS)), result["excess"])
    axes[4].set_xticks(np.arange(len(TOTAL_KEYS)))
    axes[4].set_xticklabels(TOTAL_KEYS)
    axes[4].set_ylabel("excess")
    axes[4].set_title("Offset-phase total excess")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_top_scale_summary(data: dict[str, np.ndarray | dict[str, float]]) -> None:
    JlocQ = data["JlocQ"]  # type: ignore[assignment]
    Wsum = data["Wsum"]  # type: ignore[assignment]
    Hloc_factor = data["Hloc_factor"]  # type: ignore[assignment]
    top = np.argsort(JlocQ)[::-1][: min(3, len(JlocQ))]

    print("Top JlocQ scales:")
    for k in top:
        print(
            f"  k={int(k)} JlocQ_k={JlocQ[k]:.12g} "
            f"Wsum_k={Wsum[k]:.12g} Hloc_factor_k={Hloc_factor[k]:.12g}"
        )


def print_channel_summary(data: dict[str, np.ndarray | dict[str, float]]) -> None:
    W_channel = data["W_channel"]  # type: ignore[assignment]
    labels = channel_labels(W_channel.shape[1])
    dominant = np.sum(W_channel, axis=0)
    order = np.argsort(dominant)[::-1]

    print()
    print("Dominant q-weighted channels:")
    for idx in order[: min(3, len(order))]:
        print(f"  {labels[idx]} total_W={dominant[idx]:.12g}")


def print_terminal_summary(
    data: dict[str, np.ndarray | dict[str, float]],
    offset_result: dict[str, np.ndarray | list[tuple[int, int]]] | None,
    phase_null_result: dict[str, np.ndarray | np.ndarray | dict[str, np.ndarray]] | None,
    offset_phase_result: dict[str, np.ndarray] | None,
    diagnostics_level: str,
) -> None:
    summary = data["summary"]  # type: ignore[assignment]

    print_top_scale_summary(data)
    print_channel_summary(data)
    print()
    print(f"{display_name('JlocQ')} diagnostics")
    print()
    print("Unshifted original:")
    print(f"  {display_name('C')} = {summary['C_total']:.12g}")
    print(f"  {display_name('Jglob')} = {summary['Jglob_total']:.12g}")
    print(f"  {display_name('JlocQ')} = {summary['JlocQ_total']:.12g}")
    if diagnostics_level == "full":
        print(f"  O = {summary['O_total']:.12g}")
        print(f"  Odiv = {summary['Odiv_total']:.12g}")
        print(f"  Jloc = {summary['Jloc_total']:.12g}")

    if offset_result is not None:
        mean_val = offset_result["total_mean"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        std_val = offset_result["total_std"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        min_val = offset_result["total_min"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        max_val = offset_result["total_max"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        rel = std_val / mean_val if abs(mean_val) > EPS else float("nan")
        print()
        print("Offset average:")
        print(f"  mean ± std = {mean_val:.12g} ± {std_val:.12g}")
        print(f"  min/max = {min_val:.12g} / {max_val:.12g}")
        print(f"  relative std = {rel:.12g}")

    if phase_null_result is not None:
        mean_val = phase_null_result["total_mean"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        std_val = phase_null_result["total_std"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        excess_val = phase_null_result["excess"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        z_val = phase_null_result["excess_z"][TOTAL_KEYS.index("JlocQ")]  # type: ignore[index]
        print()
        print("Phase null:")
        print(f"  mean ± std = {mean_val:.12g} ± {std_val:.12g}")
        print(f"  {phase_name('JlocQ')} = {excess_val:.12g}")
        print(f"  {phase_name('JlocQ')} z = {z_val:.12g}")

    if offset_phase_result is not None:
        original_mean = offset_phase_result["original_mean"][TOTAL_KEYS.index("JlocQ")]
        phase_mean = offset_phase_result["phase_mean"][TOTAL_KEYS.index("JlocQ")]
        phase_std = offset_phase_result["phase_std"][TOTAL_KEYS.index("JlocQ")]
        excess = offset_phase_result["excess"][TOTAL_KEYS.index("JlocQ")]
        z_val = offset_phase_result["excess_z"][TOTAL_KEYS.index("JlocQ")]
        print()
        print("Offset + phase null:")
        print(f"  {display_name('JlocQ')} offset mean = {original_mean:.12g}")
        print(f"  phase offset mean ± std = {phase_mean:.12g} ± {phase_std:.12g}")
        print(f"  {phase_name('JlocQ')} = {excess:.12g}")
        print(f"  {phase_name('JlocQ')} z = {z_val:.12g}")

    print()
    print("Hints:")
    print(f"  If offset averaging strongly reduces {display_name('JlocQ')}, the outlier is likely dyadic-grid resonance.")
    print(f"  If offset averaging does not reduce {display_name('JlocQ')}, the pattern is genuinely high for this block-Haar observer.")
    print(f"  If {phase_name('JlocQ')} is small, high absolute {display_name('JlocQ')} is mostly explained by phase-scramble-preserved spectral or edge statistics.")
    print(f"  If {phase_name('JlocQ')} is large, original phase organization contributes substantially beyond the null.")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        image = load_image(
            args.image,
            size=parse_size(args.size),
            mode=args.mode,
            value_range=args.value_range,
        )
        label = str(args.image)
    else:
        size = int(args.size)
        image = make_wavy_stripes(
            size=size,
            stripe_period=args.stripe_period,
            wave_amplitude=args.wave_amplitude,
            wave_period=args.wave_period,
            threshold=args.threshold,
            binary=args.binary,
        )
        label = "generated_wavy_stripes"

    save_image(image, args.out_dir / "input_image.png")

    data = compute_profiles(image, n_steps=args.n_steps, connectivity=args.connectivity)
    summary = data["summary"]  # type: ignore[assignment]

    save_summary_csv(args.out_dir / "summary.csv", [totals_row(label, summary)])
    save_profiles_csv(args.out_dir / "profiles.csv", data)
    save_channel_csv(args.out_dir / "channel_energy.csv", "E", data["E_channel"])  # type: ignore[arg-type]
    save_channel_csv(args.out_dir / "q_weighted_channel_energy.csv", "W", data["W_channel"])  # type: ignore[arg-type]
    save_diagnostic_profiles_plot(args.out_dir / "diagnostic_profiles.png", image, data)
    save_channel_profiles_plot(args.out_dir / "channel_energy_profiles.png", data)

    selected_scales = top_scales_for_maps(data["JlocQ"], args.map_scales)  # type: ignore[arg-type]
    save_selected_scale_maps(args.out_dir, data, selected_scales)

    phase_stats = run_phase_scramble_summary(
        image,
        n_seeds=args.phase_scramble_seeds,
        n_steps=args.n_steps,
        connectivity=args.connectivity,
        out_dir=args.out_dir,
    )
    if phase_stats is not None:
        _, mean_row, std_row = phase_stats
        save_summary_csv(
            args.out_dir / "summary.csv",
            [
                totals_row(label, summary),
                {"label": "phase_scrambled_mean", **mean_row},
                {"label": "phase_scrambled_std", **std_row},
            ],
        )

    offset_result = None
    if args.offset_average:
        offset_result = run_offset_average(image, args, args.out_dir)
        save_offset_profiles_plot(
            args.out_dir / "offset_profiles.png",
            image=image,
            original_data=data,
            result=offset_result,
        )

    phase_null_result = run_phase_null(image, args, args.out_dir)
    if phase_null_result is not None:
        save_phase_null_profiles_plot(
            args.out_dir / "phase_null_profiles.png",
            image=image,
            result=phase_null_result,
        )

    offset_phase_result = run_offset_phase_null(image, args, args.out_dir)
    if offset_phase_result is not None:
        save_offset_phase_null_profiles_plot(
            args.out_dir / "offset_phase_null_profiles.png",
            image=image,
            original_data=data,
            result=offset_phase_result,
        )

    if args.generate == "wavy_stripes":
        run_sweeps(args, args.out_dir)

    print(f"Saved diagnostics to: {args.out_dir}")
    print_terminal_summary(
        data,
        offset_result,
        phase_null_result,
        offset_phase_result,
        diagnostics_level=args.diagnostics_level,
    )


if __name__ == "__main__":
    main()
