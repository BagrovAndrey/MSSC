from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mssc.complexity import complexity_profile
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
        help="If > 0, analyze phase-scrambled seeds 0..N-1.",
    )
    parser.add_argument("--sweep-wave-amplitude", default=None)
    parser.add_argument("--sweep-stripe-period", default=None)
    parser.add_argument("--sweep-wave-period", default=None)

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


def profile_entropy(profile: np.ndarray, eps: float = 1e-15) -> float:
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))
    if total <= eps:
        return 0.0

    p = x / total
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def entropic_complexity(profile: np.ndarray, eps: float = 1e-15) -> float:
    total = float(np.sum(profile))
    if total <= eps:
        return 0.0
    return total * profile_entropy(profile, eps=eps)


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
    valid = Wsum > 1e-15
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
    eps: float = 1e-15,
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
    axes[1].set_ylabel("C")
    axes[1].set_title("Naive MSSC profile")

    axes[2].plot(k, data["Q"], marker="o")  # type: ignore[arg-type]
    axes[2].set_ylabel("Q")
    axes[2].set_title("Local orientation coherence")

    axes[3].plot(k, data["D"], marker="o")  # type: ignore[arg-type]
    axes[3].set_ylabel("D")
    axes[3].set_title("Orientation entropy / diversity")

    axes[4].plot(k, data["Jglob"], marker="o", label="Jglob")  # type: ignore[arg-type]
    axes[4].plot(k, data["Jloc"], marker="o", label="Jloc")  # type: ignore[arg-type]
    axes[4].plot(k, data["JlocQ"], marker="o", label="JlocQ")  # type: ignore[arg-type]
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
    totals = {key: [] for key in ["C", "O", "Odiv", "Jglob", "Jloc", "JlocQ"]}

    for seed in range(n_seeds):
        scrambled = phase_scramble(image, seed=seed, preserve_dc=True)
        data = compute_profiles(scrambled, n_steps=n_steps, connectivity=connectivity)
        summary = data["summary"]  # type: ignore[assignment]

        row = {
            "seed": seed,
            "C": summary["C_total"],
            "O": summary["O_total"],
            "Odiv": summary["Odiv_total"],
            "Jglob": summary["Jglob_total"],
            "Jloc": summary["Jloc_total"],
            "JlocQ": summary["JlocQ_total"],
        }
        rows.append(row)

        for key in totals:
            totals[key].append(float(row[key]))

    save_csv_rows(
        out_dir / "phase_scramble_summary.csv",
        ["seed", "C", "O", "Odiv", "Jglob", "Jloc", "JlocQ"],
        rows,
    )

    mean_row = {f"{key}_total": float(np.mean(values)) for key, values in totals.items()}
    std_row = {f"{key}_total": float(np.std(values)) for key, values in totals.items()}

    return rows, mean_row, std_row


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


def print_terminal_summary(
    data: dict[str, np.ndarray | dict[str, float]],
    phase_stats: tuple[list[dict[str, float | int]], dict[str, float], dict[str, float]] | None,
) -> None:
    JlocQ = data["JlocQ"]  # type: ignore[assignment]
    Wsum = data["Wsum"]  # type: ignore[assignment]
    Hloc_factor = data["Hloc_factor"]  # type: ignore[assignment]
    W_channel = data["W_channel"]  # type: ignore[assignment]
    summary = data["summary"]  # type: ignore[assignment]

    top = np.argsort(JlocQ)[::-1][: min(3, len(JlocQ))]
    print("Top JlocQ scales:")
    for k in top:
        print(
            f"  k={int(k)} JlocQ_k={JlocQ[k]:.12g} "
            f"Wsum_k={Wsum[k]:.12g} Hloc_factor_k={Hloc_factor[k]:.12g}"
        )

    labels = channel_labels(W_channel.shape[1])
    dominant = np.sum(W_channel, axis=0)
    order = np.argsort(dominant)[::-1]
    print()
    print("Dominant q-weighted channels:")
    for idx in order[: min(3, len(order))]:
        print(f"  {labels[idx]} total_W={dominant[idx]:.12g}")

    if phase_stats is not None:
        _, mean_row, std_row = phase_stats
        print()
        print("Phase-scramble comparison:")
        print(f"  original JlocQ = {summary['JlocQ_total']:.12g}")
        print(
            f"  phase mean ± std = "
            f"{mean_row['JlocQ_total']:.12g} ± {std_row['JlocQ_total']:.12g}"
        )


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

    if args.generate == "wavy_stripes":
        run_sweeps(args, args.out_dir)

    print(f"Saved diagnostics to: {args.out_dir}")
    print_terminal_summary(data, phase_stats)


if __name__ == "__main__":
    main()
