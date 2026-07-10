from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mssc.complexity import complexity_profile, max_steps
from mssc.display import display_name
from mssc.image_io import save_image
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
from scripts.diagnose_jlocq_outlier import make_wavy_stripes


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def profile_entropy(profile: np.ndarray, eps: float = 1e-15) -> float:
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))

    if total <= eps:
        return 0.0

    p = x / total
    p = p[p > eps]

    return float(-np.sum(p * np.log(p)))


def entropic_complexity(profile: np.ndarray, eps: float = 1e-15) -> float:
    x = np.asarray(profile, dtype=float)
    total = float(np.sum(x))

    if total <= eps:
        return 0.0

    return total * profile_entropy(x, eps=eps)


def make_stripes(L: int, period: int = 16, orientation: str = "vertical") -> np.ndarray:
    x = np.arange(L)
    vals = ((x // period) % 2) * 2 - 1

    if orientation == "vertical":
        return np.tile(vals[None, :], (L, 1)).astype(np.float64)
    if orientation == "horizontal":
        return np.tile(vals[:, None], (1, L)).astype(np.float64)
    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def make_checkerboard(L: int, cell_size: int = 1) -> np.ndarray:
    y, x = np.indices((L, L))
    return ((((x // cell_size) + (y // cell_size)) % 2) * 2 - 1).astype(np.float64)


def make_patchwork(L: int) -> np.ndarray:
    half = L // 2
    image = np.empty((L, L), dtype=np.float64)
    image[:half, :half] = make_stripes(half, period=8, orientation="vertical")
    image[:half, half:] = make_stripes(half, period=8, orientation="horizontal")
    image[half:, :half] = make_checkerboard(half, cell_size=1)
    image[half:, half:] = make_checkerboard(half, cell_size=16)
    return image


def square_wave_grid(L: int, cell_size: int, mode: str) -> np.ndarray:
    y, x = np.indices((L, L))

    if mode == "x":
        bits = (x // cell_size) % 2
    elif mode == "y":
        bits = (y // cell_size) % 2
    elif mode == "xy":
        bits = ((x // cell_size) + (y // cell_size)) % 2
    else:
        raise ValueError(mode)

    return (2 * bits - 1).astype(np.float64)


def make_nested_dyadic(L: int) -> np.ndarray:
    """
    Deterministic nested dyadic benchmark image.

    The image is built as an additive Haar-like cascade. At each level,
    every dyadic block receives an asymmetric 2x2 motif

        + +
        + -

    with a level-dependent amplitude. This keeps organized detail
    contributions alive along the same local RG histories at several scales.
    """
    image = np.zeros((L, L), dtype=np.float64)
    levels = max(1, int(np.log2(L)))
    decay = 0.65

    for level in range(levels):
        block = L // (2 ** level)
        if block < 2:
            break

        half = block // 2
        amp = decay ** level

        if level % 3 == 0:
            motif = ((1.0, 1.0), (1.0, -1.0))
        elif level % 3 == 1:
            motif = ((1.0, -1.0), (1.0, 1.0))
        else:
            motif = ((1.0, 1.0), (-1.0, 1.0))

        for i in range(0, L, block):
            for j in range(0, L, block):
                image[i : i + half, j : j + half] += amp * motif[0][0]
                image[i : i + half, j + half : j + block] += amp * motif[0][1]
                image[i + half : i + block, j : j + half] += amp * motif[1][0]
                image[i + half : i + block, j + half : j + block] += amp * motif[1][1]

    image -= np.mean(image)
    max_abs = np.max(np.abs(image))

    if max_abs > 0:
        image /= max_abs

    return image


def make_spectral_fractal_binary(L: int, beta: float = 2.5, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(L, L))
    F = np.fft.fftn(noise)

    ky = np.fft.fftfreq(L)[:, None]
    kx = np.fft.fftfreq(L)[None, :]
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = np.inf

    amp = k2 ** (-beta / 4.0)
    field = np.fft.ifftn(F * amp).real
    field -= field.mean()
    field /= field.std() + 1e-15

    threshold = np.median(field)
    return np.where(field >= threshold, 1.0, -1.0).astype(np.float64)


def make_noise(L: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(L, L)).astype(np.float64)


def generate_benchmark_images(L: int, seed: int) -> dict[str, np.ndarray]:
    return {
        "stripes": make_stripes(L, period=16, orientation="vertical"),
        "checkerboard": make_checkerboard(L, cell_size=1),
        "patchwork": make_patchwork(L),
        "nested_dyadic": make_nested_dyadic(L),
        "fractal": make_spectral_fractal_binary(L, beta=2.5, seed=seed),
        "noise": make_noise(L, seed=seed),
        "wavy_stripes": make_wavy_stripes(
            size=L,
            stripe_period=64.0,
            wave_amplitude=24.0,
            wave_period=256.0,
            threshold=0.0,
            binary=True,
        ),
    }


def summarize_available_profiles(profiles: dict[str, np.ndarray], qmap_mean: float | None) -> dict[str, float]:
    summary: dict[str, float] = {}

    for name, profile in profiles.items():
        summary[f"{name}_total"] = float(np.sum(profile))
        summary[f"H_{name}"] = profile_entropy(profile)
        summary[f"S_{name}"] = entropic_complexity(profile)

    if "Q" in profiles:
        summary["Q_mean"] = float(np.mean(profiles["Q"])) if len(profiles["Q"]) else 0.0
    if "D" in profiles:
        summary["D_mean"] = float(np.mean(profiles["D"])) if len(profiles["D"]) else 0.0
    if qmap_mean is not None:
        summary["Qmap_mean"] = float(qmap_mean)

    return summary


def analyze_array(
    image: np.ndarray,
    n_steps: int | None = None,
    connectivity: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    profiles: dict[str, np.ndarray] = {}

    profiles["C"] = complexity_profile(image, block_size=2, n_steps=n_steps)
    profiles["Q"] = local_orientation_coherence_profile(image, n_steps=n_steps)
    profiles["D"] = orientation_entropy_profile(image, n_steps=n_steps)
    profiles["O"] = organized_profile(profiles["C"], profiles["Q"])
    profiles["Odiv"] = orientation_diverse_organized_profile(
        profiles["C"],
        profiles["Q"],
        profiles["D"],
    )

    qmap_mean: float | None = None

    try:
        E = haar_channel_energy_profile(image, n_steps=n_steps)
        lifted_E = lifted_haar_channel_energy_profile(image, n_steps=n_steps)
        profiles["Jglob"] = scale_orientation_entropy_profile(E, profiles["Q"])
        profiles["Jloc"] = local_scale_orientation_entropy_profile(lifted_E, profiles["Q"])

        lifted_q = lifted_local_orientation_coherence_profile(
            image,
            n_steps=n_steps,
            connectivity=connectivity,
        )
        profiles["JlocQ"] = local_scale_orientation_entropy_profile_with_local_q(
            lifted_E,
            lifted_q,
        )
        qmap_mean = float(np.mean(lifted_q)) if lifted_q.size else 0.0
    except (ImportError, AttributeError, NameError):
        pass

    summary = summarize_available_profiles(profiles, qmap_mean=qmap_mean)
    return profiles, summary


def selected_metric_name(requested_metric: str, available_metrics: set[str]) -> str:
    fallbacks = {
        "JlocQ": ["JlocQ", "Jloc", "Odiv", "C"],
        "Jloc": ["Jloc", "Odiv", "C"],
        "Jglob": ["Jglob", "Jloc", "Odiv", "C"],
        "Odiv": ["Odiv", "C"],
        "C": ["C"],
    }

    for candidate in fallbacks.get(requested_metric, [requested_metric, "JlocQ", "Jloc", "Odiv", "C"]):
        if candidate in available_metrics:
            return candidate

    raise ValueError("no usable metric available")


def save_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    columns: list[str] = ["name"]
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_profiles_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    columns: list[str] = ["name", "k"]
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_benchmark_panel(
    path: Path,
    images: dict[str, np.ndarray],
    summaries: dict[str, dict[str, float]],
    metric: str,
    diagnostics_level: str,
) -> None:
    names = list(images.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.9))

    for ax, name in zip(axes, names):
        ax.imshow(images[name], cmap="gray", vmin=-1.0, vmax=1.0, interpolation="nearest")
        if diagnostics_level == "full":
            title = (
                f"{name.replace('_', ' ')}\n"
                f"{display_name('C')}={summaries[name]['C_total']:.3g}, "
                f"{display_name('Jglob')}={summaries[name]['Jglob_total']:.3g}, "
                f"{display_name('JlocQ')}={summaries[name]['JlocQ_total']:.3g}"
            )
        else:
            title = f"{name.replace('_', ' ')}\n{display_name(metric)}={summaries[name][f'{metric}_total']:.3g}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_profile_plot(
    path: Path,
    profiles_by_name: dict[str, dict[str, np.ndarray]],
    metric: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    for name, profiles in profiles_by_name.items():
        k = np.arange(len(profiles[metric]))
        axes[0].plot(k, profiles[metric], marker="o", label=name)

    axes[0].set_title(f"{metric}_k benchmark profiles")
    axes[0].set_xlabel("Scale index k")
    axes[0].set_ylabel(metric)
    axes[0].legend()

    for name, profiles in profiles_by_name.items():
        k = np.arange(len(profiles["C"]))
        axes[1].plot(k, profiles["C"], marker="o", label=name)

    axes[1].set_title("C_k benchmark profiles")
    axes[1].set_xlabel("Scale index k")
    axes[1].set_ylabel("C")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_rankings(
    summaries: dict[str, dict[str, float]],
    metric: str,
) -> None:
    ranking = sorted(
        ((name, values[f"{metric}_total"]) for name, values in summaries.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print(f"Ranking by {metric}_total:")
    for name, value in ranking:
        print(f"  {name:<14s} {value:.12g}")


def print_warnings(
    summaries: dict[str, dict[str, float]],
    metric: str,
) -> None:
    fractal = summaries["fractal"][f"{metric}_total"]
    noise = summaries["noise"][f"{metric}_total"]
    nested = summaries["nested_dyadic"][f"{metric}_total"]
    patchwork = summaries["patchwork"][f"{metric}_total"]

    if noise > 1.5 * fractal:
        print(
            f"Warning: noise {metric}_total={noise:.6g} exceeds 1.5 * fractal "
            f"{metric}_total={fractal:.6g}"
        )

    if patchwork > 1.5 * nested:
        print(
            f"Warning: patchwork {metric}_total={patchwork:.6g} exceeds 1.5 * nested_dyadic "
            f"{metric}_total={nested:.6g}"
        )


def parse_n_steps(value: str) -> int | None:
    if value == "auto":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and benchmark synthetic multiscale images."
    )
    parser.add_argument("--size", type=int, default=512, help="Image size. Must be a power of two.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for stochastic generators.")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_toys"))
    parser.add_argument("--metric", default="JlocQ", help="Metric shown in the panel title.")
    parser.add_argument("--n-steps", default="auto", help="'auto' or an integer.")
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=4)
    parser.add_argument("--save-images", action="store_true", help="Save PNG versions of the generated arrays.")
    parser.add_argument(
        "--diagnostics-level",
        choices=["core", "full"],
        default="core",
        help="Default summary detail level. Default: core.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not is_power_of_two(args.size):
        raise ValueError("--size must be a power of two")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = parse_n_steps(args.n_steps)

    images = generate_benchmark_images(args.size, seed=args.seed)

    for name, image in images.items():
        if image.shape != (args.size, args.size):
            raise ValueError(f"{name} has wrong shape {image.shape}")
        if not np.isfinite(image).all():
            raise ValueError(f"{name} contains non-finite values")
        if image.min() < -1.0 or image.max() > 1.0:
            raise ValueError(f"{name} has values outside [-1, 1]")

    profiles_by_name: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, float]] = {}
    summary_rows: list[dict[str, float | str]] = []
    profile_rows: list[dict[str, float | str]] = []

    for name, image in images.items():
        profiles, summary = analyze_array(
            image,
            n_steps=n_steps,
            connectivity=args.connectivity,
        )
        profiles_by_name[name] = profiles
        summaries[name] = summary

        summary_row: dict[str, float | str] = {"name": name}
        summary_row.update(summary)
        summary_rows.append(summary_row)

        n_profile_steps = len(profiles["C"])
        for k in range(n_profile_steps):
            row: dict[str, float | str] = {"name": name, "k": k}
            for metric_name, profile in profiles.items():
                row[metric_name] = float(profile[k])
            profile_rows.append(row)

        if args.save_images:
            save_image(image, args.out_dir / f"{name}.png")

    available_metrics = {key[:-6] for key in summaries["stripes"].keys() if key.endswith("_total")}
    metric = selected_metric_name(args.metric, available_metrics)

    print(f"Selected metric for panel: {metric}")
    print(f"Image size: {args.size}")
    print(f"RG steps: {n_steps if n_steps is not None else max_steps(args.size, block_size=2)}")
    print(f"Connectivity: {args.connectivity}")

    print()
    print("Summary totals:")
    for name in images:
        if args.diagnostics_level == "full":
            payload = " ".join(
                f"{key}={value:.6g}"
                for key, value in summaries[name].items()
                if key.endswith("_total")
            )
        else:
            payload = (
                f"{display_name('C')}={summaries[name]['C_total']:.6g} "
                f"{display_name('Jglob')}={summaries[name]['Jglob_total']:.6g} "
                f"{display_name('JlocQ')}={summaries[name]['JlocQ_total']:.6g}"
            )
        print(f"  {name:<14s} {payload}")

    print_rankings(summaries, metric)
    print_warnings(summaries, metric)

    checker_total = summaries["checkerboard"]["C_total"]
    if abs(checker_total - 0.5) >= 1e-8:
        print(f"Checkerboard C_total diagnostic: {checker_total:.12g}")

    save_summary_csv(args.out_dir / "benchmark_summary.csv", summary_rows)
    save_profiles_csv(args.out_dir / "benchmark_profiles.csv", profile_rows)
    save_benchmark_panel(
        args.out_dir / "benchmark_panel.png",
        images,
        summaries,
        metric,
        diagnostics_level=args.diagnostics_level,
    )
    save_profile_plot(args.out_dir / "benchmark_profiles.png", profiles_by_name, metric)

    print()
    print(f"Saved summary CSV: {args.out_dir / 'benchmark_summary.csv'}")
    print(f"Saved profiles CSV: {args.out_dir / 'benchmark_profiles.csv'}")
    print(f"Saved panel plot: {args.out_dir / 'benchmark_panel.png'}")
    print(f"Saved profile plot: {args.out_dir / 'benchmark_profiles.png'}")


if __name__ == "__main__":
    main()
