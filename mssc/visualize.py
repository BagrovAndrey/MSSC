from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .complexity import coarse_grain, complexity_profile, max_steps, validate_image


def rg_layers(
    image: np.ndarray,
    block_size: int = 2,
    n_steps: int | None = None,
) -> list[np.ndarray]:
    """
    Return the full stack of RG layers:
        [image_0, image_1, image_2, ...]

    image_0 is the original image.
    image_{k+1} is obtained by coarse-graining image_k.
    """
    validate_image(image)

    image = image.astype(float, copy=False)

    L = image.shape[0]
    max_n = max_steps(L, block_size=block_size)

    if n_steps is None:
        n_steps = max_n

    if n_steps > max_n:
        raise ValueError(
            f"n_steps={n_steps} is too large for image size {L}; "
            f"maximum is {max_n}"
        )

    layers = [image]
    current = image

    for _ in range(n_steps):
        current = coarse_grain(current, block_size=block_size)
        layers.append(current)

    return layers


def _to_display_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an internal image array to something imshow can display.

    Works for:
      - [0, 1]
      - [-1, 1]
      - arbitrary scalar arrays
    """
    arr = np.asarray(image, dtype=float)

    if arr.ndim == 3:
        # Assume RGB/vector image. For display, clip after mapping if needed.
        if arr.min() < 0:
            arr = 0.5 * (arr + 1.0)
        return np.clip(arr, 0.0, 1.0)

    # Scalar image.
    vmin = arr.min()
    vmax = arr.max()

    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    return arr


def plot_rg_layers(
    image: np.ndarray,
    block_size: int = 2,
    n_steps: int | None = None,
    path: str | Path | None = None,
) -> None:
    """
    Visualize the RG/coarse-graining layers.

    The layers are displayed with nearest-neighbor interpolation so that
    block structure is visible.
    """
    layers = rg_layers(
        image,
        block_size=block_size,
        n_steps=n_steps,
    )

    profile = complexity_profile(
        image,
        block_size=block_size,
        n_steps=len(layers) - 1,
    )

    ncols = len(layers)
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.2))

    if ncols == 1:
        axes = [axes]

    for k, (ax, layer) in enumerate(zip(axes, layers)):
        ax.imshow(_to_display_image(layer), interpolation="nearest")

        if k == 0:
            title = f"k=0\n{layer.shape[0]}x{layer.shape[1]}"
        else:
            title = f"k={k}\n{layer.shape[0]}x{layer.shape[1]}\nC_{k-1}={profile[k-1]:.3g}"

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    else:
        plt.show()
