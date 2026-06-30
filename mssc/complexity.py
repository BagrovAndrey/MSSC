from __future__ import annotations

import numpy as np


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (L, L) or (L, L, C)")

    if image.shape[0] != image.shape[1]:
        raise ValueError("image must be square")


def max_steps(size: int, block_size: int = 2) -> int:
    if block_size <= 1:
        raise ValueError("block_size must be larger than 1")

    n = 0
    while size % block_size == 0:
        size //= block_size
        n += 1

    return n


def coarse_grain(image: np.ndarray, block_size: int = 2) -> np.ndarray:
    validate_image(image)

    L = image.shape[0]

    if L % block_size != 0:
        raise ValueError(f"image size {L} is not divisible by block_size={block_size}")

    n = L // block_size

    if image.ndim == 2:
        blocks = image.reshape(n, block_size, n, block_size)
        return blocks.mean(axis=(1, 3))

    channels = image.shape[-1]
    blocks = image.reshape(n, block_size, n, block_size, channels)
    return blocks.mean(axis=(1, 3))


def upscale_nearest(image: np.ndarray, block_size: int = 2) -> np.ndarray:
    validate_image(image)

    out = np.repeat(image, block_size, axis=0)
    out = np.repeat(out, block_size, axis=1)

    return out


def partial_complexity(
    fine: np.ndarray,
    coarse: np.ndarray,
    block_size: int = 2,
) -> float:
    upscaled = upscale_nearest(coarse, block_size=block_size)

    if upscaled.shape != fine.shape:
        raise ValueError(
            f"shape mismatch: fine has shape {fine.shape}, "
            f"upscaled coarse has shape {upscaled.shape}"
        )

    diff = fine - upscaled

    if diff.ndim == 3:
        return float(0.5 * np.mean(np.sum(diff * diff, axis=-1)))

    return float(0.5 * np.mean(diff * diff))


def complexity_profile(
    image: np.ndarray,
    block_size: int = 2,
    n_steps: int | None = None,
) -> np.ndarray:
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

    profile = []
    current = image

    for _ in range(n_steps):
        coarse = coarse_grain(current, block_size=block_size)
        Ck = partial_complexity(current, coarse, block_size=block_size)
        profile.append(Ck)
        current = coarse

    return np.array(profile)


def total_complexity(
    image: np.ndarray,
    block_size: int = 2,
    n_steps: int | None = None,
    skip_first: bool = False,
) -> float:
    profile = complexity_profile(
        image,
        block_size=block_size,
        n_steps=n_steps,
    )

    if skip_first:
        return float(profile[1:].sum())

    return float(profile.sum())
