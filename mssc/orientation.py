from __future__ import annotations

import numpy as np

from .complexity import coarse_grain, max_steps, validate_image


def haar_detail_vectors(image: np.ndarray) -> np.ndarray:
    """
    Compute local Haar detail vectors for all 2x2 blocks.

    For scalar image:
        output shape: (L/2, L/2, 3)

    For vector/RGB image:
        output shape: (L/2, L/2, 3*C)

    The three Haar channels are:
        h_x  : left-right contrast
        h_y  : top-bottom contrast
        h_xy : diagonal/checkerboard contrast
    """
    validate_image(image)

    L = image.shape[0]
    if L % 2 != 0:
        raise ValueError("image size must be divisible by 2")

    n = L // 2
    img = np.asarray(image, dtype=float)

    if img.ndim == 2:
        blocks = img.reshape(n, 2, n, 2).swapaxes(1, 2)

        a = blocks[:, :, 0, 0]
        b = blocks[:, :, 0, 1]
        c = blocks[:, :, 1, 0]
        d = blocks[:, :, 1, 1]

        h_x = (a + c - b - d) / 4.0
        h_y = (a + b - c - d) / 4.0
        h_xy = (a - b - c + d) / 4.0

        return np.stack([h_x, h_y, h_xy], axis=-1)

    channels = img.shape[-1]
    blocks = img.reshape(n, 2, n, 2, channels).swapaxes(1, 2)

    a = blocks[:, :, 0, 0, :]
    b = blocks[:, :, 0, 1, :]
    c = blocks[:, :, 1, 0, :]
    d = blocks[:, :, 1, 1, :]

    h_x = (a + c - b - d) / 4.0
    h_y = (a + b - c - d) / 4.0
    h_xy = (a - b - c + d) / 4.0

    h = np.stack([h_x, h_y, h_xy], axis=-2)
    return h.reshape(n, n, 3 * channels)


def local_orientation_coherence_from_h(
    h: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute energy-weighted nearest-neighbor coherence of local Haar vectors.

    h has shape (n, n, d), where d=3 for scalar images and d=3*C for RGB.

    Returns Q in approximately [-1, 1].
    If there is essentially no detail energy, returns 0.
    """
    if h.ndim != 3:
        raise ValueError("h must have shape (n, n, d)")

    energy = np.sum(h * h, axis=-1)

    if float(energy.sum()) <= eps:
        return 0.0

    norm = np.sqrt(energy + eps)
    unit = h / norm[..., None]

    dots = []
    weights = []

    # Horizontal neighboring block pairs.
    dot_x = np.sum(unit[:, :-1, :] * unit[:, 1:, :], axis=-1)
    w_x = np.sqrt(energy[:, :-1] * energy[:, 1:])

    dots.append(dot_x)
    weights.append(w_x)

    # Vertical neighboring block pairs.
    dot_y = np.sum(unit[:-1, :, :] * unit[1:, :, :], axis=-1)
    w_y = np.sqrt(energy[:-1, :] * energy[1:, :])

    dots.append(dot_y)
    weights.append(w_y)

    numerator = 0.0
    denominator = 0.0

    for dot, weight in zip(dots, weights):
        numerator += float(np.sum(weight * dot))
        denominator += float(np.sum(weight))

    if denominator <= eps:
        return 0.0

    return numerator / denominator


def local_orientation_coherence(
    image: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute Q for one image scale.
    """
    h = haar_detail_vectors(image)
    return local_orientation_coherence_from_h(h, eps=eps)


def local_orientation_coherence_profile(
    image: np.ndarray,
    n_steps: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute Q_k for all RG scales.

    Currently this is defined for the same 2x2 block coarse-graining
    as the minimal MSSC implementation.

    Q_k is computed from the Haar detail vectors inside 2x2 blocks
    of f_k, then f_k is coarse-grained to f_{k+1}.
    """
    validate_image(image)

    img = np.asarray(image, dtype=float)
    L = img.shape[0]

    max_n = max_steps(L, block_size=2)

    if n_steps is None:
        n_steps = max_n

    if n_steps > max_n:
        raise ValueError(
            f"n_steps={n_steps} is too large for image size {L}; "
            f"maximum is {max_n}"
        )

    values = []
    current = img

    for _ in range(n_steps):
        values.append(local_orientation_coherence(current, eps=eps))
        current = coarse_grain(current, block_size=2)

    return np.asarray(values, dtype=float)


def organized_profile(
    complexity_profile: np.ndarray,
    coherence_profile: np.ndarray,
) -> np.ndarray:
    """
    Compute O_k = C_k * max(Q_k, 0).
    """
    C = np.asarray(complexity_profile, dtype=float)
    Q = np.asarray(coherence_profile, dtype=float)

    if C.shape != Q.shape:
        raise ValueError("complexity_profile and coherence_profile must have the same shape")

    return C * np.maximum(Q, 0.0)
