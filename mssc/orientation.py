from __future__ import annotations

import numpy as np

from mssc.complexity import coarse_grain, max_steps, validate_image


def haar_detail_vectors(image: np.ndarray) -> np.ndarray:
    """
    Compute local Haar-like detail vectors for non-overlapping 2x2 blocks.

    For a scalar block

        [[a, b],
         [c, d]]

    the three detail channels are

        h_x  = (a + c - b - d) / 4
        h_y  = (a + b - c - d) / 4
        h_xy = (a - b - c + d) / 4

    For grayscale input, the output has shape (L/2, L/2, 3).
    For RGB/vector input, the output has shape (L/2, L/2, 3*C).
    """
    validate_image(image)

    img = np.asarray(image, dtype=float)
    L = img.shape[0]

    if L % 2 != 0:
        raise ValueError("image size must be divisible by 2")

    a = img[0::2, 0::2]
    b = img[0::2, 1::2]
    c = img[1::2, 0::2]
    d = img[1::2, 1::2]

    h_x = (a + c - b - d) / 4.0
    h_y = (a + b - c - d) / 4.0
    h_xy = (a - b - c + d) / 4.0

    if img.ndim == 2:
        return np.stack([h_x, h_y, h_xy], axis=-1)

    # RGB/vector image: concatenate Haar channels for all color channels.
    # Shape before reshape: (L/2, L/2, 3, C)
    h = np.stack([h_x, h_y, h_xy], axis=-2)
    return h.reshape(h.shape[0], h.shape[1], -1)


def local_orientation_coherence_from_h(
    h: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Energy-weighted local nematic coherence of Haar-detail directions.

    The sign of the Haar vector is ignored via (u_B dot u_B')^2.
    Random directions give approximately zero after subtracting the
    isotropic baseline 1/d.
    """
    h = np.asarray(h, dtype=float)

    if h.ndim != 3:
        raise ValueError("h must have shape (n, n, d)")

    energy = np.sum(h * h, axis=-1)
    total_energy = float(np.sum(energy))

    if total_energy <= eps:
        return 0.0

    d = h.shape[-1]
    unit = h / np.sqrt(energy + eps)[..., None]

    values = []
    weights = []

    # Horizontal neighboring pairs.
    if h.shape[1] > 1:
        dot_x = np.sum(unit[:, :-1, :] * unit[:, 1:, :], axis=-1)
        w_x = np.sqrt(energy[:, :-1] * energy[:, 1:])
        values.append(dot_x * dot_x)
        weights.append(w_x)

    # Vertical neighboring pairs.
    if h.shape[0] > 1:
        dot_y = np.sum(unit[:-1, :, :] * unit[1:, :, :], axis=-1)
        w_y = np.sqrt(energy[:-1, :] * energy[1:, :])
        values.append(dot_y * dot_y)
        weights.append(w_y)

    if not values:
        return 0.0

    value = np.concatenate([v.ravel() for v in values])
    weight = np.concatenate([w.ravel() for w in weights])

    weight_sum = float(np.sum(weight))
    if weight_sum <= eps:
        return 0.0

    mean_dot2 = float(np.sum(weight * value) / weight_sum)

    baseline = 1.0 / d
    coherence = (mean_dot2 - baseline) / (1.0 - baseline)

    return float(max(coherence, 0.0))


def local_orientation_coherence(
    image: np.ndarray,
    eps: float = 1e-12,
) -> float:
    h = haar_detail_vectors(image)
    return local_orientation_coherence_from_h(h, eps=eps)


def orientation_entropy_from_h(
    h: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Energy-weighted entropy of Haar-detail directions.

    Construct the orientation tensor

        M = sum_B e_B u_B u_B^T / sum_B e_B,

    where

        e_B = |h_B|^2,
        u_B = h_B / |h_B|.

    The normalized entropy of the eigenvalues of M is returned.

    Interpretation:
        0: all strong details point in essentially one Haar direction.
        1: strong details occupy Haar-detail space isotropically.
    """
    h = np.asarray(h, dtype=float)

    if h.ndim != 3:
        raise ValueError("h must have shape (n, n, d)")

    d = h.shape[-1]
    flat = h.reshape(-1, d)

    energy = np.sum(flat * flat, axis=-1)
    total_energy = float(np.sum(energy))

    if total_energy <= eps:
        return 0.0

    # Because e_B u_B u_B^T = h_B h_B^T, this is the normalized
    # second-moment tensor of Haar-detail vectors.
    M = (flat.T @ flat) / total_energy

    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.clip(eigvals, 0.0, None)

    norm = float(np.sum(eigvals))
    if norm <= eps:
        return 0.0

    eigvals = eigvals / norm
    eigvals = eigvals[eigvals > eps]

    if len(eigvals) == 0 or d <= 1:
        return 0.0

    entropy = -float(np.sum(eigvals * np.log(eigvals)))
    entropy /= np.log(d)

    return float(entropy)


def orientation_entropy(
    image: np.ndarray,
    eps: float = 1e-12,
) -> float:
    h = haar_detail_vectors(image)
    return orientation_entropy_from_h(h, eps=eps)


def haar_channel_energy(image: np.ndarray) -> np.ndarray:
    """
    Per-channel Haar detail energies for one RG layer.

    Returns E_alpha = 0.5 * mean_B h_alpha^2 over non-overlapping 2x2 blocks.
    For grayscale images this has length 3; for RGB/vector images it has
    length 3 * C after channel concatenation.
    """
    h = haar_detail_vectors(image)
    return 0.5 * np.mean(h * h, axis=(0, 1))


def haar_channel_energy_profile(
    image: np.ndarray,
    n_steps: int | None = None,
) -> np.ndarray:
    """
    Per-scale Haar detail channel energies E_{k,alpha}.

    The returned array has shape (n_steps, d), where d is the Haar-detail
    dimension of one RG layer.
    """
    validate_image(image)

    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    profile = []

    for _ in range(n_steps):
        profile.append(haar_channel_energy(current))
        current = coarse_grain(current, block_size=2)

    return np.asarray(profile, dtype=float)


def local_orientation_coherence_profile(
    image: np.ndarray,
    n_steps: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute Q_k for each RG layer f_k.

    This is currently defined only for 2x2 block coarse-graining.
    """
    validate_image(image)

    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    profile = []

    for _ in range(n_steps):
        profile.append(local_orientation_coherence(current, eps=eps))
        current = coarse_grain(current, block_size=2)

    return np.asarray(profile, dtype=float)


def orientation_entropy_profile(
    image: np.ndarray,
    n_steps: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute D_k = H_orient,k for each RG layer f_k.

    D_k measures diversity of strong local Haar-detail directions.
    """
    validate_image(image)

    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    profile = []

    for _ in range(n_steps):
        profile.append(orientation_entropy(current, eps=eps))
        current = coarse_grain(current, block_size=2)

    return np.asarray(profile, dtype=float)


def organized_profile(
    complexity_profile: np.ndarray,
    coherence_profile: np.ndarray,
) -> np.ndarray:
    """
    Old diagnostic: ordered contrast energy.

        O_k = C_k max(Q_k, 0)
    """
    C = np.asarray(complexity_profile, dtype=float)
    Q = np.asarray(coherence_profile, dtype=float)

    if C.shape != Q.shape:
        raise ValueError("complexity_profile and coherence_profile must have same shape")

    return C * np.maximum(Q, 0.0)


def orientation_diverse_organized_profile(
    complexity_profile: np.ndarray,
    coherence_profile: np.ndarray,
    orientation_entropy_profile: np.ndarray,
) -> np.ndarray:
    """
    Orientation-diverse organized complexity.

        O_div,k = C_k max(Q_k, 0) D_k

    where D_k is the normalized entropy of Haar-detail directions.
    """
    C = np.asarray(complexity_profile, dtype=float)
    Q = np.asarray(coherence_profile, dtype=float)
    D = np.asarray(orientation_entropy_profile, dtype=float)

    if C.shape != Q.shape or C.shape != D.shape:
        raise ValueError("C, Q, and D profiles must have the same shape")

    return C * np.maximum(Q, 0.0) * D


def scale_orientation_entropy_profile(
    channel_energy_profile: np.ndarray,
    coherence_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Entropy contribution profile over the joint scale-orientation distribution.

    For channel energies E_{k,alpha} and local coherence Q_k, define

        W_{k,alpha} = max(Q_k, 0) E_{k,alpha}.

    This function returns

        J_k = - sum_alpha W_{k,alpha} log(W_{k,alpha} / W_tot),

    with zero-weight entries contributing zero.
    """
    E = np.asarray(channel_energy_profile, dtype=float)
    Q = np.asarray(coherence_profile, dtype=float)

    if E.ndim != 2:
        raise ValueError("channel_energy_profile must have shape (n_steps, d)")
    if Q.ndim != 1:
        raise ValueError("coherence_profile must have shape (n_steps,)")
    if E.shape[0] != Q.shape[0]:
        raise ValueError("channel_energy_profile and coherence_profile must agree on n_steps")

    W = E * np.maximum(Q, 0.0)[:, None]
    W_tot = float(np.sum(W))

    if W_tot <= eps:
        return np.zeros(E.shape[0], dtype=float)

    P = W / W_tot
    J_terms = np.zeros_like(W)
    mask = P > eps
    J_terms[mask] = -W[mask] * np.log(P[mask])

    return np.sum(J_terms, axis=1)
