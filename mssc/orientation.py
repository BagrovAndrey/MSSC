from __future__ import annotations

import numpy as np

from mssc.complexity import coarse_grain, max_steps, validate_image


def _neighbor_shifts(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(0, 1), (1, 0)]
    if connectivity == 8:
        return [(0, 1), (1, 0), (1, 1), (1, -1)]
    raise ValueError("connectivity must be 4 or 8")


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


def local_orientation_coherence_map_from_h(
    h: np.ndarray,
    connectivity: int = 4,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Block-level nematic coherence map from Haar-detail vectors.

    The returned map has one value per 2x2 block. Each block is compared to
    its nearest neighbors using the same sign-insensitive alignment used in the
    scale-global coherence observable.
    """
    h = np.asarray(h, dtype=float)

    if h.ndim != 3:
        raise ValueError("h must have shape (n, n, d)")

    nrows, ncols, d = h.shape
    qsum = np.zeros((nrows, ncols), dtype=float)
    wsum = np.zeros((nrows, ncols), dtype=float)

    if nrows == 0 or ncols == 0 or (nrows == 1 and ncols == 1):
        return np.zeros((nrows, ncols), dtype=float)

    energy = np.sum(h * h, axis=-1)
    unit = np.zeros_like(h, dtype=float)
    valid = energy > eps
    unit[valid] = h[valid] / np.sqrt(energy[valid])[:, None]

    for dr, dc in _neighbor_shifts(connectivity):
        if dr >= 0:
            src_r = slice(0, nrows - dr)
            dst_r = slice(dr, nrows)
        else:
            src_r = slice(-dr, nrows)
            dst_r = slice(0, nrows + dr)

        if dc >= 0:
            src_c = slice(0, ncols - dc)
            dst_c = slice(dc, ncols)
        else:
            src_c = slice(-dc, ncols)
            dst_c = slice(0, ncols + dc)

        if (src_r.stop - src_r.start) <= 0 or (src_c.stop - src_c.start) <= 0:
            continue

        unit_a = unit[src_r, src_c]
        unit_b = unit[dst_r, dst_c]
        energy_a = energy[src_r, src_c]
        energy_b = energy[dst_r, dst_c]

        weight = np.sqrt(energy_a * energy_b)
        dot = np.sum(unit_a * unit_b, axis=-1)
        value = dot * dot

        qsum[src_r, src_c] += weight * value
        qsum[dst_r, dst_c] += weight * value
        wsum[src_r, src_c] += weight
        wsum[dst_r, dst_c] += weight

    qmap = np.zeros((nrows, ncols), dtype=float)
    valid_blocks = wsum > eps

    if not np.any(valid_blocks):
        return qmap

    mean_dot2 = np.zeros_like(qmap)
    mean_dot2[valid_blocks] = qsum[valid_blocks] / wsum[valid_blocks]

    baseline = 1.0 / d
    qmap[valid_blocks] = (mean_dot2[valid_blocks] - baseline) / (1.0 - baseline)

    return np.maximum(qmap, 0.0)


def local_orientation_coherence_map(
    image: np.ndarray,
    connectivity: int = 4,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Return a block-level map of local orientation coherence.

    The result has shape (L/2, L/2) for one RG layer of shape (L, L) or
    (L, L, C).
    """
    h = haar_detail_vectors(image)
    return local_orientation_coherence_map_from_h(
        h,
        connectivity=connectivity,
        eps=eps,
    )


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


def haar_channel_energy_map(image: np.ndarray) -> np.ndarray:
    """
    Local Haar-channel energies on the RG-block grid for one image layer.

    Returns E_{B,alpha} = 0.5 * h_{B,alpha}^2 with shape (L/2, L/2, d).
    """
    h = haar_detail_vectors(image)
    return 0.5 * h * h


def detail_energy_map_from_h(h: np.ndarray) -> np.ndarray:
    """
    Return native block detail energy e_B = |h_B|^2.

    This is defined on the native 2x2 block grid of one RG layer and is used
    by the orientation-blind energy-coherence diagnostic.
    """
    h = np.asarray(h, dtype=float)

    if h.ndim != 3:
        raise ValueError("h must have shape (n, n, d)")

    return np.sum(h * h, axis=-1)


def detail_energy_map(image: np.ndarray) -> np.ndarray:
    """
    Return native block detail energy e_B = |h_B|^2 for one RG layer.
    """
    h = haar_detail_vectors(image)
    return detail_energy_map_from_h(h)


def detail_energy_coherence_from_map(
    energy_map: np.ndarray,
    eps: float = 1e-15,
) -> tuple[float, float, bool]:
    """
    Pearson nearest-neighbor correlation of native block detail-energy
    fluctuations.

    The correlation is computed on the native RG block grid before any lifting
    back to original-image coordinates. It therefore measures spatial
    organization already present on the RG layer, not artificial correlation
    introduced by nearest-neighbor attribution.

    Returns
    -------
    Qenergy : float
        max(rho, 0) when the Pearson correlation is defined, else NaN.
    variance : float
        Variance of the concatenated pair arrays used for the Pearson
        computation.
    defined : bool
        Whether the Pearson correlation is defined at this scale.
    """
    e = np.asarray(energy_map, dtype=float)

    if e.ndim != 2:
        raise ValueError("energy_map must have shape (n, n)")

    if e.size == 0:
        return float("nan"), 0.0, False

    pairs_a = []
    pairs_b = []

    if e.shape[1] > 1:
        pairs_a.append(e[:, :-1].ravel())
        pairs_b.append(e[:, 1:].ravel())

    if e.shape[0] > 1:
        pairs_a.append(e[:-1, :].ravel())
        pairs_b.append(e[1:, :].ravel())

    if not pairs_a:
        return float("nan"), 0.0, False

    a = np.concatenate(pairs_a)
    b = np.concatenate(pairs_b)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_energy = 0.5 * (mean_a + mean_b)

    if mean_energy <= eps:
        return float("nan"), 0.0, False

    da = a - mean_a
    db = b - mean_b
    var_a = float(np.mean(da * da))
    var_b = float(np.mean(db * db))
    variance = 0.5 * (var_a + var_b)

    if var_a <= eps or var_b <= eps:
        return float("nan"), variance, False

    cov = float(np.mean(da * db))
    rho = cov / np.sqrt(var_a * var_b)

    return float(max(rho, 0.0)), variance, True


def detail_energy_coherence_profile(
    image: np.ndarray,
    n_steps: int | None = None,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return native-grid detail-energy coherence diagnostics along the RG
    trajectory.

    Returns
    -------
    Ebar : ndarray
        Mean native block detail energy at each scale.
    Qenergy : ndarray
        Nonnegative Pearson nearest-neighbor correlation of block-energy
        fluctuations. Undefined entries are NaN.
    variance : ndarray
        Variance diagnostic for the native block-energy field.
    defined : ndarray
        Boolean mask indicating where Qenergy is defined.
    """
    validate_image(image)

    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    ebar = []
    qenergy = []
    variance = []
    defined = []

    for _ in range(n_steps):
        e_map = detail_energy_map(current)
        q_val, var_val, is_defined = detail_energy_coherence_from_map(e_map, eps=eps)
        ebar.append(float(np.mean(e_map)) if e_map.size else 0.0)
        qenergy.append(q_val)
        variance.append(var_val)
        defined.append(is_defined)
        current = coarse_grain(current, block_size=2)

    return (
        np.asarray(ebar, dtype=float),
        np.asarray(qenergy, dtype=float),
        np.asarray(variance, dtype=float),
        np.asarray(defined, dtype=bool),
    )


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


def lifted_haar_channel_energy_profile(
    image: np.ndarray,
    n_steps: int | None = None,
) -> np.ndarray:
    """
    Per-scale Haar-channel energies lifted back to the original image grid.

    For each RG layer f_k, the Haar-channel energy map on the 2x2 block grid is
    repeated back to original-image resolution using nearest-neighbor lifting.
    The output has shape (n_steps, L0, L0, d).
    """
    validate_image(image)

    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    original_size = image.shape[0]
    lifted_profile = []

    for k in range(n_steps):
        energy_map = haar_channel_energy_map(current)
        factor = 2 ** (k + 1)
        lifted = np.repeat(np.repeat(energy_map, factor, axis=0), factor, axis=1)

        if lifted.shape[:2] != (original_size, original_size):
            raise ValueError(
                "lifted Haar channel energy map has incorrect shape: "
                f"expected {(original_size, original_size)}, got {lifted.shape[:2]}"
            )

        lifted_profile.append(lifted)
        current = coarse_grain(current, block_size=2)

    return np.asarray(lifted_profile, dtype=float)


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


def local_orientation_coherence_map_profile(
    image: np.ndarray,
    n_steps: int | None = None,
    connectivity: int = 4,
    eps: float = 1e-15,
) -> list[np.ndarray]:
    """
    Return block-level local coherence maps along the RG trajectory.

    Entry k has shape (L_k/2, L_k/2), where L_k is the side length of the
    RG layer f_k.
    """
    validate_image(image)

    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    profile = []

    for _ in range(n_steps):
        profile.append(
            local_orientation_coherence_map(
                current,
                connectivity=connectivity,
                eps=eps,
            )
        )
        current = coarse_grain(current, block_size=2)

    return profile


def lifted_local_orientation_coherence_profile(
    image: np.ndarray,
    n_steps: int | None = None,
    connectivity: int = 4,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Return local coherence maps lifted to the original image grid.

    The output has shape (n_steps, L0, L0).
    """
    validate_image(image)

    if n_steps is None:
        n_steps = max_steps(image.shape[0], block_size=2)

    current = np.asarray(image, dtype=float)
    original_size = image.shape[0]
    lifted_profile = []

    for k in range(n_steps):
        qmap = local_orientation_coherence_map(
            current,
            connectivity=connectivity,
            eps=eps,
        )
        factor = 2 ** (k + 1)
        lifted = np.repeat(np.repeat(qmap, factor, axis=0), factor, axis=1)

        if lifted.shape != (original_size, original_size):
            raise ValueError(
                "lifted local orientation coherence map has incorrect shape: "
                f"expected {(original_size, original_size)}, got {lifted.shape}"
            )

        lifted_profile.append(lifted)
        current = coarse_grain(current, block_size=2)

    return np.asarray(lifted_profile, dtype=float)


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


def local_scale_orientation_entropy_profile(
    lifted_channel_energy_profile: np.ndarray,
    coherence_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Local/nested entropy contribution profile over scale-orientation channels.

    For lifted local ordered channel weights W_{k,alpha}(x), this returns

        Jloc_k = - mean_x sum_alpha W_{k,alpha}(x) log(W_{k,alpha}(x) / W_tot(x)),

    where W_tot(x) sums over all scales and Haar channels at x.
    """
    E = np.asarray(lifted_channel_energy_profile, dtype=float)
    Q = np.asarray(coherence_profile, dtype=float)

    if E.ndim != 4:
        raise ValueError("lifted_channel_energy_profile must have shape (n_steps, L, L, d)")
    if Q.ndim != 1:
        raise ValueError("coherence_profile must have shape (n_steps,)")
    if E.shape[0] != Q.shape[0]:
        raise ValueError("lifted_channel_energy_profile and coherence_profile must agree on n_steps")

    W = E * np.maximum(Q, 0.0)[:, None, None, None]
    Wtot = np.sum(W, axis=(0, 3))

    Jloc = np.zeros(E.shape[0], dtype=float)
    valid_points = Wtot > eps

    if not np.any(valid_points):
        return Jloc

    for k in range(E.shape[0]):
        ratio = np.zeros_like(W[k])
        np.divide(W[k], Wtot[..., None], out=ratio, where=valid_points[..., None])

        terms = np.zeros_like(W[k])
        mask = (W[k] > eps) & valid_points[..., None]
        terms[mask] = -W[k][mask] * np.log(ratio[mask])
        Jloc[k] = float(np.mean(np.sum(terms, axis=-1)))

    return Jloc


def local_scale_orientation_entropy_profile_with_local_q(
    lifted_channel_energy_profile: np.ndarray,
    lifted_local_q_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Compute Jloc using spatially local coherence weights.

    The local ordered channel weights are

        W_{k,alpha}(x) = max(q_k(x), 0) E_{k,alpha}(x).
    """
    E = np.asarray(lifted_channel_energy_profile, dtype=float)
    q = np.asarray(lifted_local_q_profile, dtype=float)

    if E.ndim != 4:
        raise ValueError("lifted_channel_energy_profile must have shape (n_steps, L, L, d)")
    if q.ndim != 3:
        raise ValueError("lifted_local_q_profile must have shape (n_steps, L, L)")
    if E.shape[:3] != q.shape:
        raise ValueError("lifted_channel_energy_profile and lifted_local_q_profile must agree on (n_steps, L, L)")

    W = E * np.maximum(q, 0.0)[..., None]
    Wtot = np.sum(W, axis=(0, 3))

    JlocQ = np.zeros(E.shape[0], dtype=float)
    valid_points = Wtot > eps

    if not np.any(valid_points):
        return JlocQ

    for k in range(E.shape[0]):
        ratio = np.ones_like(W[k])
        np.divide(W[k], Wtot[..., None], out=ratio, where=valid_points[..., None])

        terms = np.zeros_like(W[k])
        mask = (W[k] > eps) & valid_points[..., None]
        terms[mask] = -W[k][mask] * np.log(ratio[mask])
        JlocQ[k] = float(np.mean(np.sum(terms, axis=-1)))

    return JlocQ


def local_scale_orientation_weights(
    lifted_channel_energy_profile: np.ndarray,
    lifted_local_q_profile: np.ndarray,
) -> np.ndarray:
    """
    Return the local organized channel weights used by Jnested.

    W_{k,alpha}(x) = max(q_k(x), 0) E_{k,alpha}(x)

    The result has shape (n_steps, L, L, d).
    """
    E = np.asarray(lifted_channel_energy_profile, dtype=float)
    q = np.asarray(lifted_local_q_profile, dtype=float)

    if E.ndim != 4:
        raise ValueError("lifted_channel_energy_profile must have shape (n_steps, L, L, d)")
    if q.ndim != 3:
        raise ValueError("lifted_local_q_profile must have shape (n_steps, L, L)")
    if E.shape[:3] != q.shape:
        raise ValueError("lifted_channel_energy_profile and lifted_local_q_profile must agree on (n_steps, L, L)")

    return E * np.maximum(q, 0.0)[..., None]


def structural_complexity_profile_from_weights(
    local_weight_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Return the global scale-orientation catalog profile from local weights.

    For local weights W_{k,alpha}(x), define the global channel masses

        W_{k,alpha} = sum_x W_{k,alpha}(x)

    and the global channel distribution p(k, alpha) = W_{k,alpha} / sum W.

    This function returns the per-scale entropy contribution profile

        Jstruct_k = -barW sum_alpha p(k, alpha) log p(k, alpha),

    where barW is the mean total organized weight per pixel.
    """
    W = np.asarray(local_weight_profile, dtype=float)

    if W.ndim != 4:
        raise ValueError("local_weight_profile must have shape (n_steps, L, L, d)")

    n_steps = W.shape[0]
    total_weight = float(np.sum(W))

    if total_weight <= eps:
        return np.zeros(n_steps, dtype=float)

    n_pix = W.shape[1] * W.shape[2]
    channel_mass = np.sum(W, axis=(1, 2))
    p = channel_mass / total_weight

    profile = np.zeros_like(channel_mass)
    mask = p > eps
    profile[mask] = -p[mask] * np.log(p[mask])

    return (total_weight / n_pix) * np.sum(profile, axis=1)


def structural_complexity_from_weights(
    local_weight_profile: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Return Jstruct from the same local weights used by Jnested.
    """
    return float(np.sum(structural_complexity_profile_from_weights(local_weight_profile, eps=eps)))


def heterogeneous_complexity_profile_from_weights(
    local_weight_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Return the scale-resolved heterogeneous contribution profile.

    By construction:

        Jhetero_k = Jstruct_k - Jnested_k
    """
    W = np.asarray(local_weight_profile, dtype=float)

    if W.ndim != 4:
        raise ValueError("local_weight_profile must have shape (n_steps, L, L, d)")

    struct_profile = structural_complexity_profile_from_weights(W, eps=eps)
    nested_profile = local_scale_orientation_entropy_profile_from_weights(W, eps=eps)
    hetero_profile = struct_profile - nested_profile

    tiny_negative = hetero_profile > -1e-12
    hetero_profile = hetero_profile.copy()
    hetero_profile[tiny_negative] = np.maximum(hetero_profile[tiny_negative], 0.0)

    if np.any(hetero_profile < -1e-12):
        raise ValueError("heterogeneous profile became substantially negative; check weight construction")

    return hetero_profile


def heterogeneous_complexity_from_weights(
    local_weight_profile: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Return Jhetero = Jstruct - Jnested from local weights.
    """
    return float(np.sum(heterogeneous_complexity_profile_from_weights(local_weight_profile, eps=eps)))


def local_scale_orientation_entropy_profile_from_weights(
    local_weight_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Compute Jnested directly from local organized weights W_{k,alpha}(x).
    """
    W = np.asarray(local_weight_profile, dtype=float)

    if W.ndim != 4:
        raise ValueError("local_weight_profile must have shape (n_steps, L, L, d)")

    Jnested = np.zeros(W.shape[0], dtype=float)
    Wtot = np.sum(W, axis=(0, 3))
    valid_points = Wtot > eps

    if not np.any(valid_points):
        return Jnested

    for k in range(W.shape[0]):
        ratio = np.ones_like(W[k])
        np.divide(W[k], Wtot[..., None], out=ratio, where=valid_points[..., None])

        terms = np.zeros_like(W[k])
        mask = (W[k] > eps) & valid_points[..., None]
        terms[mask] = -W[k][mask] * np.log(ratio[mask])
        Jnested[k] = float(np.mean(np.sum(terms, axis=-1)))

    return Jnested


def structural_complexity_profile(
    lifted_channel_energy_profile: np.ndarray,
    lifted_local_q_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Return Jstruct_k using the same local weights as Jnested.
    """
    W = local_scale_orientation_weights(
        lifted_channel_energy_profile,
        lifted_local_q_profile,
    )
    return structural_complexity_profile_from_weights(W, eps=eps)


def structural_complexity(
    lifted_channel_energy_profile: np.ndarray,
    lifted_local_q_profile: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Return Jstruct using the same local weights as Jnested.
    """
    W = local_scale_orientation_weights(
        lifted_channel_energy_profile,
        lifted_local_q_profile,
    )
    return structural_complexity_from_weights(W, eps=eps)


def heterogeneous_complexity_profile(
    lifted_channel_energy_profile: np.ndarray,
    lifted_local_q_profile: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Return Jhetero_k using the same local weights as Jnested.
    """
    W = local_scale_orientation_weights(
        lifted_channel_energy_profile,
        lifted_local_q_profile,
    )
    return heterogeneous_complexity_profile_from_weights(W, eps=eps)


def heterogeneous_complexity(
    lifted_channel_energy_profile: np.ndarray,
    lifted_local_q_profile: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Return Jhetero using the same local weights as Jnested.
    """
    W = local_scale_orientation_weights(
        lifted_channel_energy_profile,
        lifted_local_q_profile,
    )
    return heterogeneous_complexity_from_weights(W, eps=eps)
