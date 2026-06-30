from __future__ import annotations

import numpy as np

from .complexity import validate_image


def tile_shuffle(
    image: np.ndarray,
    tile_size: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Split image into non-overlapping square tiles and randomly reshuffle them.

    The content of each tile is preserved.
    Only tile positions are randomized.
    """
    validate_image(image)

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    L = image.shape[0]

    if L % tile_size != 0:
        raise ValueError(
            f"image size {L} is not divisible by tile_size={tile_size}"
        )

    n = L // tile_size

    if image.ndim == 2:
        tiles = image.reshape(n, tile_size, n, tile_size)
        tiles = tiles.swapaxes(1, 2)
        tiles = tiles.reshape(n * n, tile_size, tile_size)
    else:
        channels = image.shape[-1]
        tiles = image.reshape(n, tile_size, n, tile_size, channels)
        tiles = tiles.swapaxes(1, 2)
        tiles = tiles.reshape(n * n, tile_size, tile_size, channels)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n * n)
    tiles = tiles[perm]

    if image.ndim == 2:
        out = tiles.reshape(n, n, tile_size, tile_size)
        out = out.swapaxes(1, 2)
        out = out.reshape(L, L)
    else:
        out = tiles.reshape(n, n, tile_size, tile_size, channels)
        out = out.swapaxes(1, 2)
        out = out.reshape(L, L, channels)

    return out


def _phase_scramble_2d(
    image: np.ndarray,
    rng: np.random.Generator,
    preserve_dc: bool = True,
) -> np.ndarray:
    """
    Phase-scramble a single scalar 2D image while preserving its Fourier amplitude.

    The construction uses random real noise to generate phases. This guarantees
    Hermitian symmetry of the randomized Fourier field, so the inverse FFT is real
    up to numerical roundoff.
    """
    F = np.fft.fft2(image)
    amplitude = np.abs(F)

    noise = rng.normal(size=image.shape)
    random_phase = np.angle(np.fft.fft2(noise))

    F_scrambled = amplitude * np.exp(1j * random_phase)

    if preserve_dc:
        F_scrambled[0, 0] = F[0, 0]

    scrambled = np.fft.ifft2(F_scrambled).real

    return scrambled


def phase_scramble(
    image: np.ndarray,
    seed: int | None = None,
    preserve_dc: bool = True,
) -> np.ndarray:
    """
    Fourier phase-scramble an image while preserving its power spectrum.

    For grayscale images:
        input shape:  (L, L)
        output shape: (L, L)

    For RGB/vector-valued images:
        input shape:  (L, L, C)
        output shape: (L, L, C)

    The Fourier amplitude of each channel is preserved.
    The Fourier phase is randomized.

    Important:
        The returned array is raw float data. It is not clipped or normalized
        for display. It may contain values outside the original input range.
    """
    validate_image(image)

    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=float)

    if img.ndim == 2:
        return _phase_scramble_2d(
            img,
            rng=rng,
            preserve_dc=preserve_dc,
        )

    channels = []
    for c in range(img.shape[-1]):
        scrambled_c = _phase_scramble_2d(
            img[..., c],
            rng=rng,
            preserve_dc=preserve_dc,
        )
        channels.append(scrambled_c)

    return np.stack(channels, axis=-1)


def power_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Return |FFT(image)|^2.

    For RGB/vector images, returns one power spectrum per channel:
        grayscale: (L, L)
        RGB:       (L, L, C)
    """
    validate_image(image)

    img = np.asarray(image, dtype=float)

    if img.ndim == 2:
        F = np.fft.fft2(img)
        return np.abs(F) ** 2

    spectra = []
    for c in range(img.shape[-1]):
        F = np.fft.fft2(img[..., c])
        spectra.append(np.abs(F) ** 2)

    return np.stack(spectra, axis=-1)
