from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def nearest_power_of_two(n: int) -> int:
    """
    Return the power of two closest to n.

    Examples:
        600  -> 512
        875  -> 1024
        1000 -> 1024
    """
    if n <= 0:
        raise ValueError("n must be positive")

    lower = 2 ** int(np.floor(np.log2(n)))
    upper = 2 ** int(np.ceil(np.log2(n)))

    if abs(n - lower) <= abs(upper - n):
        return lower

    return upper


def auto_square_size(width: int, height: int) -> int:
    """
    Choose target square size from the larger image dimension.
    """
    return nearest_power_of_two(max(width, height))


def load_image(
    path: str | Path,
    size: int | str | None = "auto",
    mode: str = "rgb",
    value_range: str = "minus1_1",
) -> np.ndarray:
    """
    Load image as a square NumPy array.

    size:
      - "auto": resize to nearest power of two based on max(width, height)
      - int: resize to (size, size)
      - None: keep original size, but require the image to be already square

    mode:
      - "rgb": return shape (L, L, 3)
      - "grayscale": return shape (L, L)

    value_range:
      - "0_1": values in [0, 1]
      - "minus1_1": values in [-1, 1]

    Important:
      This function only performs one global affine conversion of pixel values:
          uint8 -> [0, 1]
      or
          uint8 -> [-1, 1]
      It does not normalize RG/coarse-grained layers independently.
    """
    path = Path(path)

    if mode not in {"rgb", "grayscale"}:
        raise ValueError("mode must be 'rgb' or 'grayscale'")

    if value_range not in {"0_1", "minus1_1"}:
        raise ValueError("value_range must be '0_1' or 'minus1_1'")

    with Image.open(path) as img:
        if mode == "rgb":
            img = img.convert("RGB")
        else:
            img = img.convert("L")

        width, height = img.size

        if size == "auto":
            target_size = auto_square_size(width, height)
            img = img.resize(
                (target_size, target_size),
                resample=Image.Resampling.BICUBIC,
            )

        elif isinstance(size, int):
            if size <= 0:
                raise ValueError("size must be positive")
            img = img.resize(
                (size, size),
                resample=Image.Resampling.BICUBIC,
            )

        elif size is None:
            if width != height:
                raise ValueError(
                    "size=None requires an already square image; "
                    f"got {width}x{height}"
                )

        else:
            raise ValueError("size must be 'auto', an integer, or None")

        arr = np.asarray(img, dtype=float) / 255.0

    if value_range == "minus1_1":
        arr = 2.0 * arr - 1.0

    return arr


def save_image(array: np.ndarray, path: str | Path) -> None:
    """
    Save an array as an image.

    Accepts arrays roughly in [0, 1] or [-1, 1].
    """
    path = Path(path)

    arr = np.asarray(array, dtype=float)

    if arr.ndim not in (2, 3):
        raise ValueError("array must have shape (L, L) or (L, L, C)")

    if arr.min() < 0:
        arr = 0.5 * (arr + 1.0)

    arr = np.clip(arr, 0.0, 1.0)
    arr = np.round(255.0 * arr).astype(np.uint8)

    img = Image.fromarray(arr)
    img.save(path)
