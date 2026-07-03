# MSSC Image

Minimal Python tools for experimenting with multi-scale structural complexity (MSSC) on images.

The repository is intentionally small and NumPy-based. It currently contains:

- A basic real-space MSSC profile based on repeated `2 x 2` block averaging.
- Experimental orientation-aware diagnostics built on local Haar-like detail vectors.
- CLI scripts for analysis, scrambling comparisons, toy-image generation, RG-layer visualization, and benchmark panels.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts from the repository root:

```bash
PYTHONPATH=. python3 scripts/<script>.py ...
```

## Repository layout

```text
mssc/
  __init__.py
  complexity.py
  image_io.py
  orientation.py
  shuffle.py
  visualize.py

scripts/
  benchmark_toy_panel.py
  compute_complexity.py
  diagnose_jlocq_outlier.py
  generate_toy_images.py
  make_binary_image.py
  shuffle_compare.py
  visualize_layers.py
```

## Core MSSC definition

An image is treated as either:

```python
(L, L)        # grayscale
(L, L, C)     # RGB or vector-valued
```

At each RG step:

```text
f_{k+1} = coarse_grain(f_k)
d_k = f_k - U f_{k+1}
C_k = 0.5 * mean(d_k^2)
```

where `U` is nearest-neighbor upscaling back to the previous resolution. For multi-channel images, squared differences are summed over channels before spatial averaging.

The naive MSSC profile is:

```text
C_0, C_1, C_2, ...
```

and the total naive complexity is:

```text
C = sum_k C_k
```

## Image loading behavior

`mssc.image_io.load_image(...)` supports:

```python
load_image(path, size="auto", mode="rgb" | "grayscale", value_range="minus1_1" | "0_1")
```

Current behavior:

- `size="auto"` resizes to a square whose side is the nearest power of two to `max(width, height)`.
- `size=<int>` forces `<int> x <int>`.
- `size=None` means no resize; the image must already be square and power-of-two.
- `mode="rgb"` returns `(L, L, 3)`.
- `mode="grayscale"` returns `(L, L)`.
- `value_range="0_1"` maps pixel values to `[0, 1]`.
- `value_range="minus1_1"` maps pixel values to `[-1, 1]`.

The code does one global pixel-value conversion at load time. It does not renormalize coarse-grained layers independently.

## Orientation-aware diagnostics

`mssc/orientation.py` adds observables tied specifically to the current `2 x 2` coarse-graining protocol.

For each `2 x 2` block

```text
a b
c d
```

the local Haar-like detail channels are

```text
h_x  = (a + c - b - d) / 4
h_y  = (a + b - c - d) / 4
h_xy = (a - b - c + d) / 4
```

The main derived profiles are:

- `Q_k`: local orientation coherence.
- `D_k`: orientation entropy / diversity.
- `O_k = C_k * max(Q_k, 0)`.
- `Odiv_k = C_k * max(Q_k, 0) * D_k`.
- `Jglob_k`: global scale-orientation entropy contribution profile.
- `Jloc_k`: local/nested entropy profile weighted by scale-global `Q_k`.
- `JlocQ_k`: local/nested entropy profile weighted by spatially local coherence maps.

These are experimental diagnostics, not stable final definitions of complexity.

## Scrambling modes

`mssc/shuffle.py` currently provides two null models.

Tile shuffle:

```bash
--scramble tile --tile-size 32
```

- Preserves tile contents.
- Randomizes tile positions.
- Destroys global arrangement.
- Introduces tile-boundary artifacts.

Phase scramble:

```bash
--scramble phase
```

- Preserves Fourier amplitude per channel.
- Randomizes Fourier phase.
- Preserves the DC component by default.
- Returns raw float data for analysis; a saved PNG is only a display version.

## Scripts

### `compute_complexity.py`

Computes the naive MSSC profile `C_k`.

```bash
PYTHONPATH=. python3 scripts/compute_complexity.py image.png \
  --out-csv profile.csv \
  --out-plot profile.png
```

Important options:

- `--size auto|none|INT`
- `--mode rgb|grayscale` (default: `rgb`)
- `--value-range minus1_1|0_1` (default: `minus1_1`)
- `--block-size INT` (default: `2`)
- `--n-steps INT`

Outputs:

- Prints `C_k`, total complexity, and total complexity without `C_0`.
- Optional CSV with columns `k, C_k`.
- Optional line plot of the profile.

### `visualize_layers.py`

Saves the original image and successive coarse-grained layers.

```bash
PYTHONPATH=. python3 scripts/visualize_layers.py image.png \
  --n-steps 6 \
  --out layers.png
```

Important options:

- `--size auto|none|INT`
- `--mode rgb|grayscale` (default: `rgb`)
- `--value-range minus1_1|0_1`
- `--block-size INT` (default: `2`)
- `--n-steps INT` (default: `6`)

Each layer panel includes its scale index and, for `k > 0`, the previous-step partial complexity.

### `shuffle_compare.py`

Compares an image against a scrambled version using both naive and orientation-aware profiles.

Tile-shuffle example:

```bash
PYTHONPATH=. python3 scripts/shuffle_compare.py image.png \
  --scramble tile \
  --tile-size 32 \
  --seed 123 \
  --out-plot comparison_tile.png \
  --out-csv comparison_tile.csv
```

Phase-scramble example:

```bash
PYTHONPATH=. python3 scripts/shuffle_compare.py image.png \
  --scramble phase \
  --seed 123 \
  --out-plot comparison_phase.png \
  --out-csv comparison_phase.csv
```

Current defaults and options:

- `--mode grayscale` by default.
- `--size auto|none|INT`
- `--value-range minus1_1|0_1`
- `--normalize-intensity none|minmax` (default: `none`)
- `--compare-normalized` runs both raw and minmax-normalized analyses and writes suffixed outputs.
- `--block-size INT` exists, but orientation observables currently require `--block-size 2`.
- `--n-steps INT`
- `--connectivity 4|8` for local coherence maps (default: `4`)
- `--scramble tile|phase`
- `--tile-size INT` is required for tile shuffle
- `--seed INT`
- `--save-scrambled PATH`

Printed summaries include totals, entropies, entropic complexities, and mean coherence/diversity values for both original and scrambled images.

The comparison CSV contains:

```text
k,
original_C, original_Q, original_D, original_O, original_Odiv, original_Jglob, original_Jloc, original_JlocQ,
scrambled_C, scrambled_Q, scrambled_D, scrambled_O, scrambled_Odiv, scrambled_Jglob, scrambled_Jloc, scrambled_JlocQ
```

The comparison plot includes:

- original and scrambled images
- `C_k`
- `Q_k`
- `D_k`
- `O_k` and `Odiv_k`
- `Jglob_k`, `Jloc_k`, and `JlocQ_k`

### `generate_toy_images.py`

Writes a small set of canonical toy images.

```bash
PYTHONPATH=. python3 scripts/generate_toy_images.py toy_images
```

Current outputs:

- `uniform.png`
- `noise.png`
- `stripes.png`
- `checkerboard.png`
- `multiscale.png`

Important options:

- `--size INT` (default: `256`; must be even)
- `--value-range 0_1|minus1_1` (default: `minus1_1`)
- `--seed INT` for noise (default: `0`)
- `--stripe-width INT` (default: `16`)
- `--checker-tile INT` (default: `16`)

### `make_binary_image.py`

Converts an arbitrary image into a square binary black/white PNG.

```bash
PYTHONPATH=. python3 scripts/make_binary_image.py image.png
```

Example with an explicit output file:

```bash
PYTHONPATH=. python3 scripts/make_binary_image.py image.png \
  --size 512 \
  --threshold 0.15 \
  --output test_images/image_binary_512.png
```

Default pipeline:

```text
1. Load as grayscale
2. Resize to nearest power-of-two square
3. Min-max normalize in the [-1, 1] range
4. Threshold at 0.0
5. Save a binary black/white PNG
```

Important options:

- `--size auto|none|INT`
- `--threshold FLOAT` (default: `0.0`)
- `--normalize minmax|none` (default: `minmax`)
- `--out-dir PATH` (default: `test_images`)
- `--output PATH` to override the default generated filename

Current behavior details:

- The script always loads the image in grayscale mode.
- Internal values are kept in the `[-1, 1]` range before thresholding.
- Pixels `>= threshold` become white (`+1`); pixels below threshold become black (`-1`).
- `--normalize none` skips min-max stretching and thresholds the raw loaded grayscale values directly.
- When `--output` is provided, it takes precedence over `--out-dir`.

By default the output path is:

```text
test_images/<input_stem>_binary.png
```

The saved image is suitable as a clean binary input for later MSSC experiments on synthetic or thresholded real-world shapes.

### `benchmark_toy_panel.py`

Generates synthetic benchmark arrays in memory, analyzes them, and writes summary artifacts.

```bash
PYTHONPATH=. python3 scripts/benchmark_toy_panel.py --out-dir benchmark_toys
```

Generated benchmark families:

- `stripes`
- `checkerboard`
- `patchwork`
- `nested_dyadic`
- `fractal`
- `noise`

Important options:

- `--size INT` (default: `512`; must be power-of-two)
- `--seed INT` (default: `123`)
- `--out-dir PATH`
- `--metric NAME` (default: `JlocQ`) for the panel title and ranking
- `--n-steps auto|INT`
- `--connectivity 4|8`
- `--save-images`

Current outputs:

- `benchmark_summary.csv`
- `benchmark_profiles.csv`
- `benchmark_panel.png`
- `benchmark_profiles.png`

The script also prints rankings by the selected metric and emits simple warnings when benchmark orderings look suspicious.

### `diagnose_jlocq_outlier.py`

Diagnoses why a particular image produces a high `JlocQ` by decomposing the current local-Q entropy pipeline into its ingredients.

Analyze an existing image:

```bash
PYTHONPATH=. python3 scripts/diagnose_jlocq_outlier.py \
  --image path/to/image.png \
  --mode grayscale \
  --value-range minus1_1 \
  --size auto \
  --out-dir diagnostics/image_case
```

Generate synthetic wavy stripes directly as NumPy arrays:

```bash
PYTHONPATH=. python3 scripts/diagnose_jlocq_outlier.py \
  --generate wavy_stripes \
  --size 512 \
  --stripe-period 64 \
  --wave-amplitude 24 \
  --wave-period 256 \
  --threshold 0.0 \
  --out-dir diagnostics/generated_wavy_stripes
```

What it computes:

- the existing profiles `C`, `Q`, `D`, `O`, `Odiv`, `Jglob`, `Jloc`, `JlocQ`
- `Wsum_k`, the q-weighted organized energy per scale before the entropy factor
- `Hloc_factor_k = JlocQ_k / Wsum_k`
- unweighted Haar-channel energies
- q-weighted Haar-channel energies
- local maps for selected scales: lifted `q`, summed channel energy, summed q-weighted energy, and per-scale `JlocQ` contribution maps

Important options:

- `--image PATH` or `--generate wavy_stripes`
- `--size auto|none|INT` for image mode, integer size for generated mode
- `--mode rgb|grayscale` (default: `grayscale`)
- `--value-range minus1_1|0_1`
- `--n-steps INT`
- `--connectivity 4|8`
- generated wavy-stripe controls:
  `--stripe-period`, `--wave-amplitude`, `--wave-period`, `--threshold`, `--binary/--no-binary`
- `--map-scales 3,4,5` to override the default top-3 `JlocQ_k` scales
- `--phase-scramble-seeds N` to repeat analysis over phase-scrambled seeds `0..N-1`
- parameter sweeps:
  `--sweep-wave-amplitude ...`, `--sweep-stripe-period ...`, `--sweep-wave-period ...`

Current outputs:

- `input_image.png`
- `summary.csv`
- `profiles.csv`
- `channel_energy.csv`
- `q_weighted_channel_energy.csv`
- `diagnostic_profiles.png`
- `channel_energy_profiles.png`
- `selected_scale_maps/`
- optional `phase_scramble_summary.csv`
- optional sweep CSV/PNG files

This script is diagnostic only. It does not change the definitions of the existing MSSC-derived metrics.

## Python API

The package exports the main helpers through `mssc/__init__.py`, including:

- `complexity_profile`, `total_complexity`, `coarse_grain`, `upscale_nearest`
- `load_image`, `save_image`
- `rg_layers`, `plot_rg_layers`
- `tile_shuffle`, `phase_scramble`, `power_spectrum`
- orientation-profile helpers such as `local_orientation_coherence_profile`, `orientation_entropy_profile`, `organized_profile`, `orientation_diverse_organized_profile`, `scale_orientation_entropy_profile`, `local_scale_orientation_entropy_profile`, and `local_scale_orientation_entropy_profile_with_local_q`

## Limitations

- Orientation-aware observables are currently implemented only for `2 x 2` coarse-graining.
- `shuffle_compare.py` will raise if you request orientation analysis with `--block-size` other than `2`.
- `size="auto"` resizes to a square without preserving original aspect ratio.
- RGB images are treated as vector-valued arrays without color-space correction.
- Saved phase-scrambled images are display exports, not quantitative source data.
- This is research code: metrics such as `Odiv`, `Jglob`, `Jloc`, and `JlocQ` should be treated as exploratory.
