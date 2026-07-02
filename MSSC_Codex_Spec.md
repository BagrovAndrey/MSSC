# MSSC Image Complexity: Current Spec and Development Notes

This repository implements a minimal image-based version of multi-scale structural complexity (MSSC). The current goal is not to build a polished package, but to keep a small, transparent research codebase for testing definitions of image complexity.

The code should remain simple, NumPy-based, and easy to inspect. Avoid adding large dependencies. Current intended dependencies are:

```bash
numpy
pillow
matplotlib
```

Run scripts from the repository root with:

```bash
PYTHONPATH=. python scripts/...
```

## Current repository structure

Expected structure:

```text
mssc/
  __init__.py
  complexity.py
  image_io.py
  visualize.py
  shuffle.py
  orientation.py

scripts/
  compute_complexity.py
  visualize_layers.py
  shuffle_compare.py
```

Main experimental script:

```bash
PYTHONPATH=. python scripts/shuffle_compare.py image.png \
  --mode grayscale \
  --scramble phase \
  --seed 123 \
  --out-plot result.png \
  --out-csv result.csv
```

## Core MSSC definition

The core MSSC pipeline is based on iterative coarse-graining:

```text
f_0 -> f_1 -> f_2 -> ...
```

For now, the concrete image RG uses non-overlapping `2x2` block averaging.

At each RG step:

```text
f_k        = image at scale k
f_{k+1}    = coarse_grain(f_k)
U f_{k+1}  = nearest-neighbor upscaling of f_{k+1} back to the shape of f_k
d_k        = f_k - U f_{k+1}
```

The naive partial complexity is:

```text
C_k = 0.5 * mean(d_k^2)
```

For RGB/vector images, the square is summed over channels before spatial averaging:

```text
C_k = 0.5 * mean_xy sum_c d_k(x,y,c)^2
```

The total naive complexity is:

```text
C = sum_k C_k
```

This quantity is best interpreted as scale-resolved residual/detail energy, not as a complete complexity measure.

## Image loading conventions

`mssc/image_io.py` should support:

```python
load_image(path, size="auto", mode="rgb"|"grayscale", value_range="minus1_1"|"0_1")
```

Conventions:

- `size="auto"` resizes the image to a square whose size is the nearest power of two to `max(width, height)`.
- Aspect ratio preservation is not required for the current toy experiments.
- `mode="rgb"` returns shape `(L, L, 3)`.
- `mode="grayscale"` returns shape `(L, L)`.
- `value_range="minus1_1"` maps image values to `[-1, 1]`.
- `value_range="0_1"` maps image values to `[0, 1]`.

Main experiments should be performed in grayscale unless color structure is explicitly the object of study.

Reason: RGB MSSC measures complexity of a three-component chromatic field. Strongly colored synthetic patterns can get a large complexity boost simply because independent structure lives in multiple color channels. For comparisons of geometry, texture, and shape, grayscale is the default.

## Scrambling / null models

`mssc/shuffle.py` should contain:

```python
tile_shuffle(image, tile_size, seed=None)
phase_scramble(image, seed=None, preserve_dc=True)
power_spectrum(image)
```

Interpretation:

- `tile_shuffle` preserves local patches but destroys global arrangement. It also introduces artificial tile-boundary artifacts, especially for small tile sizes.
- `phase_scramble` preserves the Fourier amplitude spectrum and DC component but randomizes phase. It destroys phase organization but can still produce visually structured psychedelic patterns.

Do not assume phase-scrambled natural images are “structureless”. Their absolute complexity diagnostics remain meaningful. Null subtraction may be useful later, but should not replace absolute values.

## Orientation observables

`mssc/orientation.py` implements observables tied specifically to the current `2x2` block image RG. These are not part of universal MSSC core; they are protocol-specific diagnostics.

For each non-overlapping `2x2` block:

```text
[[a, b],
 [c, d]]
```

define Haar-like local detail channels:

```text
h_x  = (a + c - b - d) / 4
h_y  = (a + b - c - d) / 4
h_xy = (a - b - c + d) / 4
```

For grayscale images:

```text
h_B = (h_x, h_y, h_xy)
```

For RGB/vector images, concatenate the Haar channels for all color channels.

### Local orientation coherence Q_k

For each block `B`, define:

```text
e_B = |h_B|^2
u_B = h_B / |h_B|
```

Neighboring block pairs are compared using nematic/sign-insensitive alignment:

```text
(u_B dot u_B')^2
```

The pair weight is:

```text
w_BB' = sqrt(e_B e_B')
```

Then:

```text
mean_dot2 = sum_<BB'> w_BB' (u_B dot u_B')^2 / sum_<BB'> w_BB'
```

Subtract the isotropic baseline:

```text
Q_k = (mean_dot2 - 1/d) / (1 - 1/d)
```

and clip below at zero:

```text
Q_k = max(Q_k, 0)
```

where `d` is the dimension of the Haar-detail vector.

Interpretation:

```text
Q_k near 0:
  local detail directions are random or incoherent

Q_k near 1:
  strong local details have highly aligned/nematic Haar directions
```

Important: high `Q_k` is not complexity. Simple stripes can have very high `Q_k`.

### Orientation entropy / diversity D_k

The current key improvement is orientation-diversity entropy.

Construct the energy-weighted orientation tensor:

```text
M_k = sum_B e_B u_B u_B^T / sum_B e_B
```

Equivalently:

```text
M_k = sum_B h_B h_B^T / sum_B |h_B|^2
```

Let `lambda_a` be the eigenvalues of `M_k`, normalized so that:

```text
sum_a lambda_a = 1
```

Define normalized orientation entropy:

```text
D_k = - sum_a lambda_a log(lambda_a) / log(d)
```

Interpretation:

```text
D_k near 0:
  strong details mostly live in one Haar-detail direction
  example: simple stripes, very regular patterns

D_k near 1:
  strong details occupy Haar-detail space more isotropically
  example: diverse local structures, or noise
```

Noise can have high `D_k`, so `D_k` alone is not complexity.

## Organized profiles

Keep both old and new profiles.

Old diagnostic:

```text
O_k = C_k * max(Q_k, 0)
```

Interpretation: ordered contrast energy. This is useful but overestimates simple regular patterns such as stripes and checkerboards.

New orientation-diverse organized profile:

```text
Odiv_k = C_k * max(Q_k, 0) * D_k
```

Interpretation:

```text
C_k:
  how much detail energy exists at scale k

Q_k:
  whether strong local details have coherent local organization

D_k:
  whether this organization is diverse rather than a single trivial direction
```

The intended behavior:

```text
white noise:
  high C_k, low Q_k, high D_k -> low Odiv_k

simple stripes:
  high C_k, high Q_k, low/moderate D_k -> reduced Odiv_k

natural structured image:
  moderate/high C_k, nonzero Q_k, nonzero D_k -> significant Odiv_k
```

Do not introduce hand-tuned penalties such as `Q(1-Q)` unless explicitly requested. The current preference is to avoid arbitrary “optimal Q” or arbitrary correlation-length parameters.

## Scale entropies

For any nonnegative profile `X_k`, define:

```text
p_k = X_k / sum_j X_j
H_X = - sum_k p_k log(p_k)
S_X = (sum_k X_k) * H_X
```

Currently useful summaries:

```text
C_total    = sum C_k
O_total    = sum O_k
Odiv_total = sum Odiv_k

H_C
H_O
H_Odiv

S_C
S_O
S_Odiv

Q_mean
D_mean
```

`S_Odiv` is currently the most interesting scalar candidate, but do not treat it as final. This is still exploratory.

## Intensity normalization

Do not normalize by the image variance as a main metric.

Reasoning:

- Total variance is itself a primitive but meaningful kind of structural information.
- Dividing by variance makes uniform images problematic.
- It artificially removes contrast differences that may be physically meaningful.

Optional diagnostic intensity normalization is allowed:

```bash
--normalize-intensity none
--normalize-intensity minmax
--compare-normalized
```

`minmax` should affinely map the loaded image to its full natural range:

```text
[-1, 1] if input contains negative values
[0, 1] otherwise
```

This is a diagnostic mode only. It answers: what happens if absolute contrast amplitude is ignored and only the shape/geometry of the intensity pattern is compared?

## shuffle_compare.py expected behavior

`scripts/shuffle_compare.py` should:

1. Load image.
2. Optionally apply intensity normalization.
3. Scramble the image using either tile shuffle or phase scrambling.
4. Compute profiles for original and scrambled:
   - `C_k`
   - `Q_k`
   - `D_k`
   - `O_k`
   - `Odiv_k`
5. Print summaries.
6. Optionally save CSV.
7. Optionally save plot.

Expected CLI options:

```bash
--mode rgb|grayscale
--value-range 0_1|minus1_1
--normalize-intensity none|minmax
--compare-normalized
--scramble tile|phase
--tile-size INT
--seed INT
--out-plot PATH
--out-csv PATH
--save-scrambled PATH
```

Default mode should preferably be:

```bash
--mode grayscale
```

The plot should show:

```text
row 1: original and scrambled images
row 2: C_k profiles
row 3: Q_k profiles
row 4: D_k profiles
row 5: O_k and Odiv_k profiles
```

CSV columns should include:

```text
k
original_C
original_Q
original_D
original_O
original_Odiv
scrambled_C
scrambled_Q
scrambled_D
scrambled_O
scrambled_Odiv
```

## Scientific interpretation to preserve

The current conceptual problem is:

```text
C_k likes detail energy, including noise and simple high-contrast patterns.
Q_k likes local order.
O_k = C_k Q_k over-rewards simple regular patterns because stripes/checkerboards have both large C_k and large Q_k.
```

The new idea is to distinguish “ordered but trivial” from “ordered and diverse” by adding orientation entropy:

```text
Odiv_k = C_k Q_k D_k
```

This is not the final theory, but it is the current preferred next diagnostic.

## Implementation style

- Keep functions small and explicit.
- Prefer full-file replacements when making major changes.
- Avoid clever abstractions.
- Avoid adding dependencies beyond NumPy, Pillow, Matplotlib.
- Keep grayscale as the main experimental path.
- Keep RGB support, but treat it as a chromatic-field diagnostic.
- Do not remove old `O_k`; keep it for comparison.
- Do not silently change definitions of `C_k`, `Q_k`, or `D_k`.
- Add comments/docstrings for any new diagnostic.
- When adding tests, focus on toy images:
  - uniform image
  - white noise
  - horizontal stripes
  - checkerboard
  - multi-scale synthetic pattern
  - natural image
  - phase-scrambled natural image
  - tile-shuffled natural image

## Immediate next tasks for Codex

1. Inspect the current repository and verify that:
   - `mssc/orientation.py` contains `orientation_entropy_profile`.
   - `scripts/shuffle_compare.py` computes and plots `D_k` and `Odiv_k`.
   - default image mode is grayscale.
   - no variance normalization remains in the main summary.

2. If needed, clean up `shuffle_compare.py` while preserving behavior.

3. Add a small script or test helper for generating canonical toy images:
   - uniform
   - noise
   - stripes
   - checkerboard
   - multi-scale stripes or hierarchy

4. Add a batch benchmark script that runs `shuffle_compare`-style analysis on toy images and writes a compact CSV summary with:
   - `C_total`
   - `O_total`
   - `Odiv_total`
   - `S_C`
   - `S_O`
   - `S_Odiv`
   - `Q_mean`
   - `D_mean`

5. Do not implement null subtraction yet. Keep phase scrambling as diagnostic only.
