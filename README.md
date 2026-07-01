# MSSC Image

Minimal Python tools for computing multi-scale structural complexity profiles of images.

This repository implements a simple real-space version of multi-scale structural complexity (MSSC) for images. The code is intentionally small and experimental. The current goal is not to provide a polished package, but to explore how different image transformations affect scale-resolved complexity profiles.

## What it computes

The core object is an image represented as a square NumPy array:

```python
(L, L)        # grayscale image
(L, L, C)     # RGB or vector-valued image
```

The image is repeatedly coarse-grained by averaging non-overlapping `2 x 2` blocks. At each step, the coarse image is upscaled back to the previous resolution and compared with the previous image.

If

```text
f_k     = image at coarse-graining step k
f_{k+1} = coarse_grain(f_k)
U       = nearest-neighbor upscaling
```

then the detail residual is

```text
d_k = f_k - U f_{k+1}
```

and the partial complexity is

```text
C_k = 0.5 * mean(d_k^2)
```

For RGB images, the squared difference is summed over color channels before spatial averaging.

The full MSSC profile is the sequence

```text
C_0, C_1, C_2, ...
```

and the total naive MSSC is

```text
C = sum_k C_k
```

## Current experimental extension: local orientation coherence

The repository also contains an experimental phase/organization-sensitive extension.

For every `2 x 2` block, the code computes a local Haar-like detail vector:

```text
h_x   = left-right contrast
h_y   = top-bottom contrast
h_xy  = diagonal/checkerboard contrast
```

For a scalar block

```text
a b
c d
```

the components are

```text
h_x  = (a + c - b - d) / 4
h_y  = (a + b - c - d) / 4
h_xy = (a - b - c + d) / 4
```

For RGB images, the same construction is applied channel-wise and concatenated into one local detail vector.

The current local orientation coherence profile is

```text
Q_k
```

where `Q_k` measures energy-weighted signed alignment of local Haar detail vectors between neighboring blocks at scale `k`.

The organized MSSC profile is then defined as

```text
O_k = C_k * max(Q_k, 0)
```

and the total organized MSSC is

```text
O = sum_k O_k
```

Important: this definition is experimental. In particular, the current signed coherence may strongly suppress natural images with curved or alternating local structures. A future version may replace it with a nematic/sign-insensitive coherence based on squared local alignment.

## Entropic summaries over scales

The code also computes entropy-based summaries of scale profiles.

For any non-negative profile `P_k`, define

```text
p_k = P_k / sum_j P_j
```

and

```text
H(P) = - sum_k p_k log(p_k)
```

The entropic complexity is

```text
S(P) = sum_k P_k * H(P)
```

The comparison script reports:

```text
C     = sum_k C_k
O     = sum_k O_k

H_C   = entropy of normalized C_k
H_O   = entropy of normalized O_k

S_C   = C * H_C
S_O   = O * H_O
```

Interpretation:

```text
C_k:
  scale-resolved residual energy

Q_k:
  local orientation coherence

O_k:
  organized residual energy

H_C, H_O:
  how broadly the corresponding profile is distributed across scales

S_C, S_O:
  total amount of structure weighted by its multiscale spread
```

## Image loading

Images are loaded with Pillow and converted to either RGB or grayscale.

By default, images are resized to a square whose size is the nearest power of two based on the larger input dimension:

```text
1000 x 875 -> 1024 x 1024
600 x 600  -> 512 x 512
```

This is controlled by the command-line argument:

```bash
--size auto
```

Other options:

```bash
--size 1024   # force 1024 x 1024
--size none   # do not resize; image must already be square
```

Pixel values are converted once from `uint8` to either `[0, 1]` or `[-1, 1]`.

The code does not normalize the coarse-grained layers independently.

## Scrambling modes

The repository currently implements two null models.

### Tile shuffle

```bash
--scramble tile --tile-size 32
```

The image is cut into non-overlapping square tiles and the tile positions are randomly shuffled.

This preserves local patches but destroys their global arrangement. It also introduces artificial tile boundaries.

### Phase scrambling

```bash
--scramble phase
```

The Fourier amplitude of each channel is preserved while the Fourier phase is randomized. This approximately preserves the power spectrum but destroys the spatial phase organization of the image.

The phase-scrambled image is kept as a raw floating-point array for analysis. If it is saved as an image, the saved file is only a display version.

## Minimal setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pillow matplotlib
```

Run scripts from the repository root with:

```bash
PYTHONPATH=. python scripts/script_name.py ...
```

## Compute the naive MSSC profile

```bash
PYTHONPATH=. python scripts/compute_complexity.py image.png \
  --out-csv profile.csv \
  --out-plot profile.png
```

Optional arguments:

```bash
--size auto
--size 512
--size none

--mode rgb
--mode grayscale

--value-range minus1_1
--value-range 0_1

--n-steps 8
```

## Visualize coarse-graining layers

```bash
PYTHONPATH=. python scripts/visualize_layers.py image.png \
  --n-steps 6 \
  --out layers.png
```

This shows the original image and successive coarse-grained layers.

## Compare original and scrambled images

Tile shuffle:

```bash
PYTHONPATH=. python scripts/shuffle_compare.py image.png \
  --scramble tile \
  --tile-size 32 \
  --seed 123 \
  --out-plot comparison_tile.png \
  --out-csv comparison_tile.csv \
  --save-scrambled tile_scrambled.png
```

Phase scrambling:

```bash
PYTHONPATH=. python scripts/shuffle_compare.py image.png \
  --scramble phase \
  --seed 123 \
  --out-plot comparison_phase.png \
  --out-csv comparison_phase.csv \
  --save-scrambled phase_scrambled_display.png
```

The comparison plot shows:

```text
original image
scrambled image

C_k profiles
Q_k profiles
O_k profiles
```

The figure title reports cumulative values:

```text
C, O, S_C, S_O
```

The CSV contains:

```text
k,
original_C, original_Q, original_O,
scrambled_C, scrambled_Q, scrambled_O
```

## Current file structure

```text
mssc-image/
  mssc/
    __init__.py
    complexity.py
    image_io.py
    shuffle.py
    orientation.py
    visualize.py

  scripts/
    compute_complexity.py
    visualize_layers.py
    shuffle_compare.py
```

## Scientific interpretation

The naive MSSC profile `C_k` is best interpreted as a scale-resolved residual-energy profile. It is closely related in spirit to a power spectrum, although the current implementation is based on real-space block coarse-graining rather than Fourier filtering.

The organized profile `O_k` is an experimental attempt to include local organization of details. It should not yet be treated as a final definition of structural complexity.

A useful diagnostic is to compare:

```text
original
tile-shuffled
phase-scrambled
```

If `C_k` is preserved but `O_k` is suppressed, the transformation preserves scale energy while destroying local organization.

## Known limitations

The current implementation is deliberately minimal.

Known limitations:

```text
1. Coarse-graining is currently hard-coded as 2 x 2 block averaging.

2. Local orientation coherence is tied to this block coarse-graining.

3. The current Q_k is signed and may underestimate natural images with curved, alternating, or sign-flipped structures.

4. RGB images are treated as vector-valued arrays, but no color-space correction is performed.

5. PNG/JPEG loading uses ordinary image intensities, not calibrated physical intensities.

6. Saved phase-scrambled images are display-normalized or clipped and should not be reloaded for quantitative analysis.
```

## Next possible improvements

Planned or natural next steps:

```text
1. Replace signed orientation coherence with nematic/sign-insensitive coherence.

2. Keep the core MSSC protocol-agnostic:
   coarse-graining + lifting + dissimilarity.

3. Treat orientation coherence as a protocol-specific observable.

4. Add Fourier coarse-graining as an alternative observer.

5. Add graph coarse-graining later for network complexity.

6. Add tests for constant images, white noise, periodic patterns, natural images, tile shuffle, and phase scramble.
```
