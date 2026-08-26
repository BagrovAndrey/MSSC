# MSSC metrics: canonical terminology

This file defines the current metric hierarchy for the image-based MSSC repository.

The purpose is to keep one stable vocabulary while preserving older exploratory diagnostics in the codebase.

None of the orientation-aware quantities below should yet be treated as a universal or final definition of complexity. They are specific to the current `2 x 2` block-averaging / Haar-detail observer.

## Current Tree

```text
Structural complexity Jstruct
|
+-- Nested complexity Jnested
|   |
|   +-- spectral-null baseline Jspectral_null
|   |
|   +-- phase-specific signed correction Jphase
|
+-- Heterogeneous complexity Jhetero
```

## 1. Core MSSC quantity

### `Cdetail` (`C` in the current code)

At RG scale `k`, let

```text
f_k       = image at scale k
f_{k+1}   = coarse_grain(f_k)
U f_{k+1} = nearest-neighbor lifting back to the shape of f_k
d_k       = f_k - U f_{k+1}
```

Then

```text
C_k = 0.5 * mean(d_k^2)
```

For vector-valued images, the square is summed over channels before spatial averaging.

Canonical interpretation:

```text
Cdetail_k = scale-resolved discarded detail energy
Cdetail   = sum_k Cdetail_k
```

`Cdetail` belongs directly to the protocol-agnostic MSSC core. It is not a complete complexity measure by itself: noise and simple high-contrast periodic patterns can both have large `Cdetail`.

## 2. Supporting Haar/block diagnostics

These quantities describe ingredients of the current block-Haar observer. They are useful for interpretation and debugging, but they are not primary scalar complexity outputs.

### `Q_k`: scale-level orientation coherence

`Q_k` measures energy-weighted nematic alignment of local Haar-detail vectors between neighboring blocks at scale `k`.

Interpretation:

```text
Q_k near 0: local detail directions are incoherent or noise-like
Q_k near 1: local detail directions are strongly aligned
```

High `Q_k` is not the same as high complexity. Straight stripes can have very high `Q_k`.

### `q_k(x)`: local orientation coherence

`q_k(x)` is the spatially resolved counterpart of `Q_k`, lifted to original-image coordinates.

It is used by the current nested and structural branches to suppress locally incoherent detail without assigning one global coherence value to the whole image.

### `D_k`: within-scale orientation diversity

`D_k` is the entropy of the energy-weighted Haar orientation tensor at scale `k`.

Interpretation:

```text
D_k near 0: one Haar-detail direction dominates
D_k near 1: energy is broadly distributed in Haar-detail space
```

Noise may have high `D_k`, so `D_k` is not a complexity measure by itself.

## 3. Canonical output metrics

The current public-facing vocabulary should use the following names.

### `Jstruct`

`Jstruct` is the global organized scale-orientation catalog entropy built from the same local weights used by `Jnested`.

For each scale, Haar channel, and original-space point, define

```text
W_{k,alpha}(x) = q_k(x) * E_{k,alpha}(x)
W(x)           = sum_{j,beta} W_{j,beta}(x)
```

and let

```text
p(k, alpha) = sum_x W_{k,alpha}(x) / sum_{x,j,beta} W_{j,beta}(x)
barW        = mean_x W(x)
```

Then

```text
Jstruct_k = -barW * sum_alpha p(k, alpha) log p(k, alpha)
Jstruct   = sum_k Jstruct_k
```

Canonical interpretation:

```text
Jstruct = total organized diversity of scale-orientation structure
          present anywhere in the image
```

This quantity intentionally includes both nested and spatially heterogeneous complexity.

### `Jnested` (`JlocQ` in the current code)

`Jnested` is the locally organized multiscale complexity branch within the current block-Haar observer.

For each scale, Haar channel, and original-space point,

```text
W_{k,alpha}(x) = q_k(x) * E_{k,alpha}(x)
W_tot(x)       = sum_{j,beta} W_{j,beta}(x)
```

and

```text
Jnested_k = -mean_x sum_alpha W_{k,alpha}(x)
                         * log(W_{k,alpha}(x) / W_tot(x))
```

Canonical interpretation:

```text
Jnested = locally organized detail energy distributed over a nontrivial
          scale-orientation RG history
```

What it is designed to suppress:

```text
simple single-scale order
spatial patchwork without local nesting
locally incoherent noise
```

Known caveat:

```text
Jnested can overestimate smoothly deformed regular patterns, such as
wavy stripes, because they generate coherent Haar contributions on
several scales.
```

This is why `Jnested` should no longer be treated as the entire structural complexity criterion by itself.

### `Jhetero`

`Jhetero` is the heterogeneous branch:

```text
Jhetero = Jstruct - Jnested
```

Equivalently,

```text
Jhetero = barW * I(X; C)
```

where `X` is spatial position and `C = (k, alpha)` is the scale-orientation channel.

Canonical interpretation:

```text
Jhetero = diversity between local RG histories across space
```

Important limitation:

`Jhetero` detects that different local structural types occur at different positions, but it does not encode the geometry of how those regions are arranged. Spatially permuting whole local-history types can preserve `Jhetero`.

Expected contrast:

```text
wavy_stripes:
  high Jnested, comparatively lower Jhetero

patchwork:
  simpler local histories, but higher Jhetero
```

Numerically, `Jhetero` should be non-negative up to tiny floating-point noise.

### `Jspectral_null`

`Jspectral_null` is the phase-null baseline of the nested branch:

```text
Jspectral_null = mean_seed Jnested(phase_scramble(original, seed))
```

Canonical interpretation:

```text
Jspectral_null = nested complexity expected from the Fourier amplitude
                 spectrum alone under the phase-scramble null
```

This quantity depends on the chosen null model.

### `Jphase`

`Jphase` is the signed phase-specific correction to the nested branch:

```text
Jphase = Jnested(original) - Jspectral_null
```

The corresponding profile is

```text
Jphase_k = Jnested_k(original)
         - mean_seed Jnested_k(phase_scramble(original, seed))
```

Canonical interpretation:

```text
Jphase = part of Jnested not explained by the spectrum-preserving null
```

Important cautions:

- `Jphase` may be negative.
- The unclipped value is the primary diagnostic.
- A small or negative `Jphase` does not imply that an image is simple.
- `Jphase` is a decomposition of the nested branch, not an extra positive term to add again.

Useful auxiliary diagnostics:

```text
Jphase_relative = Jphase / Jnested(original)
Jphase_z        = Jphase / std_seed(Jnested(null))
```

These should not be promoted to independent complexity metrics.

### Exact additive decomposition

The current conceptual hierarchy is:

```text
Jstruct = Jnested + Jhetero
Jnested = Jspectral_null + Jphase
```

Therefore

```text
Jstruct = Jhetero + Jspectral_null + Jphase
```

This is an additive decomposition relative to the phase-null model. Because `Jphase` is signed, it is not a decomposition into three positive fractions.

## 4. Comparison and failure-mode diagnostics

### `Jglobal` (`Jglob` in the current code)

`Jglobal` is the entropy contribution of the global joint distribution over RG scale and Haar-detail channel using scale-level `Q_k` weights:

```text
W_{k,alpha} = max(Q_k, 0) * E_{k,alpha}
W_tot       = sum_{k,alpha} W_{k,alpha}
Jglobal_k   = -sum_alpha W_{k,alpha} * log(W_{k,alpha} / W_tot)
```

Canonical interpretation:

```text
Jglobal = diversity of the global scale-orientation catalog
```

Known failure mode:

```text
Jglobal rewards spatial patchwork.
```

Different simple patterns in different image regions can produce a broad global catalog even when no local region has a nontrivial multiscale history.

For this reason, `Jglobal` is primarily a comparison and failure-mode diagnostic, not the main global branch in the current structural decomposition.

## 5. Legacy and development diagnostics

The following quantities should remain implemented for regression tests and scientific interpretation, but should be hidden from the default summary and default plots.

### `O_k`

```text
O_k = C_k * max(Q_k, 0)
```

Interpretation:

```text
organized detail energy
```

Known failure mode:

```text
over-rewards simple coherent patterns such as stripes and checkerboards
```

### `Odiv_k`

```text
Odiv_k = C_k * max(Q_k, 0) * D_k
```

Interpretation:

```text
organized detail energy weighted by within-scale orientation diversity
```

Known failure mode:

```text
can become exactly zero when each scale contains only one active Haar
channel, even if several different scales are present
```

### `Jloc`

`Jloc` is the earlier local/nested entropy construction that uses scale-level `Q_k` rather than local `q_k(x)`.

It was an important intermediate step between `Jglobal` and `Jnested`, but it is no longer the preferred public-facing quantity.

## 6. Canonical hierarchy

### Primary outputs

```text
Cdetail   how much detail is discarded along the RG trajectory
Jstruct   total organized structural diversity
Jnested   how rich the locally organized RG histories are
Jhetero   how strongly local RG histories differ across space
```

### Nested-branch decomposition

```text
Jspectral_null how much of Jnested is explained by the preserved spectrum under the chosen null
Jphase    signed phase-specific correction to the nested branch
```

### Comparison / failure-mode diagnostic

```text
Jglobal   how diverse the global scale-orientation catalog is under the older
          scale-level weighting
```

### Supporting diagnostics

```text
Q_k
q_k(x)
D_k
```

### Legacy development diagnostics

```text
O_k
Odiv_k
Jloc_k
```

## 7. Recommended default presentation

### Default scalar summary

```text
Cdetail
Jstruct
Jnested
Jhetero
Jspectral_null
Jphase
Jphase_z
```

### Default profile plot

Show only:

```text
Cdetail_k
Jstruct_k
Jnested_k
Jhetero_k
```

When phase-null analysis is enabled, add:

```text
Jnested_k(original)
mean Jnested_k(null)
Jphase_k
```

### Full diagnostic mode

A `--full-diagnostics` or equivalent option may additionally expose:

```text
Q_k
D_k
O_k
Odiv_k
Jloc_k
Jglobal_k
```

No existing metric implementation should be deleted.

## 8. Short glossary

```text
Cdetail:
    amount of discarded detail

Jstruct:
    total organized diversity of scale-orientation structure present
    anywhere in the image

Jnested:
    locally organized multiscale RG-history richness

Jhetero:
    diversity between local RG histories across space

Jspectral_null:
    nested complexity explained by the phase-scramble null

Jphase:
    signed phase-specific excess beyond the spectral baseline

Jglobal:
    legacy global diversity across scale-orientation channels;
    useful as a patchwork-sensitive comparison diagnostic
```
