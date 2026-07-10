# MSSC metrics: canonical terminology

This file defines the current metric hierarchy for the image-based MSSC repository.

The purpose is to keep one stable vocabulary while preserving older exploratory diagnostics in the codebase.

None of the orientation-aware quantities below should yet be treated as a universal or final definition of complexity. They are specific to the current `2 x 2` block-averaging / Haar-detail observer.

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

`Cdetail` is the only quantity in this document that belongs directly to the protocol-agnostic MSSC core. It is not a complete complexity measure by itself: noise and simple high-contrast periodic patterns can both have large `Cdetail`.

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

It is used by the current nested/local metric to suppress locally incoherent detail without assigning one global coherence value to the whole image.

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

### `Jglobal` (`Jglob` in the current code)

`Jglobal` is the entropy contribution of the global joint distribution over RG scale and Haar-detail channel.

Using ordered channel weights

```text
W_{k,alpha} = max(Q_k, 0) * E_{k,alpha}
W_tot       = sum_{k,alpha} W_{k,alpha}
```

its scale profile is

```text
Jglobal_k = -sum_alpha W_{k,alpha} * log(W_{k,alpha} / W_tot)
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

For this reason, `Jglobal` is primarily a comparison and failure-mode diagnostic, not the main complexity candidate.

### `Jnested` (`JlocQ` in the current code)

`Jnested` is the current working candidate for absolute organized multiscale complexity within the block-Haar observer.

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

Known failure mode:

```text
Jnested can overestimate smoothly deformed regular patterns, such as
wavy stripes, because they generate coherent Haar contributions on
several scales.
```

`Jnested` should therefore be described as the current working candidate, not as a settled final metric.

### `Jphase`: phase-specific excess of `Jnested`

`Jphase` compares `Jnested` to a Fourier phase-scrambled null ensemble that preserves the amplitude spectrum:

```text
Jphase = Jnested(original)
       - mean_seed Jnested(phase_scramble(original, seed))
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
- `max(Jphase, 0)` may be shown only as an auxiliary visualization.
- A small `Jphase` does not imply that an image is simple. Some fractal or textural organization is already strongly encoded in the power spectrum.
- `Jphase` is a second coordinate, not a replacement for `Jnested`.

Useful auxiliary diagnostics:

```text
Jphase_relative = Jphase / Jnested(original)
Jphase_z        = Jphase / std_seed(Jnested(null))
```

These should not be promoted to independent complexity metrics.

## 4. Legacy and development diagnostics

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

## 5. Canonical hierarchy

### Primary outputs

```text
Cdetail   how much detail is discarded along the RG trajectory
Jnested   how rich the locally organized RG histories are
```

### Important secondary coordinate

```text
Jphase    how much of Jnested is not explained by the preserved spectrum
```

### Comparison / failure-mode diagnostic

```text
Jglobal   how diverse the global scale-orientation catalog is
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

## 6. Recommended default presentation

### Default scalar summary

```text
Cdetail
Jglobal
Jnested
```

When a phase-null ensemble is requested, also show:

```text
Jphase
Jphase_relative
Jphase_z
```

### Default profile plot

Show only:

```text
Cdetail_k
Jglobal_k
Jnested_k
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
```

No existing metric implementation should be deleted.

## 7. Short glossary

```text
Cdetail:
    amount of discarded detail

Jglobal:
    global diversity across scale-orientation channels

Jnested:
    locally organized multiscale RG-history richness

Jphase:
    phase-specific excess beyond a spectrum-preserving null
```

This glossary should be treated as the canonical terminology for the current repository stage.
