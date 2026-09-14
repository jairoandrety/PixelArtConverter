# Performance notes

Developer note for the v2.8 pipeline work. Numbers come from `bench.py`-style runs on
the development machine (Windows 11, Python 3.14, NumPy only) and are meant for relative
comparison, not as absolute guarantees.

## What changed

Connected-region segmentation is the only stage with a per-pixel Python loop, and both
the internal outlines and the noise cleanup depend on it. Two changes were made:

1. **Run-based flood fill.** The fill walks maximal horizontal runs of identical color
   instead of individual pixels. This is exactly equivalent — two identical adjacent
   pixels are at distance 0 and so always share a region, whatever the threshold — but
   pixel art is built from long uniform runs, so the graph is far smaller than the image.
   Runs are found with vectorized NumPy comparisons; the fill itself uses plain Python
   integers, because NumPy scalar indexing dominated the old runtime.
2. **Bounded memory.** The old implementation retained a list of `(y, x)` tuples for
   every component. A single reusable buffer now holds the component being labelled, so
   memory is bounded by the image rather than by the number or size of regions.

Equivalence was checked against the previous implementation over 8004 randomized cases
(varying sizes, palettes, alpha, thresholds and minimum areas) plus degenerate shapes and
a fully transparent image: zero mismatches.

## Measurements

Segmentation, threshold 18, minimum area 3:

| Case | Before | After | Speedup | Peak MB before | Peak MB after |
| --- | --- | --- | --- | --- | --- |
| One large region, 128 px | 0.65 s | 0.00 s | 204x | 2.6 | 0.5 |
| One large region, 256 px | 2.68 s | 0.01 s | 410x | 10.3 | 1.9 |
| One large region, 512 px | 11.33 s | 0.03 s | 412x | 45.7 | 7.6 |
| Many small regions, 128 px | 0.82 s | 0.74 s | 1.1x | 0.1 | 2.9 |
| Many small regions, 256 px | 3.36 s | 3.44 s | 1.0x | 0.4 | 11.6 |
| Many small regions, 512 px | 15.16 s | 16.70 s | 0.9x | 1.6 | 54.8 |

Region-aware cleanup, tolerance 20, maximum area 3, on a two-region image with scattered
one-pixel speckles:

| Case | Before | After | Speedup |
| --- | --- | --- | --- |
| 128 px | 0.77 s | 0.06 s | 13x |
| 256 px | 3.01 s | 0.22 s | 14x |
| 512 px | 13.90 s | 1.15 s | 12x |

Full pipeline at 512 px with cleanup and internal outlines both enabled:

| | Time | Peak memory |
| --- | --- | --- |
| Before | 26.85 s | 31.6 MB |
| After | 1.24 s | 14.7 MB |

A 512 px preview of realistic pixel art (two large regions plus speckles) renders in
about 0.14 s with both stages enabled, which is inside the 100 ms debounce plus display
work and keeps the preview interactive.

## The one case that did not improve

"Many small regions" is uniform random RGB noise: every pixel is its own color, so runs
degenerate to single pixels and the run machinery pays overhead for nothing. It is at
parity in time and uses more peak memory, because the run tables are materialized up
front regardless of how many runs there are.

This is a synthetic worst case rather than pixel art, and the memory is still bounded by
the image. It is recorded here because it is the honest cost of the trade, and because it
is the case to re-measure if the segmentation is ever revisited.

## Dependencies

The plan called for profiling before adding a dependency such as Numba. After the run
based rewrite, segmentation is no longer the bottleneck for realistic input, so **no new
runtime dependency is justified**. The project still needs only Pillow and NumPy.

If the pathological case ever matters in practice, the next step to measure would be a
union-find over run adjacency rather than a stack-based fill, still dependency-free.

## Interaction

Renders that overrun the debounce window are reported: the status bar shows
`Processing preview...` before a render that is expected to be slow, and appends the
measured duration afterwards. The expectation is calibrated from the previous render, so
fast edits never flash the notice.
