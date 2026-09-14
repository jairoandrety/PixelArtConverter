
# Pixel Art Converter v2.8

Desktop utility for converting illustrations into crisp pixel-art assets.

## Processing modes

- **Pixel Perfect Resize:** resizes the image without applying reconstruction adjustments.
- **Pixel Art Reconstruction:** adds cluster optimization, detail simplification, edge
  adjustments, and color clustering through vectorized NumPy operations while preserving
  the original image dimensions.

## Features

- Opens PNG, JPG, WebP, BMP, and TIFF images.
- Auto-crops near-black or transparent margins.
- Supports exact-size, width-based, or height-based output.
- Provides progressive reduction and a final *Nearest Neighbor* pixel-grid lock. The
  intermediate stages follow whichever dimension has to shrink the most, so very wide
  and very tall sources are treated equivalently.
- Adjusts contrast, cleans isolated noise, and preserves the original alpha channel exactly.
  `Preserve transparency` keeps alpha in the export; disabling it flattens transparent
  pixels to black.
- Includes reconstruction controls for clusters, details, edges, and color clusters.
- Optionally reduces the palette to 2–256 colors, with optional dithering.
- Includes invert, grayscale, and independent red, green, and blue tint controls.
- Keeps every configuration accessible through a scrollable settings panel that
  scrolls from anywhere in it, including over sliders and checkboxes.
- Provides configurable outer borders with a selectable width and color.
- Provides internal color-group outline controls with a selectable width, threshold,
  minimum-area setting, and color.
- Shows the current numeric value beside every processing slider.
- Provides a crisp nearest-neighbor preview with 25%–800% zoom.
- Includes **Fit** and **100%** preview buttons, which also return the view to the
  top-left corner, and scrollbars for enlarged previews.
- Validates typed Width, Height, and Palette values and refreshes the preview after a
  short pause, reporting the offending field in the status bar instead of failing.
- Explains color tolerance, minimum area, transparency, and border geometry through
  hover tooltips.
- Disables reconstruction-only controls in **Pixel Perfect Resize** while keeping
  their values for when reconstruction is selected again.
- Removes small stray color fragments with a conservative, region-aware cleanup.
- Records an undo/redo history of setting changes, with `Ctrl+Z` and `Ctrl+Y`.
- Supports non-destructive rectangle, connected-region, and brush selections on the
  preview, and can restrict color effects to the selected pixels.
- Exports PNG, WebP, and JPEG.

## Color and outline tools

- **Invert colors:** reverses every RGB channel while retaining alpha.
- **Monochrome:** offers a single `Grayscale` mode that maps each visible pixel to
  its luminance while retaining alpha.
- **RGB tints:** independently enable and adjust red, green, and blue channels from
  `-100` to `100`, with the toggle, slider, and numeric value on one row.
- **Outer border:** adds a selectable border around the visible alpha silhouette.
  The output keeps its requested dimensions, so the border can only occupy transparent
  pixels that already exist inside the canvas. When visible content touches an edge the
  status bar reports `Outer border clipped`.
- **Internal object borders:** uses 4-neighbor connected-region segmentation after the
  image reaches its target resolution. Pixels join a region when they are connected
  and within the selected RGB color tolerance; the interface provides configurable
  minimum-area and outline-width controls for the resulting internal outlines. Regions
  smaller than the minimum area are discarded and never produce an outline.

## Noise cleanup

Region-aware cleanup replaces small stray color fragments — an off-yellow pixel inside a
yellow wall — with the color of the region around them. It runs at the final resolution,
after palette reduction and before any outline, so it reuses the output the user sees:

```text
crop -> resize -> reconstruction -> color effects -> palette reduction
     -> noise cleanup -> outer border -> internal outlines
```

The stage works with two different tolerances, and the difference matters:

| Control | Meaning |
| --- | --- |
| **Fragment grouping tolerance** | How similar two neighboring pixels must be to count as *the same fragment*. |
| **Cleanup color tolerance** | How different an adjacent fragment may be before it can *absorb* a small one. |

Grouping must stay below the merge tolerance, or a fragment is swallowed while grouping
and never becomes a candidate at all; the pipeline clamps it if you set it higher.

At grouping `0` only identical colors group together. That suits artwork that has already
been quantized through **Palette**, but on continuous-tone art every pixel becomes its own
fragment and almost nothing can be cleaned — which is why the default is `2` rather than
`0`.

A fragment is merged only when all of the following hold:

- its area is at most the limit derived from **Cleanup strength**;
- a directly adjacent fragment is *strictly larger* than it;
- that neighbor's color is within **Cleanup color tolerance**.

Requiring a strictly larger neighbor is what stops two speckles from merging into each
other, and it keeps **Cleanup strength** monotonic: raising it can only ever clean more,
never less. Alpha is never modified, and **Protect silhouette** (on by default) leaves
fragments that touch transparency or the canvas edge untouched.

`Cleanup strength` scales from one-pixel speckles up to the advanced `Maximum region
area` ceiling and never past it. The status bar reports how many pixels the stage
actually changed, so a setting that cleans nothing says so instead of looking broken.

## Edit history

Undo and redo record *setting values only* — never rendered images — so the stack stays
small regardless of image size. One entry is created per discrete action: a full slider
drag, a validated typed value, a color choice, one brush stroke, or `Reset to defaults`.
Selections travel with the history as 1-bit packed masks — 32 KB for a 512 px mask — so
undo restores both the settings and the selection. History is bounded to 40 actions and
is cleared when a new source image is opened.

| Shortcut | Action |
| --- | --- |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` or `Ctrl+Shift+Z` | Redo |

## Selection and local effects

Selections mark pixels; they never modify the image.

| Mode | Gesture |
| --- | --- |
| Rectangle | Drag on the preview |
| Connected region | Click, using the internal-outline tolerance |
| Brush (add) | Paint to add pixels |
| Brush (subtract) | Paint to remove pixels |

**Connected region** reuses the same segmentation as the internal outlines, so a click
selects exactly the region that would be outlined. The brush interpolates along pointer
motion, so a fast drag leaves no gaps, and each completed stroke is one undoable action.
Canvas coordinates are converted through zoom and scroll offsets, so the same pixels are
addressed at any zoom level.

With **Apply color effects to selection only**, invert, grayscale, RGB tints, and noise
cleanup are confined to the selected pixels. Contrast runs before the resize, so it stays
global. Alpha is preserved either way.

If the output dimensions change, remapping would be ambiguous, so the selection is
cleared and reported rather than silently applied to the wrong pixels. The pipeline
independently ignores a mask that does not match the image it is given.

## Preview navigation

Wheel behavior is identical on Windows, macOS, and X11 (`Button-4` / `Button-5`):

| Input | Preview | Settings panel |
| --- | --- | --- |
| Wheel | Scrolls vertically | Scrolls vertically |
| Shift + wheel | Scrolls horizontally | Scrolls vertically |
| Ctrl + wheel | Zooms 25%–800% | Scrolls vertically |

## Architecture

The image pipeline lives in `pixel_pipeline.py` and does not import Tkinter. It takes a
source image plus a normalized `Settings` record and returns a PIL image:

```python
from pixel_pipeline import Pipeline, Settings

pipeline = Pipeline(Settings(height=256, noise_cleanup=True))
output = pipeline.run(source_image)
```

`pixel_art_converter.py` owns the desktop experience only — variables, dialogs, preview
scheduling, selection overlays, and edit history. It collects its Tk variables into
`Settings` and delegates the run, so the pipeline stays unit-testable and reusable for
batch conversion or a command-line front end.

## Performance

Segmentation walks horizontal runs of identical color rather than individual pixels,
which is exactly equivalent but far cheaper on pixel art. A 512 px preview with both
noise cleanup and internal outlines enabled renders in a fraction of a second, against
roughly 27 seconds before the rewrite, and peak memory is bounded by the image rather
than by the number of regions. The project still needs only Pillow and NumPy.

Measurements, the equivalence check, and the one synthetic case that did not improve are
recorded in [PERFORMANCE.md](PERFORMANCE.md). A render that overruns the debounce window
reports `Processing preview...` and then its measured duration.

## Installation and running

Requires Python 3.10 or later.

```powershell
python -m pip install -r requirements.txt
python pixel_art_converter.py
```

## Build the Windows executable

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --onefile --windowed --name PixelArtConverter pixel_art_converter.py
```

The result is generated at `dist\PixelArtConverter.exe`.

## Tests

```powershell
python -m unittest discover -s tests
```

The suite renders generated Pillow fixtures and covers connected-region segmentation,
minimum-area filtering, color tolerance, alpha preservation, outer-border geometry,
fixed output dimensions, wheel normalization, typed-input validation, noise cleanup,
selection masks, and history descriptions. It needs no additional dependency beyond
`requirements.txt`.
