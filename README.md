
# Pixel Art Converter v2.12

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
- Removes small stray color fragments with two complementary cleanup methods, one
  for dirty conversions and one for already-flat artwork.
- Records an undo/redo history of setting changes, with `Ctrl+Z` and `Ctrl+Y`.
- Supports non-destructive rectangle, connected-region, and brush selections on the
  preview, and can restrict color effects to the selected pixels.
- Detects material classes and their individual objects, turns either into a
  selection, and can repaint whole classes with a color of your choice.
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

Cleanup runs at the final resolution, after palette reduction and before any outline, so
it works on the output you actually see:

```text
crop -> resize -> reconstruction -> color effects -> palette reduction
     -> noise cleanup -> outer border -> internal outlines
```

Two methods are available, because they solve different problems:

| Method | How it works | Use it for |
| --- | --- | --- |
| **Local dominant color** (default) | Snaps an isolated pixel to the dominant color of the window around it | Dirty conversions with many tonal variations |
| **Merge small regions** | Folds a small fragment into a larger adjacent fragment | Artwork that is already flat or palette-reduced |
| **Both** | Dominant color first to consolidate, then region merging | Mixed sources |

Neither method modifies alpha, and **Protect silhouette** (on by default) leaves fragments
that touch transparency or the canvas edge untouched. The status bar reports how many
pixels the stage actually changed, so a setting that cleans nothing says so.

### Local dominant color

Region merging needs regions to exist. On a dirty conversion almost every pixel is a
slightly different tone, so there are none — which is why that method can look inert on
exactly the images that need cleaning most. This method works on a local window instead.

Colors are bucketed at the cluster radius, which is what makes "most common color"
meaningful when every pixel is unique. A pixel is replaced by the mean of its window's
dominant bucket only when at most **Isolation threshold** window pixels are within the
cluster radius of it.

That isolation test compares colors *by distance, never by shared bucket*. Bucket
boundaries fall arbitrarily, so on an anti-aliased edge each pixel of a one-pixel window
frame lands in a different bucket and the whole line reads as isolated pixels — which
erased frames, railings and corner lines on downscaled artwork. Measuring distance
instead keeps them: a thin line has near-identical pixels along its length, a stray tone
has none. Flat artwork comes out **byte-for-byte identical at every strength**.

**Isolation threshold** therefore doubles as a stylization dial. At `1` structure stays
faithful. At `3` or `4` organic texture such as foliage flattens into larger blocks while
architectural lines still hold, which is useful when only part of the image should read
as chunky pixel art — combine it with a brush selection and **Apply color effects to
selection only** to get both looks in one image.

`Cleanup strength` scales the cluster radius up to the `Cleanup color tolerance` ceiling.
On a noisy brick facade, raising it takes local variance down by roughly half. On a
perfectly smooth gradient the effect peaks and then eases off, because past a certain
radius the filter correctly concludes that neighboring tones are all the same color and
leaves them alone.

### Merge small regions

Fragments are connected groups of near-identical color, controlled by two tolerances:

| Control | Meaning |
| --- | --- |
| **Fragment grouping tolerance** | How similar two neighboring pixels must be to count as *the same fragment*. |
| **Cleanup color tolerance** | How different an adjacent fragment may be before it can *absorb* a small one. |

Grouping must stay below the merge tolerance, or a fragment is swallowed while grouping
and never becomes a candidate; the pipeline clamps it if you set it higher.

A fragment is merged only when all of the following hold:

- its area is at most the limit derived from **Cleanup strength**;
- a directly adjacent fragment is *strictly larger* than it;
- that neighbor's color is within **Cleanup color tolerance**.

Requiring a strictly larger neighbor is what stops two speckles from merging into each
other, and it keeps the strength control monotonic. `Cleanup strength` scales from
one-pixel speckles up to the advanced `Maximum region area` ceiling and never past it.

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

## Element detection

Detection answers "where are the windows, the walls, the foliage" in two levels:

| Level | What it is | How it is found |
| --- | --- | --- |
| **Class** | A material: lit brick, shaded brick, glass, foliage, balcony slab | k-means over the output colors |
| **Object** | One instance of a class, such as a single window | Connected components inside that class |

Press **Detect elements** and the panel lists each class with its color, its share of the
image, and how many separate objects it forms. Selecting a class makes it the active
selection, so everything that already works on a selection — local color effects, noise
cleanup, outlines — works per material. The `<` and `>` buttons step through the
individual objects of the selected class, with the whole class as the first entry.

Clustering runs on RGB. Dropping luminance was measured to be worse here: value is most
of what separates glass from render from stone in pixel art, and chromaticity-only
clustering fragmented facades along lines that did not follow the architecture.

Two things worth knowing:

- **A class is a material, not a name.** The tool reports "class 6, blue, 5.2% of the
  image, 87 objects"; it does not know that blue means glass. Naming it is your call.
- **Lighting creates separate classes.** A lit wall and a shaded wall are two classes,
  which is correct for pixel art — those faces are edited separately. Treat them as two
  selections rather than expecting one "wall" class.

### Recoloring a class

Select a class and press **Recolor class...** to give it a replacement color. What is
stored is a *mapping*, not an edit to a rendered image: on every render each visible pixel
is matched to its nearest class center and the mapped classes take their new color. Three
things follow from that.

- **Several classes can be recolored at once.** Give the glass one blue and the shaded
  wall another, and both hold. Editing through the global color controls could only ever
  apply the last change, which is what made the selection feel like it did nothing.
- **A recolor survives other settings.** Change the border, the cleanup or the palette and
  the classes keep their colors.
- **Untouched classes keep their own pixels.** Only the mapped classes are flattened to a
  single color; everything else keeps its per-pixel shading.

Recoloring runs after cleanup, so cleanup cannot undo it, and before the outlines, so
borders are drawn around the final colors. Alpha is never modified, and each recolor is
one undoable action. Detecting again clears the mapping, since it refers to the previous
clustering.

Detection is tied to the output geometry. If the target size changes it is reported as
stale rather than being applied to the wrong pixels.

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

With **Apply everything to selection only**, every stage that runs at the final
resolution is confined to the selected pixels:

| Restricted | Always global |
| --- | --- |
| Pixel-art reconstruction | Auto-crop |
| Invert, grayscale, RGB tints | Contrast |
| Average blend | Median noise removal |
| Palette reduction | Progressive resize |
| Noise cleanup | |
| Outer border and internal outlines | |

The right-hand column runs *before* the resize, where the mask coordinates do not exist
yet, so those stages cannot be restricted without guessing where the selection would land.

Note that restricting reconstruction is blunt: with the toggle on, the area outside the
selection is left unreconstructed. That is the point of the toggle, but it is worth
knowing that it changes more than the color stages. Alpha is preserved either way.

If the output dimensions change, remapping would be ambiguous, so the selection is
cleared and reported rather than silently applied to the wrong pixels. The pipeline
independently ignores a mask that does not match the image it is given.

## Average blend

**Average blend** drags every pixel towards a single average color: `0` leaves the image
alone, `100` flattens the area to one flat tone, and the steps in between are a smooth
progression. It is the blunt counterpart to recoloring a class — no detection is needed,
and it works on whatever is selected.

**Average color from** decides where the average is sampled:

- `Selection` uses the active class, object or brush mask, and falls back to the whole
  image when nothing is selected. This is what unifies one material without dragging in
  the colors of everything around it.
- `Whole image` always samples every visible pixel.

The sampling source and the affected area are separate: pair `Selection` with **Apply
everything to selection only** to sample *and* apply within one material, or sample from
the selection and apply globally to pull the whole image towards that material's tone.

Alpha is never modified, and the blend runs after class recoloring, so it can flatten a
recolored class further.

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
