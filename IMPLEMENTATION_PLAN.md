# Pixel Art Converter — Implementation Plan

## Scope and current assessment

This plan covers the next development cycle after **v2.4.1**. The application is a
single-file Tkinter/Pillow/NumPy desktop tool with an increasingly capable processing
pipeline. The recent changes are compatible with continued development: writable NumPy
buffers now protect the reconstruction, color-effect, outer-border, and internal-border
stages from Pillow's read-only array views.

The highest-value work is not a new visual effect. It is making the current controls
behave exactly as their labels promise, adding tests around the image pipeline, and then
improving performance and interaction quality.

## Findings from the current implementation

| Item | Status | Recommendation |
| --- | --- | --- |
| Writable NumPy buffers | Fixed in v2.4.1 | Keep `np.array(..., copy=True)` for every stage that mutates pixel data; cover it with tests. |
| Connected 4-neighbor segmentation | Implemented | Keep it. It runs after the target resize, which is the right place for 128/256/512 px sprites. |
| Color tolerance | Implemented | Keep seed-color distance as the first behavior; document it so users know gradual gradients can split into separate regions. |
| Minimum object area | Not effective yet | Fix first. Small regions are collected but their labels still participate in border creation. |
| Internal outline width/color | Implemented | Keep it; test width 1 and wider values on small sprite fixtures. |
| Outer border | Implemented, with a geometry limitation | Decide whether output size must remain fixed or whether an outside outline may expand the canvas. |
| Black / White monochrome modes | Ambiguous | Decide their intended visual difference before documenting them as separate effects. The current paths produce the same grayscale result. |
| Silhouette checkbox | Redundant | Remove it or give it a distinct, documented effect. Alpha is already preserved by the reconstruction pipeline. |
| Scrollable settings panel | Implemented | Improve wheel behavior on child controls and non-Windows platforms. |
| Live preview | Good baseline | Make typed Width, Height, and Palette values refresh predictably and validate invalid input. |
| Region-aware noise cleanup | Not implemented | Add after minimum-area filtering; it should remove only small, color-similar fragments without blurring edges. |
| Local selection and brush edits | Not implemented | Build a non-destructive mask model before exposing local effects. |
| Undo / redo history | Not implemented | Store compact command snapshots for settings and masks, never full rendered images. |

## Compatibility decisions

### 1. Apply the minimum-area filter to segmentation — recommended

This is fully compatible and should be the first code change. It makes the existing
`Minimum object area` control truthful without changing the UI or the image format.

Implementation approach:

1. Track `visited` pixels separately from `labels` while flood-filling a component.
2. When a component has fewer pixels than `min_area`, assign `-1` to its output labels.
3. Keep valid large components labelled with their region id.
4. Build internal boundaries only where both adjacent labels are non-negative and differ.
5. Remove the unused `regions` return value, or retain only a numeric region summary for
   diagnostics; do not retain every component's pixel tuple list in memory.

This prevents small isolated regions from being outlined and reduces memory pressure on
large sprites.

Acceptance checks:

- A 1-pixel region disappears from the internal-outline mask when `min_area` is 2.
- A 4-pixel region remains when `min_area` is 4.
- Transparent pixels never create a region or an internal outline.
- Two diagonal pixels remain separate under 4-neighbor connectivity.

### 2. Clarify monochrome behavior — product decision required

The current **Black** and **White** choices both produce grayscale, so users cannot see
a functional difference. Two valid directions are possible:

- **Recommended:** replace them with one `Grayscale` option. This accurately describes
  the current processing and has the lowest implementation risk.
- **Alternative:** make Black and White true single-color silhouette modes: replace every
  visible pixel with black or white respectively while keeping its original alpha. This
  is useful for masks and decal workflows, but discards luminance detail.

Do not add a third behavior until one of those semantics is chosen.

### 3. Define the outer-border canvas policy — product decision required

The current outer border can only occupy transparent pixels already inside the image. A
fully opaque sprite touching the canvas edge cannot receive a visible outside outline.

- **Recommended default:** preserve the requested target dimensions. Add an `Outside
  border needs transparent margin` hint when no margin is available, and keep current
  output size guarantees.
- **Optional future mode:** `Expand canvas for outer border`, which adds `2 × border
  width` pixels to the output dimensions. This must be opt-in because it changes the
  exported sprite size and Unity/import coordinates.

In either mode, retain the current writable-copy rule before assigning border pixels.

### 4. Improve interaction reliability — recommended

These changes are compatible and do not alter image output:

- Attach debounced variable traces to Width, Height, and Palette so typed changes update
  the preview. Validate empty or non-numeric intermediate text before calling `process`.
- Make ordinary mouse-wheel input scroll the preview vertically, Shift + wheel scroll
  horizontally, and Ctrl + wheel zoom. Apply the same Ctrl requirement on Linux
  `Button-4` / `Button-5` events.
- Bind configuration-panel scrolling while the pointer is over any child widget, not only
  the canvas/frame. Avoid a global binding that steals wheel input from the preview.
- Have `Fit` reset canvas scroll offsets to the top-left or center after changing zoom.
- Disable or visually de-emphasize reconstruction-only controls in `Pixel Perfect
  Resize` mode, while preserving their values for later use.

### 5. Preserve image semantics — recommended

Several older observations are still valid and can be addressed without redesigning the
application:

- Rename `Transparent background` to `Preserve transparency` unless a real background
  removal mode is added. At present, disabling alpha flattens transparent pixels to
  black; it does not identify and remove a solid background.
- Remove the redundant `Preserve silhouette` checkbox, or make it explicitly control a
  separate silhouette effect. Reconstruction currently keeps alpha exactly regardless of
  that value.
- Offer a safer crop choice: `Transparent only` versus `Transparent + near-black`.
  Near-black cropping can cut valid black artwork, whereas transparent-only cropping is
  predictable.
- Make progressive resize dimension-aware. Its intermediate stages are height-based;
  width-based output should use the dominant scale dimension so very wide or tall inputs
  receive equivalent treatment.

### 6. Add region-aware noise cleanup — recommended

This request is compatible with the current segmentation work and is more suitable for
pixel art than the existing general-purpose median filter. Its purpose is to remove small
stray color fragments from otherwise uniform regions (for example, off-yellow pixels
inside a yellow wall) while leaving intentional texture, edges, and alpha untouched.

Recommended processing order:

```text
crop -> resize -> reconstruction -> color effects -> optional palette reduction
     -> region-aware noise cleanup -> outer border -> internal outlines
```

The cleanup should reuse the final-resolution connected-region data rather than analyze
the original illustration. The first implementation should:

1. Find connected color regions using the current tolerance model.
2. Identify candidate regions below a size determined by the cleanup strength.
3. Consider only directly adjacent retained regions as replacement candidates.
4. Choose the closest adjacent region color and replace the candidate only when it is
   within the configured color tolerance.
5. Preserve alpha exactly and protect fragments that touch the transparent silhouette by
   default.

Suggested controls:

- `Clean small color noise` toggle.
- `Cleanup strength` (0–100), mapped to a conservative maximum removable region area.
- Advanced controls, initially collapsed: `Color tolerance`, `Maximum region area`, and
  `Protect silhouette`.

Strength must be deliberately conservative. It should remove one-pixel speckles first,
not merge an intended brick, window, or highlight into a wall. A preview comparison is
especially valuable for this feature.

### 7. Add local selection and brush edits — recommended as a separate feature set

Selection is compatible with the existing preview canvas, but it needs an explicit mask
model. Do not directly paint the source image: local edits should be replayable every
time the processing settings change.

Recommended incremental design:

1. **Rectangle selection:** convert canvas coordinates through zoom and scroll offsets
   into final-image coordinates; show an overlay without changing pixels.
2. **Connected-region selection:** add a click-to-select mode that uses the same final
   segmentation labels as internal outlines. This is the best first equivalent of a
   magic-wand tool for pixel art.
3. **Brush mask:** add and subtract binary mask strokes with configurable diameter. Store
   each completed stroke as one action, not one action per mouse movement.
4. **Local effects:** start with RGB tint, contrast, monochrome, and region-noise cleanup
   applied only where the active mask is set. Add fill/recolor and local outlines only
   after those basics are stable.

Selections should be represented in normalized output coordinates or remapped whenever
crop/target dimensions change. If remapping would be ambiguous, notify the user and
clear or disable the selection instead of applying an edit to the wrong pixels.

### 8. Add undo, redo, and an edit history — recommended

Use a command/history model rather than saving rendered image copies:

- A history entry records the changed settings, active selection metadata, and mask delta.
- Slider dragging creates one entry on release; typing creates one entry after validation;
  one brush stroke creates one entry.
- `Reset to defaults`, color-picker changes, and source-image changes are each one clear
  history action.
- Bind `Ctrl+Z` to undo, `Ctrl+Y` and `Ctrl+Shift+Z` to redo, and provide matching toolbar
  buttons with disabled states when no action is available.
- Start with a bounded history of 30–50 actions. Store masks as compact 1-bit or
  grayscale images and deltas where practical; do not store repeated full previews.

After the keyboard behavior is reliable, add a collapsible `Recent changes` panel. Each
entry should name the action (for example, `Red tint +12`, `Noise cleanup 35`, or
`Brush stroke`) and allow jumping back by undoing multiple later commands.

## Proposed implementation sequence

### Phase 0 — establish a safe baseline

1. Create a virtual environment and install `requirements.txt`.
2. Add a lightweight `tests/` suite using generated Pillow images; no new runtime
   dependency is needed if tests use Python's built-in `unittest`.
3. Verify the existing app opens, previews, exports PNG/WebP/JPEG, resets settings, and
   builds with PyInstaller.
4. Add a small test fixture set: transparent sprite, opaque edge-touching sprite,
   black-content sprite, two-region sprite, diagonal-pixel sprite, and a palette-limited
   sprite.

### Phase 1 — correctness release (recommended as v2.5)

1. Implement the real `min_area` label filtering described above.
2. Add direct unit tests for connected segmentation, border width, color tolerance,
   alpha preservation, and fixed output dimensions.
3. Add regression tests confirming every mutating image stage runs without a read-only
   array exception.
4. Decide and implement the monochrome semantics.
5. Decide and implement/document the outer-border canvas policy.
6. Update README examples and version after the behavior is verified.

Exit criteria: all processing tests pass; a 256 px sprite renders and exports without
exceptions; every exposed control has a distinct and documented effect.

### Phase 2 — interaction and usability release (recommended as v2.6)

1. Add validated, debounced preview refresh for typed numeric inputs.
2. Normalize mouse-wheel behavior across Windows and Linux, preserving Ctrl + wheel for
   zoom and using non-Ctrl wheel for scrolling.
3. Finish wheel support for the scrollable configuration panel.
4. Reset preview pan position after Fit and 100% actions.
5. Add tooltips or short inline help for color tolerance, minimum area, alpha behavior,
   and outer-border geometry.

Exit criteria: all settings can be reached and operated with keyboard/mouse, no valid
typed value needs an unrelated click to refresh preview, and zoom/scroll behavior is
consistent on supported platforms.

### Phase 3 — region cleanup and edit history release (recommended as v2.7)

1. Implement and test the conservative region-aware noise cleanup described above.
2. Expose cleanup strength first; keep advanced tolerance/area controls available but
   visually secondary.
3. Add the command-history core, Undo/Redo buttons, and keyboard shortcuts.
4. Ensure settings changes, Reset, color choices, and cleanup changes participate in
   history without recording preview renders.
5. Add rectangle and connected-region selection as non-destructive masks.

Exit criteria: a one-pixel color speckle can be removed without blurring a neighboring
edge; Ctrl+Z/Ctrl+Y reliably restore settings and masks; selection effects do not alter
the source image.

### Phase 4 — local brush, quality, and performance release (recommended as v2.8)

1. Add additive/subtractive brush-mask strokes and local RGB/contrast/monochrome effects.
2. Refactor segmentation to avoid retaining `pixels` for every accepted component.
3. Benchmark region segmentation and cleanup at 128, 256, and 512 px with one large
   region and many
   small regions. Record timings in a short developer note.
4. Keep the current vectorized reconstruction operations; profile before adding a new
   dependency such as Numba.
5. Show a short `Processing preview...` status if a render exceeds the debounce window.
6. Rework progressive reduction to respect the chosen resize dimension.

Exit criteria: a 512 px preview remains usable on the target machine, memory growth is
bounded, and no new runtime dependency is required without measured benefit.

## Candidate improvements after the current plan

These are useful enhancements, but should wait until the correctness work above is done.

- **Before/after comparison:** split view or temporary original overlay in the preview.
- **Local-effect history panel:** expose the existing undo stack as a named timeline once
  keyboard undo/redo and mask edits are stable.
- **Preset management:** save/load named processing presets as JSON, including settings
  for buildings, sprites, icons, and Unity exports.
- **Batch conversion:** queue multiple images using one preset and deterministic output
  names. This should be a separate module from the Tkinter callbacks.
- **Drag and drop:** accept an image dropped on the preview canvas.
- **Checkerboard alpha preview:** make transparency visually obvious instead of showing
  it against a flat dark canvas.
- **Undo/redo settings:** store snapshots of `vars` values, not image copies.
- **Export metadata:** remember the last export directory and optionally save a sidecar
  preset file for reproducibility.

## Architectural guidance

Before adding batch work or more visual effects, move the image pipeline into a small
testable module or class independent of Tkinter. Keep the GUI responsible for variables,
dialogs, and preview scheduling; keep processing functions pure where possible:

```text
source image + normalized settings
  -> crop / resize / reconstruction / color effects / outlines
  -> PIL output image
```

This separation will make unit testing, batch conversion, and future command-line usage
far easier without changing the desktop experience.

## Release and documentation checklist

1. Run syntax checks and the image-pipeline tests.
2. Manually test one transparent PNG and one opaque JPG.
3. Rebuild `dist/PixelArtConverter.exe` only after verification.
4. Update README version, controls, and limitations in English.
5. Keep generated `build/` and `dist/` directories out of Git.
6. Include the version in the commit message, for example:

```powershell
git commit -m "feat(v2.5): enforce minimum region area for internal outlines"
```
