
# Pixel Art Converter v2.4

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
- Provides progressive reduction and a final *Nearest Neighbor* pixel-grid lock.
- Adjusts contrast, cleans isolated noise, and preserves the original alpha channel exactly.
- Includes reconstruction controls for clusters, details, edges, silhouette, and color.
- Optionally reduces the palette to 2–256 colors, with optional dithering.
- Includes invert, monochrome, and independent red, green, and blue tint controls.
- Keeps every configuration accessible through a scrollable settings panel.
- Provides configurable outer borders with a selectable width and color.
- Provides internal color-group outline controls with a selectable width, threshold,
  minimum-area setting, and color.
- Shows the current numeric value beside every processing slider.
- Provides a crisp nearest-neighbor preview with 25%–800% zoom.
- Includes **Fit** and **100%** preview buttons, Ctrl + mouse-wheel zoom, and scrollbars
  for enlarged previews.
- Exports PNG, WebP, and JPEG.

## Color and outline tools

- **Invert colors:** reverses every RGB channel while retaining alpha.
- **Monochrome:** offers Black and White monochrome modes.
- **RGB tints:** independently enable and adjust red, green, and blue channels from
  `-100` to `100`, with the toggle, slider, and numeric value on one row.
- **Outer border:** adds a selectable border around the visible alpha silhouette.
- **Internal object borders:** uses 4-neighbor connected-region segmentation after the
  image reaches its target resolution. Pixels join a region when they are connected
  and within the selected RGB color tolerance; the interface provides configurable
  minimum-area and outline-width controls for the resulting internal outlines.

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
