
# Pixel Art Converter v2.2

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
- Shows the current numeric value beside every processing slider.
- Provides a crisp nearest-neighbor preview with 25%–800% zoom.
- Includes **Fit** and **100%** preview buttons, Ctrl + mouse-wheel zoom, and scrollbars
  for enlarged previews.
- Exports PNG, WebP, and JPEG.

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
