
# Pixel Art Converter

Desktop utility for converting illustrations into crisp pixel-art assets.

## Features
- Open PNG/JPG/WebP/BMP/TIFF
- Auto-crop empty near-black margins
- Output by exact size, width, or height
- Progressive reduction
- Final Nearest Neighbor pixel-grid lock
- Original colors preserved by default
- Optional palette reduction (2–256 colors)
- Optional controlled dithering
- Optional isolated-noise cleanup
- Contrast adjustment
- Transparent background option
- PNG/WebP/JPEG export
- Live preview

## Recommended settings for your building assets
- Resize based on: `height`
- Height: `256`
- Auto-crop: ON
- Remove isolated noise: OFF initially
- Contrast: ~1.06
- Progressive reduction: ON
- Hard pixel grid: ON
- Palette: `0` (preserve colors)
- Dithering: OFF
- Transparent background: according to your Unity workflow

## Run on Windows
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   `pip install -r requirements.txt`
4. Run:
   `python pixel_art_converter.py`

## Build an .exe
Install PyInstaller:
`pip install pyinstaller`

Then:
`pyinstaller --noconfirm --onefile --windowed --name PixelArtConverter pixel_art_converter.py`

The executable will appear in `dist/`.
