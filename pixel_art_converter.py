
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path

class PixelArtConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Converter")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)

        self.src = None
        self.preview_img = None

        self.vars = {
            "width": tk.IntVar(value=0),
            "height": tk.IntVar(value=256),
            "fit_mode": tk.StringVar(value="height"),
            "crop": tk.BooleanVar(value=True),
            "crop_threshold": tk.IntVar(value=8),
            "cleanup": tk.BooleanVar(value=False),
            "contrast": tk.DoubleVar(value=1.06),
            "progressive": tk.BooleanVar(value=True),
            "pixel_final": tk.BooleanVar(value=True),
            "palette": tk.IntVar(value=0),
            "dither": tk.BooleanVar(value=False),
            "alpha": tk.BooleanVar(value=False),
            "output": tk.StringVar(value="PNG"),
        }
        self.build_ui()

    def build_ui(self):
        left = ttk.Frame(self.root, padding=12)
        left.pack(side="left", fill="y")

        right = ttk.Frame(self.root, padding=12)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="PIXEL ART CONVERTER", font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 12))
        ttk.Button(left, text="Open image...", command=self.open_image).pack(fill="x", pady=3)
        ttk.Button(left, text="Export...", command=self.export_image).pack(fill="x", pady=3)

        ttk.Separator(left).pack(fill="x", pady=12)

        ttk.Label(left, text="Output resolution", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        row = ttk.Frame(left); row.pack(fill="x", pady=5)
        ttk.Label(row, text="Width").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.vars["width"], width=8).grid(row=0, column=1, padx=5)
        ttk.Label(row, text="Height").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(row, textvariable=self.vars["height"], width=8).grid(row=1, column=1, padx=5)

        ttk.Label(left, text="Resize based on").pack(anchor="w", pady=(8,2))
        ttk.Combobox(left, textvariable=self.vars["fit_mode"],
                     values=["height", "width", "exact"], state="readonly").pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Label(left, text="Preservation", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        ttk.Checkbutton(left, text="Auto-crop empty background", variable=self.vars["crop"]).pack(anchor="w", pady=3)
        ttk.Label(left, text="Background threshold").pack(anchor="w")
        ttk.Scale(left, from_=0, to=40, variable=self.vars["crop_threshold"],
                  orient="horizontal").pack(fill="x")

        ttk.Checkbutton(left, text="Remove isolated noise", variable=self.vars["cleanup"]).pack(anchor="w", pady=3)

        ttk.Label(left, text="Contrast").pack(anchor="w", pady=(5,0))
        ttk.Scale(left, from_=0.8, to=1.3, variable=self.vars["contrast"],
                  orient="horizontal").pack(fill="x")

        ttk.Checkbutton(left, text="Progressive reduction", variable=self.vars["progressive"]).pack(anchor="w", pady=3)
        ttk.Checkbutton(left, text="Hard pixel grid (final nearest)", variable=self.vars["pixel_final"]).pack(anchor="w", pady=3)

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Label(left, text="Color", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        palette = ttk.Frame(left); palette.pack(fill="x", pady=4)
        ttk.Label(palette, text="Palette colors (0 = original)").pack(side="left")
        ttk.Spinbox(palette, from_=0, to=256, textvariable=self.vars["palette"], width=7).pack(side="right")
        ttk.Checkbutton(left, text="Controlled dithering", variable=self.vars["dither"]).pack(anchor="w", pady=3)

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Checkbutton(left, text="Transparent background", variable=self.vars["alpha"]).pack(anchor="w", pady=3)

        ttk.Label(left, text="Tip: use 0 palette colors for maximum color fidelity.",
                  wraplength=240, foreground="#555").pack(anchor="w", pady=8)

        ttk.Button(left, text="Reset", command=self.reset).pack(fill="x", pady=3)

        self.canvas = tk.Canvas(right, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = ttk.Label(right, text="Open an image to begin.", anchor="w")
        self.status.pack(fill="x", pady=(8,0))

    def reset(self):
        self.vars["width"].set(0)
        self.vars["height"].set(256)
        self.vars["fit_mode"].set("height")
        self.vars["crop"].set(True)
        self.vars["crop_threshold"].set(8)
        self.vars["cleanup"].set(False)
        self.vars["contrast"].set(1.06)
        self.vars["progressive"].set(True)
        self.vars["pixel_final"].set(True)
        self.vars["palette"].set(0)
        self.vars["dither"].set(False)
        self.vars["alpha"].set(False)
        self.render_preview()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.src = Image.open(path).convert("RGBA")
            self.status.config(text=f"Loaded: {Path(path).name} — {self.src.width}×{self.src.height}")
            self.render_preview()
        except Exception as e:
            messagebox.showerror("Open error", str(e))

    def crop_empty(self, img):
        rgba = np.array(img)
        rgb = rgba[:, :, :3].astype(np.int16)
        threshold = int(self.vars["crop_threshold"].get())
        # Near-black background detection.
        mask = np.max(rgb, axis=2) > threshold
        # Also retain already-transparent pixels as empty.
        if rgba.shape[2] == 4:
            mask &= rgba[:, :, 3] > 5
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return img
        pad = 1
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(img.width, int(xs.max()) + 1 + pad)
        y1 = min(img.height, int(ys.max()) + 1 + pad)
        return img.crop((x0,y0,x1,y1))

    def process(self):
        if self.src is None:
            raise ValueError("Open an image first.")

        img = self.src.copy()

        if self.vars["crop"].get():
            img = self.crop_empty(img)

        if self.vars["cleanup"].get():
            # Very light median cleanup; disabled by default to preserve details.
            rgb = img.convert("RGB").filter(ImageFilter.MedianFilter(3))
            if img.mode == "RGBA":
                a = img.getchannel("A")
                img = Image.merge("RGBA", (*rgb.split(), a))
            else:
                img = rgb

        contrast = float(self.vars["contrast"].get())
        if abs(contrast - 1.0) > 0.001:
            rgb = ImageEnhance.Contrast(img.convert("RGB")).enhance(contrast)
            if img.mode == "RGBA":
                img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))
            else:
                img = rgb

        mode = self.vars["fit_mode"].get()
        tw, th = int(self.vars["width"].get()), int(self.vars["height"].get())

        if mode == "exact":
            target = (max(1,tw), max(1,th))
        elif mode == "width":
            target_w = max(1, tw)
            target_h = max(1, round(img.height * target_w / img.width))
            target = (target_w, target_h)
        else:
            target_h = max(1, th)
            target_w = max(1, round(img.width * target_h / img.height))
            target = (target_w, target_h)

        if self.vars["progressive"].get():
            # Gather information progressively. Final stage can be locked to nearest.
            current = img
            steps = [768, 512, 384, 320, target[1]]
            for h in steps:
                if current.height > h and h > target[1]:
                    s = h / current.height
                    current = current.resize((max(1,round(current.width*s)), h), Image.Resampling.LANCZOS)
            img = current

        resample = Image.Resampling.NEAREST if self.vars["pixel_final"].get() else Image.Resampling.LANCZOS
        img = img.resize(target, resample)

        colors = int(self.vars["palette"].get())
        if colors > 0:
            colors = max(2, min(256, colors))
            method = Image.Quantize.MEDIANCUT
            dither = Image.Dither.FLOYDSTEINBERG if self.vars["dither"].get() else Image.Dither.NONE
            if img.mode == "RGBA":
                rgb = img.convert("RGB").quantize(colors=colors, method=method, dither=dither).convert("RGB")
                img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))
            else:
                img = img.convert("RGB").quantize(colors=colors, method=method, dither=dither).convert("RGB")

        if not self.vars["alpha"].get():
            img = img.convert("RGB")
        else:
            img = img.convert("RGBA")

        return img

    def render_preview(self):
        self.canvas.delete("all")
        if self.src is None:
            self.canvas.create_text(300, 250, text="Open an image", fill="white", font=("Segoe UI", 18))
            return
        try:
            img = self.process()
            preview = img.copy()
            maxw = max(200, self.canvas.winfo_width()-30)
            maxh = max(200, self.canvas.winfo_height()-30)
            preview.thumbnail((maxw,maxh), Image.Resampling.NEAREST)
            self.preview_img = ImageTk.PhotoImage(preview)
            self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                                     image=self.preview_img, anchor="center")
            self.status.config(text=f"Preview: {img.width}×{img.height} px | mode: {img.mode}")
        except Exception as e:
            self.status.config(text=f"Preview error: {e}")

    def export_image(self):
        if self.src is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("WebP", "*.webp"), ("JPEG", "*.jpg")]
        )
        if not path:
            return
        try:
            img = self.process()
            ext = Path(path).suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                img.convert("RGB").save(path, quality=95)
            else:
                img.save(path)
            messagebox.showinfo("Export complete", f"Saved:\n{path}\n\n{img.width}×{img.height} px")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass
    app = PixelArtConverter(root)
    root.bind("<Configure>", lambda e: app.render_preview() if e.widget == root else None)
    root.mainloop()
