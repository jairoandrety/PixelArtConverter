
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path


class PixelArtConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Converter v2")
        self.root.geometry("1240x800")
        self.root.minsize(1050, 700)
        self.src = None
        self.preview_img = None
        self.vars = {
            "width": tk.IntVar(value=0),
            "height": tk.IntVar(value=256),
            "fit_mode": tk.StringVar(value="height"),
            "mode": tk.StringVar(value="Pixel Art Reconstruction"),
            "crop": tk.BooleanVar(value=True),
            "crop_threshold": tk.IntVar(value=8),
            "cleanup": tk.BooleanVar(value=False),
            "contrast": tk.DoubleVar(value=1.06),
            "progressive": tk.BooleanVar(value=True),
            "pixel_final": tk.BooleanVar(value=True),
            "palette": tk.IntVar(value=0),
            "dither": tk.BooleanVar(value=False),
            "alpha": tk.BooleanVar(value=False),
            "clusters": tk.BooleanVar(value=True),
            "detail": tk.BooleanVar(value=True),
            "edges": tk.BooleanVar(value=True),
            "silhouette": tk.BooleanVar(value=True),
            "color_clusters": tk.BooleanVar(value=True),
            "cluster_strength": tk.IntVar(value=35),
            "detail_strength": tk.IntVar(value=25),
            "edge_strength": tk.IntVar(value=40),
            "color_strength": tk.IntVar(value=30),
        }
        self.build_ui()

    def build_ui(self):
        left = ttk.Frame(self.root, padding=12)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self.root, padding=12)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="PIXEL ART CONVERTER",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(left, text="v2 • Pixel Perfect + Reconstruction",
                  foreground="#666").pack(anchor="w", pady=(0, 12))
        ttk.Button(left, text="Open image...", command=self.open_image).pack(fill="x", pady=3)
        ttk.Button(left, text="Export...", command=self.export_image).pack(fill="x", pady=3)
        ttk.Separator(left).pack(fill="x", pady=12)

        ttk.Label(left, text="PROCESSING MODE",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        mode = ttk.Combobox(
            left, textvariable=self.vars["mode"],
            values=["Pixel Perfect Resize", "Pixel Art Reconstruction"],
            state="readonly")
        mode.pack(fill="x", pady=5)
        mode.bind("<<ComboboxSelected>>", lambda e: self.render_preview())
        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="OUTPUT RESOLUTION",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        row = ttk.Frame(left)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="Width").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.vars["width"], width=8).grid(row=0, column=1, padx=6)
        ttk.Label(row, text="Height").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(row, textvariable=self.vars["height"], width=8).grid(row=1, column=1, padx=6)
        ttk.Label(left, text="Resize based on").pack(anchor="w", pady=(7, 2))
        fit = ttk.Combobox(left, textvariable=self.vars["fit_mode"],
                           values=["height", "width", "exact"], state="readonly")
        fit.pack(fill="x")
        fit.bind("<<ComboboxSelected>>", lambda e: self.render_preview())
        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="BASE PROCESSING",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Checkbutton(left, text="Auto-crop empty background",
                        variable=self.vars["crop"],
                        command=self.render_preview).pack(anchor="w", pady=2)
        ttk.Label(left, text="Background threshold").pack(anchor="w")
        ttk.Scale(left, from_=0, to=40, variable=self.vars["crop_threshold"],
                  orient="horizontal", command=lambda _: self.render_preview()).pack(fill="x")
        ttk.Checkbutton(left, text="Remove isolated noise",
                        variable=self.vars["cleanup"],
                        command=self.render_preview).pack(anchor="w", pady=2)
        ttk.Label(left, text="Contrast").pack(anchor="w")
        ttk.Scale(left, from_=0.80, to=1.30, variable=self.vars["contrast"],
                  orient="horizontal", command=lambda _: self.render_preview()).pack(fill="x")
        ttk.Checkbutton(left, text="Progressive reduction",
                        variable=self.vars["progressive"],
                        command=self.render_preview).pack(anchor="w", pady=2)
        ttk.Checkbutton(left, text="Hard pixel grid (final nearest)",
                        variable=self.vars["pixel_final"],
                        command=self.render_preview).pack(anchor="w", pady=2)

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="PIXEL ART RECONSTRUCTION",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")

        for text, key in [
            ("Optimize pixel clusters", "clusters"),
            ("Simplify small details", "detail"),
            ("Optimize edges", "edges"),
            ("Preserve silhouette", "silhouette"),
            ("Preserve color clusters", "color_clusters"),
        ]:
            ttk.Checkbutton(left, text=text, variable=self.vars[key],
                            command=self.render_preview).pack(anchor="w", pady=2)

        self.add_scale(left, "Cluster strength", "cluster_strength")
        self.add_scale(left, "Detail simplification", "detail_strength")
        self.add_scale(left, "Edge optimization", "edge_strength")
        self.add_scale(left, "Color cluster strength", "color_strength")

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="COLOR",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        row = ttk.Frame(left)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Palette (0 = original)").pack(side="left")
        ttk.Spinbox(row, from_=0, to=256, textvariable=self.vars["palette"],
                    width=7, command=self.render_preview).pack(side="right")
        ttk.Checkbutton(left, text="Controlled dithering",
                        variable=self.vars["dither"],
                        command=self.render_preview).pack(anchor="w", pady=2)
        ttk.Checkbutton(left, text="Transparent background",
                        variable=self.vars["alpha"],
                        command=self.render_preview).pack(anchor="w", pady=2)
        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Button(left, text="Reset settings", command=self.reset).pack(fill="x", pady=3)
        ttk.Label(left, text="Recommended: Reconstruction + Palette 0 + Dither OFF.",
                  wraplength=260, foreground="#555").pack(anchor="w", pady=8)

        self.canvas = tk.Canvas(right, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = ttk.Label(right, text="Open an image to begin.", anchor="w")
        self.status.pack(fill="x", pady=(8, 0))
        self.canvas.bind("<Configure>", lambda e: self.render_preview())

    def add_scale(self, parent, label, key):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(3, 0))
        ttk.Scale(parent, from_=0, to=100, variable=self.vars[key],
                  orient="horizontal",
                  command=lambda _: self.render_preview()).pack(fill="x")

    def reset(self):
        defaults = {
            "width": 0, "height": 256, "fit_mode": "height",
            "mode": "Pixel Art Reconstruction", "crop": True,
            "crop_threshold": 8, "cleanup": False, "contrast": 1.06,
            "progressive": True, "pixel_final": True, "palette": 0,
            "dither": False, "alpha": False, "clusters": True,
            "detail": True, "edges": True, "silhouette": True,
            "color_clusters": True, "cluster_strength": 35,
            "detail_strength": 25, "edge_strength": 40,
            "color_strength": 30
        }
        for k, v in defaults.items():
            self.vars[k].set(v)
        self.render_preview()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
            ("All files", "*.*")])
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
        mask = np.max(rgb, axis=2) > threshold
        mask &= rgba[:, :, 3] > 5
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return img
        return img.crop((
            max(0, int(xs.min()) - 1), max(0, int(ys.min()) - 1),
            min(img.width, int(xs.max()) + 2), min(img.height, int(ys.max()) + 2)
        ))

    def calculate_target(self, img):
        mode = self.vars["fit_mode"].get()
        tw, th = int(self.vars["width"].get()), int(self.vars["height"].get())
        if mode == "exact":
            return max(1, tw), max(1, th)
        if mode == "width":
            tw = max(1, tw)
            return tw, max(1, round(img.height * tw / img.width))
        th = max(1, th)
        return max(1, round(img.width * th / img.height)), th

    def progressive_resize(self, img, target):
        if not self.vars["progressive"].get():
            return img
        current = img
        heights = [768, 512, 384, 320, target[1]]
        for h in heights:
            if current.height > h and h > target[1]:
                s = h / current.height
                current = current.resize(
                    (max(1, round(current.width * s)), h),
                    Image.Resampling.LANCZOS)
        return current

    def reconstruct_pixel_art(self, img):
        rgba = np.array(img.convert("RGBA"))
        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3]
        h, w = rgb.shape[:2]
        if h < 3 or w < 3:
            return img

        original = rgb.copy()

        # Color clusters: subtle local quantization, not global palette reduction.
        cs = self.vars["color_strength"].get() / 100.0
        if self.vars["color_clusters"].get() and cs > 0:
            step = max(1, int(round(1 + cs * 10)))
            q = np.round(rgb / step) * step
            rgb = original * (1 - cs * 0.22) + q * (cs * 0.22)

        # Pixel cluster cleanup: only isolated pixels that disagree strongly.
        ks = self.vars["cluster_strength"].get() / 100.0
        if self.vars["clusters"].get() and ks > 0:
            arr = np.round(rgb).astype(np.int16)
            padded = np.pad(arr, ((1,1),(1,1),(0,0)), mode="edge")
            result = arr.copy()
            for y in range(h):
                for x in range(w):
                    p = arr[y, x].astype(np.float32)
                    nb = padded[y:y+3, x:x+3].reshape(-1, 3).astype(np.float32)
                    d = np.linalg.norm(nb - p, axis=1)
                    similar = nb[d < 35]
                    if len(similar) >= 4:
                        med = np.median(similar, axis=0)
                        result[y, x] = p * (1 - ks * 0.20) + med * (ks * 0.20)
            rgb = result.astype(np.float32)

        # Small-detail simplification: very weak isolated variations only.
        ds = self.vars["detail_strength"].get() / 100.0
        if self.vars["detail"].get() and ds > 0:
            arr = np.round(rgb).astype(np.float32)
            padded = np.pad(arr, ((1,1),(1,1),(0,0)), mode="edge")
            result = arr.copy()
            for y in range(h):
                for x in range(w):
                    p = arr[y, x]
                    nb = padded[y:y+3, x:x+3].reshape(-1, 3)
                    med = np.median(nb, axis=0)
                    diff = np.linalg.norm(p - med)
                    if diff < 22:
                        result[y, x] = p * (1 - ds * 0.12) + med * (ds * 0.12)
            rgb = result

        # Edge optimization: tiny adjustment, never blur.
        es = self.vars["edge_strength"].get() / 100.0
        if self.vars["edges"].get() and es > 0:
            arr = np.round(rgb).astype(np.float32)
            out = arr.copy()
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    p = arr[y, x]
                    dx = np.linalg.norm(arr[y, x-1] - arr[y, x+1])
                    dy = np.linalg.norm(arr[y-1, x] - arr[y+1, x])
                    if max(dx, dy) > 55:
                        if dx > dy:
                            a, b = arr[y, x-1], arr[y, x+1]
                        else:
                            a, b = arr[y-1, x], arr[y+1, x]
                        target = a if np.linalg.norm(p-a) < np.linalg.norm(p-b) else b
                        out[y, x] = p * (1 - es * 0.08) + target * (es * 0.08)
            rgb = out

        # Preserve alpha/silhouette boundaries.
        rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        result = Image.fromarray(rgb, "RGB").convert("RGBA")
        result.putalpha(Image.fromarray(alpha, "L"))
        return result

    def process(self):
        if self.src is None:
            raise ValueError("Open an image first.")
        img = self.src.copy()

        if self.vars["crop"].get():
            img = self.crop_empty(img)

        if self.vars["cleanup"].get():
            rgb = img.convert("RGB").filter(ImageFilter.MedianFilter(3))
            img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

        contrast = float(self.vars["contrast"].get())
        if abs(contrast - 1.0) > 0.001:
            rgb = ImageEnhance.Contrast(img.convert("RGB")).enhance(contrast)
            img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

        target = self.calculate_target(img)
        img = self.progressive_resize(img, target)

        resample = Image.Resampling.NEAREST if self.vars["pixel_final"].get() else Image.Resampling.LANCZOS
        img = img.resize(target, resample)

        if self.vars["mode"].get() == "Pixel Art Reconstruction":
            img = self.reconstruct_pixel_art(img)

        colors = int(self.vars["palette"].get())
        if colors > 0:
            colors = max(2, min(256, colors))
            dither = Image.Dither.FLOYDSTEINBERG if self.vars["dither"].get() else Image.Dither.NONE
            rgb = img.convert("RGB").quantize(
                colors=colors, method=Image.Quantize.MEDIANCUT, dither=dither).convert("RGB")
            img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

        return img.convert("RGBA" if self.vars["alpha"].get() else "RGB")

    def render_preview(self):
        if self.src is None:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="Open an image",
                                    fill="white", font=("Segoe UI", 18))
            return
        try:
            img = self.process()
            preview = img.copy()
            maxw = max(200, self.canvas.winfo_width() - 30)
            maxh = max(200, self.canvas.winfo_height() - 30)
            preview.thumbnail((maxw, maxh), Image.Resampling.NEAREST)
            self.preview_img = ImageTk.PhotoImage(preview)
            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=self.preview_img, anchor="center")
            self.status.config(
                text=f"Preview: {img.width}×{img.height} px | {self.vars['mode'].get()} | {img.mode}")
        except Exception as e:
            self.status.config(text=f"Preview error: {e}")

    def export_image(self):
        if self.src is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("WebP", "*.webp"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            img = self.process()
            ext = Path(path).suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                img.convert("RGB").save(path, quality=95, subsampling=0)
            else:
                img.save(path)
            messagebox.showinfo("Export complete",
                                f"Saved:\n{path}\n\n{img.width}×{img.height} px")
        except Exception as e:
            messagebox.showerror("Export error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass
    PixelArtConverter(root)
    root.mainloop()
