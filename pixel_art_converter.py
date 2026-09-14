
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageFilter, ImageEnhance, ImageTk
import numpy as np
from pathlib import Path


class PixelArtConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Converter v2.2")
        self.root.geometry("1320x850")
        self.root.minsize(1100, 720)

        self.src = None
        self.preview_source = None
        self.preview_img = None
        self.render_job = None

        self.zoom = tk.DoubleVar(value=100)

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

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def build_ui(self):
        left = ttk.Frame(self.root, padding=12)
        left.pack(side="left", fill="y")

        right = ttk.Frame(self.root, padding=12)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="PIXEL ART CONVERTER",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(left, text="v2.2 • Stable Fast Reconstruction",
                  foreground="#666").pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="Open image...", command=self.open_image).pack(fill="x", pady=3)
        ttk.Button(left, text="Export...", command=self.export_image).pack(fill="x", pady=3)

        ttk.Separator(left).pack(fill="x", pady=12)

        ttk.Label(left, text="PROCESSING MODE",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")

        mode = ttk.Combobox(
            left,
            textvariable=self.vars["mode"],
            values=["Pixel Perfect Resize", "Pixel Art Reconstruction"],
            state="readonly"
        )
        mode.pack(fill="x", pady=5)
        mode.bind("<<ComboboxSelected>>", lambda e: self.schedule_preview())

        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="OUTPUT RESOLUTION",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(left)
        row.pack(fill="x", pady=5)

        ttk.Label(row, text="Width").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.vars["width"], width=8).grid(
            row=0, column=1, padx=6)

        ttk.Label(row, text="Height").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(row, textvariable=self.vars["height"], width=8).grid(
            row=1, column=1, padx=6)

        ttk.Label(left, text="Resize based on").pack(anchor="w", pady=(7, 2))

        fit = ttk.Combobox(
            left,
            textvariable=self.vars["fit_mode"],
            values=["height", "width", "exact"],
            state="readonly"
        )
        fit.pack(fill="x")
        fit.bind("<<ComboboxSelected>>", lambda e: self.schedule_preview())

        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="BASE PROCESSING",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")

        ttk.Checkbutton(
            left,
            text="Auto-crop empty background",
            variable=self.vars["crop"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        self.add_scale(
            left, "Background threshold", "crop_threshold",
            0, 40, integer=True
        )

        ttk.Checkbutton(
            left,
            text="Remove isolated noise",
            variable=self.vars["cleanup"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        self.add_scale(
            left, "Contrast", "contrast",
            0.80, 1.30, integer=False, decimals=2
        )

        ttk.Checkbutton(
            left,
            text="Progressive reduction",
            variable=self.vars["progressive"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            left,
            text="Hard pixel grid (final nearest)",
            variable=self.vars["pixel_final"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

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
            ttk.Checkbutton(
                left, text=text, variable=self.vars[key],
                command=self.schedule_preview
            ).pack(anchor="w", pady=2)

        self.add_scale(left, "Cluster strength", "cluster_strength", 0, 100)
        self.add_scale(left, "Detail simplification", "detail_strength", 0, 100)
        self.add_scale(left, "Edge optimization", "edge_strength", 0, 100)
        self.add_scale(left, "Color cluster strength", "color_strength", 0, 100)

        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="COLOR",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(left)
        row.pack(fill="x", pady=3)

        ttk.Label(row, text="Palette (0 = original)").pack(side="left")

        palette = ttk.Spinbox(
            row, from_=0, to=256,
            textvariable=self.vars["palette"],
            width=7,
            command=self.schedule_preview
        )
        palette.pack(side="right")

        ttk.Checkbutton(
            left, text="Controlled dithering",
            variable=self.vars["dither"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            left, text="Transparent background",
            variable=self.vars["alpha"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Button(left, text="Reset settings",
                   command=self.reset).pack(fill="x", pady=3)

        ttk.Label(
            left,
            text="Recommended: Reconstruction + Palette 0 + Dither OFF.",
            wraplength=280,
            foreground="#555"
        ).pack(anchor="w", pady=8)

        # -----------------------------------------------------
        # Preview
        # -----------------------------------------------------

        ttk.Label(right, text="PREVIEW",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")

        zoom_row = ttk.Frame(right)
        zoom_row.pack(fill="x", pady=(5, 8))

        ttk.Button(
            zoom_row, text="Fit",
            command=self.fit_zoom,
            width=7
        ).pack(side="left")

        ttk.Button(
            zoom_row, text="100%",
            command=lambda: self.set_zoom(100),
            width=7
        ).pack(side="left", padx=4)

        ttk.Label(zoom_row, text="Zoom").pack(side="left", padx=(12, 4))

        ttk.Scale(
            zoom_row,
            from_=25,
            to=800,
            variable=self.zoom,
            orient="horizontal",
            command=lambda _: self.render_preview_image()
        ).pack(side="left", fill="x", expand=True)

        self.zoom_value = ttk.Label(
            zoom_row, text="100%", width=7, anchor="e"
        )
        self.zoom_value.pack(side="left", padx=(6, 0))

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#202020",
            highlightthickness=0
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        self.vbar = ttk.Scrollbar(
            canvas_frame,
            orient="vertical",
            command=self.canvas.yview
        )
        self.vbar.pack(side="right", fill="y")

        self.hbar = ttk.Scrollbar(
            right,
            orient="horizontal",
            command=self.canvas.xview
        )
        self.hbar.pack(fill="x")

        self.canvas.configure(
            xscrollcommand=self.hbar.set,
            yscrollcommand=self.vbar.set
        )

        self.status = ttk.Label(
            right,
            text="Open an image to begin.",
            anchor="w"
        )
        self.status.pack(fill="x", pady=(8, 0))

        self.canvas.bind(
            "<Configure>",
            lambda e: self.render_preview_image()
        )

        self.canvas.bind("<MouseWheel>", self.mouse_wheel_zoom)
        self.canvas.bind("<Button-4>", self.mouse_wheel_zoom)
        self.canvas.bind("<Button-5>", self.mouse_wheel_zoom)

    def add_scale(
        self, parent, label, key,
        minimum, maximum,
        integer=True, decimals=0
    ):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(3, 0))

        ttk.Label(row, text=label).pack(side="left")

        value_label = ttk.Label(
            row, width=7, anchor="e"
        )
        value_label.pack(side="right")

        def update_value(*_):
            value = self.vars[key].get()

            if integer:
                value_label.config(
                    text=str(int(round(value)))
                )
            else:
                value_label.config(
                    text=f"{float(value):.{decimals}f}"
                )

        self.vars[key].trace_add("write", update_value)
        update_value()

        ttk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            variable=self.vars[key],
            orient="horizontal",
            command=lambda _: self.schedule_preview()
        ).pack(fill="x")

    # ---------------------------------------------------------
    # Settings
    # ---------------------------------------------------------

    def reset(self):
        defaults = {
            "width": 0,
            "height": 256,
            "fit_mode": "height",
            "mode": "Pixel Art Reconstruction",

            "crop": True,
            "crop_threshold": 8,
            "cleanup": False,
            "contrast": 1.06,
            "progressive": True,
            "pixel_final": True,

            "palette": 0,
            "dither": False,
            "alpha": False,

            "clusters": True,
            "detail": True,
            "edges": True,
            "silhouette": True,
            "color_clusters": True,

            "cluster_strength": 35,
            "detail_strength": 25,
            "edge_strength": 40,
            "color_strength": 30,
        }

        for key, value in defaults.items():
            self.vars[key].set(value)

        self.zoom.set(100)
        self.schedule_preview()

    # ---------------------------------------------------------
    # Zoom
    # ---------------------------------------------------------

    def set_zoom(self, value):
        self.zoom.set(float(value))
        self.render_preview_image()

    def fit_zoom(self):
        if self.preview_source is None:
            return

        cw = max(1, self.canvas.winfo_width() - 20)
        ch = max(1, self.canvas.winfo_height() - 20)

        zw = cw / self.preview_source.width * 100
        zh = ch / self.preview_source.height * 100

        value = min(zw, zh)
        value = max(25, min(800, value))

        self.zoom.set(value)
        self.render_preview_image()

    def mouse_wheel_zoom(self, event):
        if self.preview_source is None:
            return "break"

        ctrl = bool(event.state & 0x0004)

        if ctrl or event.num in (4, 5):
            current = float(self.zoom.get())

            if getattr(event, "delta", 0) > 0 or event.num == 4:
                new = min(800, current * 1.15)
            else:
                new = max(25, current / 1.15)

            self.zoom.set(new)
            self.render_preview_image()
            return "break"

        return None

    def render_preview_image(self):
        if self.preview_source is None:
            return

        try:
            scale = float(self.zoom.get()) / 100.0

            w = max(
                1,
                int(round(self.preview_source.width * scale))
            )
            h = max(
                1,
                int(round(self.preview_source.height * scale))
            )

            preview = self.preview_source.resize(
                (w, h),
                Image.Resampling.NEAREST
            )

            self.preview_img = ImageTk.PhotoImage(preview)

            self.canvas.delete("all")

            self.canvas.create_image(
                10, 10,
                image=self.preview_img,
                anchor="nw"
            )

            self.canvas.configure(
                scrollregion=(0, 0, w + 20, h + 20)
            )

            self.zoom_value.config(
                text=f"{scale * 100:.0f}%"
            )

        except Exception as e:
            self.status.config(
                text=f"Preview display error: {e}"
            )

    # ---------------------------------------------------------
    # Image handling
    # ---------------------------------------------------------

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                (
                    "Images",
                    "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            self.src = Image.open(path).convert("RGBA")

            self.status.config(
                text=(
                    f"Loaded: {Path(path).name} — "
                    f"{self.src.width}×{self.src.height}"
                )
            )

            self.schedule_preview()

        except Exception as e:
            messagebox.showerror(
                "Open error",
                str(e)
            )

    def crop_empty(self, img):
        rgba = np.asarray(img)

        rgb = rgba[:, :, :3].astype(np.int16)

        threshold = int(
            round(self.vars["crop_threshold"].get())
        )

        mask = np.max(rgb, axis=2) > threshold
        mask &= rgba[:, :, 3] > 5

        ys, xs = np.where(mask)

        if len(xs) == 0:
            return img

        return img.crop((
            max(0, int(xs.min()) - 1),
            max(0, int(ys.min()) - 1),
            min(img.width, int(xs.max()) + 2),
            min(img.height, int(ys.max()) + 2)
        ))

    def calculate_target(self, img):
        mode = self.vars["fit_mode"].get()

        tw = int(self.vars["width"].get())
        th = int(self.vars["height"].get())

        if mode == "exact":
            return max(1, tw), max(1, th)

        if mode == "width":
            tw = max(1, tw)

            return (
                tw,
                max(
                    1,
                    round(img.height * tw / img.width)
                )
            )

        th = max(1, th)

        return (
            max(
                1,
                round(img.width * th / img.height)
            ),
            th
        )

    def progressive_resize(self, img, target):
        if not self.vars["progressive"].get():
            return img

        current = img

        for height in (768, 512, 384, 320):
            if (
                current.height > height
                and height > target[1]
            ):
                scale = height / current.height

                current = current.resize(
                    (
                        max(
                            1,
                            round(current.width * scale)
                        ),
                        height
                    ),
                    Image.Resampling.LANCZOS
                )

        return current

    # ---------------------------------------------------------
    # Stable vectorized reconstruction
    # ---------------------------------------------------------

    def reconstruct_pixel_art(self, img):
        """
        Stable reconstruction pass.

        All operations preserve exactly H x W dimensions.
        No Python loop over individual pixels.
        """

        rgba = np.asarray(
            img.convert("RGBA"),
            dtype=np.float32
        )

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        h, w = rgb.shape[:2]

        if h < 3 or w < 3:
            return img

        # -----------------------------------------------------
        # Color clusters
        # -----------------------------------------------------

        color_strength = (
            float(self.vars["color_strength"].get())
            / 100.0
        )

        if (
            self.vars["color_clusters"].get()
            and color_strength > 0
        ):
            # Mild quantization. It deliberately does not
            # collapse the image to a tiny global palette.
            step = max(
                1.0,
                1.0 + color_strength * 8.0
            )

            quantized = (
                np.round(rgb / step) * step
            )

            rgb = (
                rgb * (1.0 - color_strength * 0.15)
                + quantized * (color_strength * 0.15)
            )

        # -----------------------------------------------------
        # 3x3 neighborhood
        # -----------------------------------------------------

        padded = np.pad(
            rgb,
            ((1, 1), (1, 1), (0, 0)),
            mode="edge"
        )

        neighbors = np.stack(
            [
                padded[0:h, 0:w],
                padded[0:h, 1:w+1],
                padded[0:h, 2:w+2],
                padded[1:h+1, 0:w],
                padded[1:h+1, 2:w+2],
                padded[2:h+2, 0:w],
                padded[2:h+2, 1:w+1],
                padded[2:h+2, 2:w+2],
            ],
            axis=0
        )

        median = np.median(
            neighbors,
            axis=0
        )

        distance = np.linalg.norm(
            rgb - median,
            axis=2
        )

        # -----------------------------------------------------
        # Cluster optimization
        # -----------------------------------------------------

        cluster_strength = (
            float(self.vars["cluster_strength"].get())
            / 100.0
        )

        if (
            self.vars["clusters"].get()
            and cluster_strength > 0
        ):
            amount = np.clip(
                (35.0 - distance) / 35.0,
                0.0,
                1.0
            )

            amount *= cluster_strength * 0.18

            rgb = (
                rgb * (1.0 - amount[:, :, None])
                + median * amount[:, :, None]
            )

        # -----------------------------------------------------
        # Detail simplification
        # -----------------------------------------------------

        detail_strength = (
            float(self.vars["detail_strength"].get())
            / 100.0
        )

        if (
            self.vars["detail"].get()
            and detail_strength > 0
        ):
            # Only remove very small color fluctuations.
            amount = np.clip(
                (18.0 - distance) / 18.0,
                0.0,
                1.0
            )

            amount *= detail_strength * 0.10

            rgb = (
                rgb * (1.0 - amount[:, :, None])
                + median * amount[:, :, None]
            )

        # -----------------------------------------------------
        # Edge optimization
        #
        # IMPORTANT:
        # Everything stays H x W.
        # No interior/whole-image broadcasting mismatch.
        # -----------------------------------------------------

        edge_strength = (
            float(self.vars["edge_strength"].get())
            / 100.0
        )

        if (
            self.vars["edges"].get()
            and edge_strength > 0
        ):
            left = padded[1:h+1, 0:w]
            right = padded[1:h+1, 2:w+2]
            up = padded[0:h, 1:w+1]
            down = padded[2:h+2, 1:w+1]

            horizontal_difference = np.linalg.norm(
                left - right,
                axis=2
            )

            vertical_difference = np.linalg.norm(
                up - down,
                axis=2
            )

            edge_map = np.maximum(
                horizontal_difference,
                vertical_difference
            )

            horizontal_target = np.where(
                (
                    np.linalg.norm(
                        rgb - left,
                        axis=2
                    )
                    <
                    np.linalg.norm(
                        rgb - right,
                        axis=2
                    )
                )[:, :, None],
                left,
                right
            )

            vertical_target = np.where(
                (
                    np.linalg.norm(
                        rgb - up,
                        axis=2
                    )
                    <
                    np.linalg.norm(
                        rgb - down,
                        axis=2
                    )
                )[:, :, None],
                up,
                down
            )

            target = np.where(
                (
                    horizontal_difference
                    >= vertical_difference
                )[:, :, None],
                horizontal_target,
                vertical_target
            )

            # Only affect strong edges.
            edge_amount = np.clip(
                (edge_map - 55.0) / 100.0,
                0.0,
                1.0
            )

            edge_amount *= edge_strength * 0.06

            rgb = (
                rgb * (1.0 - edge_amount[:, :, None])
                + target * edge_amount[:, :, None]
            )

        # -----------------------------------------------------
        # Preserve silhouette:
        # alpha is never changed.
        # -----------------------------------------------------

        rgb = np.clip(
            np.round(rgb),
            0,
            255
        ).astype(np.uint8)

        result = Image.fromarray(
            rgb,
            "RGB"
        ).convert("RGBA")

        result.putalpha(
            Image.fromarray(
                alpha.astype(np.uint8),
                "L"
            )
        )

        return result

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    def process(self):
        if self.src is None:
            raise ValueError(
                "Open an image first."
            )

        img = self.src.copy()

        if self.vars["crop"].get():
            img = self.crop_empty(img)

        if self.vars["cleanup"].get():
            rgb = img.convert(
                "RGB"
            ).filter(
                ImageFilter.MedianFilter(3)
            )

            img = Image.merge(
                "RGBA",
                (
                    *rgb.split(),
                    img.getchannel("A")
                )
            )

        contrast = float(
            self.vars["contrast"].get()
        )

        if abs(contrast - 1.0) > 0.001:
            rgb = ImageEnhance.Contrast(
                img.convert("RGB")
            ).enhance(contrast)

            img = Image.merge(
                "RGBA",
                (
                    *rgb.split(),
                    img.getchannel("A")
                )
            )

        target = self.calculate_target(img)

        img = self.progressive_resize(
            img,
            target
        )

        resample = (
            Image.Resampling.NEAREST
            if self.vars["pixel_final"].get()
            else Image.Resampling.LANCZOS
        )

        img = img.resize(
            target,
            resample
        )

        if (
            self.vars["mode"].get()
            == "Pixel Art Reconstruction"
        ):
            img = self.reconstruct_pixel_art(img)

        colors = int(
            self.vars["palette"].get()
        )

        if colors > 0:
            colors = max(
                2,
                min(256, colors)
            )

            dither = (
                Image.Dither.FLOYDSTEINBERG
                if self.vars["dither"].get()
                else Image.Dither.NONE
            )

            rgb = img.convert(
                "RGB"
            ).quantize(
                colors=colors,
                method=Image.Quantize.MEDIANCUT,
                dither=dither
            ).convert("RGB")

            img = Image.merge(
                "RGBA",
                (
                    *rgb.split(),
                    img.getchannel("A")
                )
            )

        if self.vars["alpha"].get():
            return img.convert("RGBA")

        return img.convert("RGB")

    def render_preview(self):
        self.render_job = None

        if self.src is None:
            self.canvas.delete("all")

            self.canvas.create_text(
                400,
                300,
                text="Open an image",
                fill="white",
                font=("Segoe UI", 18)
            )

            return

        try:
            img = self.process()

            self.preview_source = img

            self.status.config(
                text=(
                    f"Preview: {img.width}×{img.height} px | "
                    f"{self.vars['mode'].get()} | "
                    f"Zoom {self.zoom.get():.0f}%"
                )
            )

            self.render_preview_image()

        except Exception as e:
            self.status.config(
                text=f"Preview error: {e}"
            )

    def schedule_preview(self):
        if self.render_job is not None:
            try:
                self.root.after_cancel(
                    self.render_job
                )
            except Exception:
                pass

        # Small debounce prevents the processor from
        # running dozens of times while dragging a slider.
        self.render_job = self.root.after(
            100,
            self.render_preview
        )

    def export_image(self):
        if self.src is None:
            messagebox.showwarning(
                "No image",
                "Open an image first."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("WebP", "*.webp"),
                ("JPEG", "*.jpg")
            ]
        )

        if not path:
            return

        try:
            img = self.process()

            ext = Path(path).suffix.lower()

            if ext in (".jpg", ".jpeg"):
                img.convert("RGB").save(
                    path,
                    quality=95,
                    subsampling=0
                )
            else:
                img.save(path)

            messagebox.showinfo(
                "Export complete",
                (
                    f"Saved:\n{path}\n\n"
                    f"{img.width}×{img.height} px"
                )
            )

        except Exception as e:
            messagebox.showerror(
                "Export error",
                str(e)
            )


if __name__ == "__main__":
    root = tk.Tk()

    try:
        root.tk.call(
            "tk",
            "scaling",
            1.0
        )
    except Exception:
        pass

    PixelArtConverter(root)
    root.mainloop()
