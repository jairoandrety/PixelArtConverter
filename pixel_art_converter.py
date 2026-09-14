
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageFilter, ImageEnhance, ImageTk
import numpy as np
from pathlib import Path


class PixelArtConverter:
    DEFAULTS = {
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

        "invert": False,
        "monochrome": "Off",
        "tint_r": False,
        "tint_g": False,
        "tint_b": False,
        "tint_r_amount": 0,
        "tint_g_amount": 0,
        "tint_b_amount": 0,

        "border": False,
        "border_width": 1,
        "border_color": "#000000",

        "object_borders": False,
        "object_border_width": 1,
        "object_threshold": 18,
        "object_min_area": 3,
        "object_border_color": "#000000",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Converter v2.4")
        self.root.geometry("1370x900")
        self.root.minsize(1150, 760)

        self.src = None
        self.preview_source = None
        self.preview_img = None
        self.render_job = None

        self.zoom = tk.DoubleVar(value=100)
        self.vars = {
            key: (
                tk.BooleanVar(value=value)
                if isinstance(value, bool)
                else tk.IntVar(value=value)
                if isinstance(value, int)
                else tk.DoubleVar(value=value)
                if isinstance(value, float)
                else tk.StringVar(value=value)
            )
            for key, value in self.DEFAULTS.items()
        }

        self.build_ui()

    # ---------------------------------------------------------
    # UI helpers
    # ---------------------------------------------------------

    def section(self, parent, text):
        ttk.Label(
            parent, text=text,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(8, 4))

    def add_scale(self, parent, label, key, minimum, maximum,
                  integer=True, decimals=0):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(3, 0))

        ttk.Label(row, text=label).pack(side="left")

        value_label = ttk.Label(row, width=7, anchor="e")
        value_label.pack(side="right")

        def update_value(*_):
            value = self.vars[key].get()
            value_label.config(
                text=str(int(round(value)))
                if integer
                else f"{float(value):.{decimals}f}"
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

    def add_tint_control(self, parent, label, toggle_key, amount_key):
        # Toggle + slider + numeric value on one horizontal row.
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        ttk.Checkbutton(
            row,
            text=label,
            variable=self.vars[toggle_key],
            command=self.schedule_preview
        ).pack(side="left", padx=(0, 5))

        slider = ttk.Scale(
            row,
            from_=-100,
            to=100,
            variable=self.vars[amount_key],
            orient="horizontal",
            command=lambda _: self.schedule_preview()
        )
        slider.pack(side="left", fill="x", expand=True)

        value = ttk.Label(row, width=6, anchor="e")

        def update(*_):
            value.config(
                text=f"{int(round(self.vars[amount_key].get())):+d}"
            )

        self.vars[amount_key].trace_add("write", update)
        update()
        value.pack(side="left", padx=(5, 0))

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def build_ui(self):
        # Scrollable configuration panel so every setting remains accessible
        # even on smaller screens.
        left_outer = ttk.Frame(self.root)
        left_outer.pack(side="left", fill="y")

        left_canvas = tk.Canvas(
            left_outer,
            width=335,
            highlightthickness=0,
            borderwidth=0
        )
        left_scroll = ttk.Scrollbar(
            left_outer,
            orient="vertical",
            command=left_canvas.yview
        )
        left_canvas.configure(
            yscrollcommand=left_scroll.set
        )

        left_canvas.pack(side="left", fill="y", expand=False)
        left_scroll.pack(side="right", fill="y")

        left = ttk.Frame(left_canvas, padding=12)
        left_window = left_canvas.create_window(
            (0, 0),
            window=left,
            anchor="nw"
        )

        def update_left_scrollregion(event=None):
            left_canvas.configure(
                scrollregion=left_canvas.bbox("all")
            )

        def resize_left_content(event):
            left_canvas.itemconfigure(
                left_window,
                width=event.width
            )

        left.bind("<Configure>", update_left_scrollregion)
        left_canvas.bind("<Configure>", resize_left_content)

        def wheel_config(event):
            left_canvas.yview_scroll(
                int(-1 * (event.delta / 120)),
                "units"
            )

        left.bind("<MouseWheel>", wheel_config)
        left_canvas.bind("<MouseWheel>", wheel_config)

        right = ttk.Frame(self.root, padding=12)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(
            left,
            text="PIXEL ART CONVERTER",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w")

        ttk.Label(
            left,
            text="v2.4 • Connected Region + Object Outline Tools",
            foreground="#666"
        ).pack(anchor="w", pady=(0, 10))

        ttk.Button(
            left, text="Open image...",
            command=self.open_image
        ).pack(fill="x", pady=3)

        ttk.Button(
            left, text="Export...",
            command=self.export_image
        ).pack(fill="x", pady=3)

        self.section(left, "PROCESSING MODE")

        mode = ttk.Combobox(
            left,
            textvariable=self.vars["mode"],
            values=[
                "Pixel Perfect Resize",
                "Pixel Art Reconstruction"
            ],
            state="readonly"
        )
        mode.pack(fill="x")
        mode.bind(
            "<<ComboboxSelected>>",
            lambda e: self.schedule_preview()
        )

        self.section(left, "OUTPUT RESOLUTION")

        # Width and Height are now side-by-side.
        resolution = ttk.Frame(left)
        resolution.pack(fill="x", pady=3)

        ttk.Label(resolution, text="Width").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(
            resolution,
            textvariable=self.vars["width"],
            width=9
        ).grid(row=0, column=1, padx=(4, 10))

        ttk.Label(resolution, text="Height").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Entry(
            resolution,
            textvariable=self.vars["height"],
            width=9
        ).grid(row=0, column=3, padx=4)

        fit = ttk.Combobox(
            left,
            textvariable=self.vars["fit_mode"],
            values=["height", "width", "exact"],
            state="readonly"
        )
        fit.pack(fill="x", pady=(5, 0))
        fit.bind(
            "<<ComboboxSelected>>",
            lambda e: self.schedule_preview()
        )

        self.section(left, "BASE PROCESSING")

        ttk.Checkbutton(
            left,
            text="Auto-crop empty background",
            variable=self.vars["crop"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        self.add_scale(
            left, "Background threshold",
            "crop_threshold", 0, 40
        )

        ttk.Checkbutton(
            left,
            text="Remove isolated noise",
            variable=self.vars["cleanup"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        self.add_scale(
            left, "Contrast",
            "contrast", 0.80, 1.30,
            integer=False, decimals=2
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

        self.section(left, "PIXEL ART RECONSTRUCTION")

        for text, key in [
            ("Optimize pixel clusters", "clusters"),
            ("Simplify small details", "detail"),
            ("Optimize edges", "edges"),
            ("Preserve silhouette", "silhouette"),
            ("Preserve color clusters", "color_clusters"),
        ]:
            ttk.Checkbutton(
                left, text=text,
                variable=self.vars[key],
                command=self.schedule_preview
            ).pack(anchor="w", pady=2)

        self.add_scale(
            left, "Cluster strength",
            "cluster_strength", 0, 100
        )
        self.add_scale(
            left, "Detail simplification",
            "detail_strength", 0, 100
        )
        self.add_scale(
            left, "Edge optimization",
            "edge_strength", 0, 100
        )
        self.add_scale(
            left, "Color cluster strength",
            "color_strength", 0, 100
        )

        self.section(left, "COLOR")

        row = ttk.Frame(left)
        row.pack(fill="x", pady=3)

        ttk.Label(
            row, text="Palette (0 = original)"
        ).pack(side="left")

        ttk.Spinbox(
            row,
            from_=0,
            to=256,
            textvariable=self.vars["palette"],
            width=7,
            command=self.schedule_preview
        ).pack(side="right")

        ttk.Checkbutton(
            left,
            text="Invert colors",
            variable=self.vars["invert"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        mono = ttk.Combobox(
            left,
            textvariable=self.vars["monochrome"],
            values=["Off", "Black", "White"],
            state="readonly"
        )
        mono.pack(fill="x", pady=2)
        mono.bind(
            "<<ComboboxSelected>>",
            lambda e: self.schedule_preview()
        )

        ttk.Label(
            left,
            text="RGB Tints",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(5, 0))

        self.add_tint_control(
            left, "Red",
            "tint_r", "tint_r_amount"
        )
        self.add_tint_control(
            left, "Green",
            "tint_g", "tint_g_amount"
        )
        self.add_tint_control(
            left, "Blue",
            "tint_b", "tint_b_amount"
        )

        ttk.Checkbutton(
            left,
            text="Controlled dithering",
            variable=self.vars["dither"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            left,
            text="Transparent background",
            variable=self.vars["alpha"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        self.section(left, "OUTLINE")

        ttk.Checkbutton(
            left,
            text="Add outer border",
            variable=self.vars["border"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        self.add_scale(
            left, "Border width",
            "border_width", 1, 10
        )

        border_row = ttk.Frame(left)
        border_row.pack(fill="x", pady=3)

        ttk.Label(border_row, text="Border color").pack(side="left")

        self.border_color_preview = tk.Label(
            border_row,
            text="  ",
            width=4,
            bg=self.vars["border_color"].get(),
            relief="solid",
            borderwidth=1
        )
        self.border_color_preview.pack(side="right")

        ttk.Button(
            border_row,
            text="Choose",
            command=self.choose_border_color
        ).pack(side="right", padx=5)

        ttk.Checkbutton(
            left,
            text="Detect objects and outline internal regions",
            variable=self.vars["object_borders"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        self.add_scale(
            left, "Object border width",
            "object_border_width", 1, 6
        )

        self.add_scale(
            left, "Color group threshold",
            "object_threshold", 1, 60
        )

        self.add_scale(
            left, "Minimum object area",
            "object_min_area", 1, 100
        )

        obj_row = ttk.Frame(left)
        obj_row.pack(fill="x", pady=3)

        ttk.Label(
            obj_row,
            text="Object border color"
        ).pack(side="left")

        self.object_color_preview = tk.Label(
            obj_row,
            text="  ",
            width=4,
            bg=self.vars["object_border_color"].get(),
            relief="solid",
            borderwidth=1
        )
        self.object_color_preview.pack(side="right")

        ttk.Button(
            obj_row,
            text="Choose",
            command=self.choose_object_color
        ).pack(side="right", padx=5)

        ttk.Separator(left).pack(fill="x", pady=8)

        # Explicit reset button.
        ttk.Button(
            left,
            text="RESET TO DEFAULTS",
            command=self.reset
        ).pack(fill="x", pady=3)

        ttk.Label(
            left,
            text=(
                "Recommended for 256 px buildings: "
                "Reconstruction, Palette 0, Dither OFF."
            ),
            wraplength=300,
            foreground="#555"
        ).pack(anchor="w", pady=7)

        # -----------------------------------------------------
        # Preview
        # -----------------------------------------------------

        ttk.Label(
            right,
            text="PREVIEW",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        zoom_row = ttk.Frame(right)
        zoom_row.pack(fill="x", pady=(5, 8))

        ttk.Button(
            zoom_row,
            text="Fit",
            command=self.fit_zoom,
            width=7
        ).pack(side="left")

        ttk.Button(
            zoom_row,
            text="100%",
            command=lambda: self.set_zoom(100),
            width=7
        ).pack(side="left", padx=4)

        ttk.Label(
            zoom_row,
            text="Zoom"
        ).pack(side="left", padx=(12, 4))

        ttk.Scale(
            zoom_row,
            from_=25,
            to=800,
            variable=self.zoom,
            orient="horizontal",
            command=lambda _: self.render_preview_image()
        ).pack(side="left", fill="x", expand=True)

        self.zoom_value = ttk.Label(
            zoom_row,
            text="100%",
            width=7,
            anchor="e"
        )
        self.zoom_value.pack(side="left", padx=(6, 0))

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#202020",
            highlightthickness=0
        )
        self.canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

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
        self.status.pack(
            fill="x",
            pady=(8, 0)
        )

        self.canvas.bind(
            "<Configure>",
            lambda e: self.render_preview_image()
        )
        self.canvas.bind(
            "<MouseWheel>",
            self.mouse_wheel_zoom
        )
        self.canvas.bind(
            "<Button-4>",
            self.mouse_wheel_zoom
        )
        self.canvas.bind(
            "<Button-5>",
            self.mouse_wheel_zoom
        )

    # ---------------------------------------------------------
    # Colors
    # ---------------------------------------------------------

    def choose_border_color(self):
        color = colorchooser.askcolor(
            initialcolor=self.vars["border_color"].get(),
            title="Choose outer border color"
        )[1]

        if color:
            self.vars["border_color"].set(color)
            self.border_color_preview.config(bg=color)
            self.schedule_preview()

    def choose_object_color(self):
        color = colorchooser.askcolor(
            initialcolor=self.vars["object_border_color"].get(),
            title="Choose object border color"
        )[1]

        if color:
            self.vars["object_border_color"].set(color)
            self.object_color_preview.config(bg=color)
            self.schedule_preview()

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):
        for key, value in self.DEFAULTS.items():
            self.vars[key].set(value)

        self.zoom.set(100)

        self.border_color_preview.config(
            bg=self.vars["border_color"].get()
        )
        self.object_color_preview.config(
            bg=self.vars["object_border_color"].get()
        )

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

        cw = max(
            1,
            self.canvas.winfo_width() - 20
        )
        ch = max(
            1,
            self.canvas.winfo_height() - 20
        )

        zw = cw / self.preview_source.width * 100
        zh = ch / self.preview_source.height * 100

        value = min(zw, zh)
        value = max(
            25,
            min(800, value)
        )

        self.zoom.set(value)
        self.render_preview_image()

    def mouse_wheel_zoom(self, event):
        if self.preview_source is None:
            return "break"

        ctrl = bool(event.state & 0x0004)

        if ctrl or event.num in (4, 5):
            current = float(self.zoom.get())

            if (
                getattr(event, "delta", 0) > 0
                or event.num == 4
            ):
                new = min(
                    800,
                    current * 1.15
                )
            else:
                new = max(
                    25,
                    current / 1.15
                )

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
                int(round(
                    self.preview_source.width * scale
                ))
            )
            h = max(
                1,
                int(round(
                    self.preview_source.height * scale
                ))
            )

            preview = self.preview_source.resize(
                (w, h),
                Image.Resampling.NEAREST
            )

            self.preview_img = ImageTk.PhotoImage(
                preview
            )

            self.canvas.delete("all")

            self.canvas.create_image(
                10,
                10,
                image=self.preview_img,
                anchor="nw"
            )

            self.canvas.configure(
                scrollregion=(
                    0, 0,
                    w + 20,
                    h + 20
                )
            )

            self.zoom_value.config(
                text=f"{scale * 100:.0f}%"
            )

        except Exception as e:
            self.status.config(
                text=f"Preview display error: {e}"
            )

    # ---------------------------------------------------------
    # Open / crop / resize
    # ---------------------------------------------------------

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                (
                    "Images",
                    "*.png *.jpg *.jpeg *.bmp "
                    "*.webp *.tif *.tiff"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            self.src = Image.open(
                path
            ).convert("RGBA")

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

        rgb = rgba[:, :, :3].astype(
            np.int16
        )

        threshold = int(
            round(
                self.vars["crop_threshold"].get()
            )
        )

        mask = np.max(
            rgb,
            axis=2
        ) > threshold

        mask &= rgba[:, :, 3] > 5

        ys, xs = np.where(mask)

        if len(xs) == 0:
            return img

        return img.crop((
            max(
                0,
                int(xs.min()) - 1
            ),
            max(
                0,
                int(ys.min()) - 1
            ),
            min(
                img.width,
                int(xs.max()) + 2
            ),
            min(
                img.height,
                int(ys.max()) + 2
            )
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
                    round(
                        img.height * tw / img.width
                    )
                )
            )

        th = max(1, th)

        return (
            max(
                1,
                round(
                    img.width * th / img.height
                )
            ),
            th
        )

    def progressive_resize(self, img, target):
        if not self.vars["progressive"].get():
            return img

        current = img

        for height in (
            768, 512, 384, 320
        ):
            if (
                current.height > height
                and height > target[1]
            ):
                scale = (
                    height / current.height
                )

                current = current.resize(
                    (
                        max(
                            1,
                            round(
                                current.width * scale
                            )
                        ),
                        height
                    ),
                    Image.Resampling.LANCZOS
                )

        return current

    # ---------------------------------------------------------
    # Pixel-art reconstruction
    # ---------------------------------------------------------

    def reconstruct_pixel_art(self, img):
        rgba = np.asarray(
            img.convert("RGBA"),
            dtype=np.float32
        )

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        h, w = rgb.shape[:2]

        if h < 3 or w < 3:
            return img

        # Color cluster preservation.
        cs = (
            float(
                self.vars["color_strength"].get()
            ) / 100.0
        )

        if (
            self.vars["color_clusters"].get()
            and cs > 0
        ):
            step = max(
                1.0,
                1.0 + cs * 8.0
            )

            quantized = (
                np.round(rgb / step) * step
            )

            rgb = (
                rgb * (1.0 - cs * 0.15)
                + quantized * (cs * 0.15)
            )

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

        cluster_strength = (
            float(
                self.vars["cluster_strength"].get()
            ) / 100.0
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

            amount *= (
                cluster_strength * 0.18
            )

            rgb = (
                rgb * (1.0 - amount[:, :, None])
                + median * amount[:, :, None]
            )

        detail_strength = (
            float(
                self.vars["detail_strength"].get()
            ) / 100.0
        )

        if (
            self.vars["detail"].get()
            and detail_strength > 0
        ):
            amount = np.clip(
                (18.0 - distance) / 18.0,
                0.0,
                1.0
            )

            amount *= (
                detail_strength * 0.10
            )

            rgb = (
                rgb * (1.0 - amount[:, :, None])
                + median * amount[:, :, None]
            )

        # Stable H x W edge optimization.
        edge_strength = (
            float(
                self.vars["edge_strength"].get()
            ) / 100.0
        )

        if (
            self.vars["edges"].get()
            and edge_strength > 0
        ):
            left = padded[1:h+1, 0:w]
            right = padded[1:h+1, 2:w+2]
            up = padded[0:h, 1:w+1]
            down = padded[2:h+2, 1:w+1]

            hd = np.linalg.norm(
                left - right,
                axis=2
            )
            vd = np.linalg.norm(
                up - down,
                axis=2
            )

            dl = np.linalg.norm(
                rgb - left,
                axis=2
            )
            dr = np.linalg.norm(
                rgb - right,
                axis=2
            )
            du = np.linalg.norm(
                rgb - up,
                axis=2
            )
            dd = np.linalg.norm(
                rgb - down,
                axis=2
            )

            horizontal_target = np.where(
                (dl < dr)[:, :, None],
                left,
                right
            )

            vertical_target = np.where(
                (du < dd)[:, :, None],
                up,
                down
            )

            target = np.where(
                (hd >= vd)[:, :, None],
                horizontal_target,
                vertical_target
            )

            edge_amount = np.clip(
                (np.maximum(hd, vd) - 55.0)
                / 100.0,
                0.0,
                1.0
            )

            edge_amount *= (
                edge_strength * 0.06
            )

            rgb = (
                rgb * (1.0 - edge_amount[:, :, None])
                + target * edge_amount[:, :, None]
            )

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
    # Color effects
    # ---------------------------------------------------------

    def apply_color_effects(self, img):
        rgba = np.asarray(
            img.convert("RGBA"),
            dtype=np.float32
        )

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        if self.vars["invert"].get():
            rgb = 255.0 - rgb

        mono = self.vars["monochrome"].get()

        if mono != "Off":
            luminance = (
                0.299 * rgb[:, :, 0]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            )

            if mono == "Black":
                # Black-and-white using luminance.
                # Black remains black; bright areas become white.
                rgb = np.repeat(
                    luminance[:, :, None],
                    3,
                    axis=2
                )

            elif mono == "White":
                # White monochrome: brightness is represented
                # as white with variable intensity.
                rgb = np.repeat(
                    luminance[:, :, None],
                    3,
                    axis=2
                )

        # RGB tint controls.
        # Each enabled component adds/subtracts from its channel.
        for enabled_key, amount_key, channel in [
            ("tint_r", "tint_r_amount", 0),
            ("tint_g", "tint_g_amount", 1),
            ("tint_b", "tint_b_amount", 2),
        ]:
            if self.vars[enabled_key].get():
                amount = float(
                    self.vars[amount_key].get()
                )

                rgb[:, :, channel] = np.clip(
                    rgb[:, :, channel] + amount,
                    0,
                    255
                )

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
    # Outer border
    # ---------------------------------------------------------

    def add_outer_border(self, img):
        if not self.vars["border"].get():
            return img

        rgba = np.asarray(
            img.convert("RGBA")
        )

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        width = int(
            round(
                self.vars["border_width"].get()
            )
        )

        color = self.hex_to_rgb(
            self.vars["border_color"].get()
        )

        # Border follows visible content/alpha.
        visible = alpha > 5

        # A border is created around the silhouette.
        for _ in range(max(1, width)):
            p = np.pad(
                visible,
                ((1, 1), (1, 1)),
                mode="constant",
                constant_values=False
            )

            expanded = (
                p[0:-2, 0:-2]
                | p[0:-2, 1:-1]
                | p[0:-2, 2:]
                | p[1:-1, 0:-2]
                | p[1:-1, 2:]
                | p[2:, 0:-2]
                | p[2:, 1:-1]
                | p[2:, 2:]
            )

            border_mask = expanded & ~visible

            # Keep original pixels untouched.
            rgb[border_mask] = color
            alpha[border_mask] = 255
            visible |= border_mask

        result = Image.fromarray(
            np.dstack((rgb, alpha)).astype(np.uint8),
            "RGBA"
        )

        return result

    # ---------------------------------------------------------
    # Object detection / internal outlines
    # ---------------------------------------------------------

    def segment_color_regions(self, rgb, alpha, threshold, min_area):
        """
        Connected-component segmentation for pixel-art objects.

        Pixels are considered part of the same region when their RGB
        distance is <= threshold and they are 4-connected. This is more
        faithful than simply drawing every color boundary because broad
        areas of a similar color become one object while genuinely
        different regions remain separate.

        The implementation works on the already-downscaled image, so it
        stays practical for 128/256/512 px sprites.
        """
        h, w = rgb.shape[:2]
        valid = alpha > 5
        labels = np.full((h, w), -1, dtype=np.int32)
        regions = []

        label = 0
        threshold_sq = float(threshold) ** 2

        # Scanline flood-fill with a stack. We compare each candidate
        # against the seed color of its connected region.
        for y in range(h):
            for x in range(w):
                if not valid[y, x] or labels[y, x] != -1:
                    continue

                seed = rgb[y, x].astype(np.int16)
                stack = [(y, x)]
                labels[y, x] = label
                pixels = []

                while stack:
                    cy, cx = stack.pop()
                    pixels.append((cy, cx))

                    if cy > 0:
                        ny, nx = cy - 1, cx
                        if (
                            valid[ny, nx]
                            and labels[ny, nx] == -1
                        ):
                            d = rgb[ny, nx].astype(np.int16) - seed
                            if int(d @ d) <= threshold_sq:
                                labels[ny, nx] = label
                                stack.append((ny, nx))

                    if cy + 1 < h:
                        ny, nx = cy + 1, cx
                        if (
                            valid[ny, nx]
                            and labels[ny, nx] == -1
                        ):
                            d = rgb[ny, nx].astype(np.int16) - seed
                            if int(d @ d) <= threshold_sq:
                                labels[ny, nx] = label
                                stack.append((ny, nx))

                    if cx > 0:
                        ny, nx = cy, cx - 1
                        if (
                            valid[ny, nx]
                            and labels[ny, nx] == -1
                        ):
                            d = rgb[ny, nx].astype(np.int16) - seed
                            if int(d @ d) <= threshold_sq:
                                labels[ny, nx] = label
                                stack.append((ny, nx))

                    if cx + 1 < w:
                        ny, nx = cy, cx + 1
                        if (
                            valid[ny, nx]
                            and labels[ny, nx] == -1
                        ):
                            d = rgb[ny, nx].astype(np.int16) - seed
                            if int(d @ d) <= threshold_sq:
                                labels[ny, nx] = label
                                stack.append((ny, nx))

                if len(pixels) >= min_area:
                    regions.append((label, pixels))

                label += 1

        return labels, regions

    def add_object_borders(self, img):
        if not self.vars["object_borders"].get():
            return img

        rgba = np.asarray(img.convert("RGBA"))
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3].copy()

        h, w = rgb.shape[:2]
        if h < 2 or w < 2:
            return img

        threshold = int(round(self.vars["object_threshold"].get()))
        width = int(round(self.vars["object_border_width"].get()))
        min_area = int(round(self.vars["object_min_area"].get()))

        border_color = self.hex_to_rgb(
            self.vars["object_border_color"].get()
        )

        labels, regions = self.segment_color_regions(
            rgb.astype(np.uint8),
            alpha,
            threshold,
            min_area
        )

        # A region is outlined only against a different region. This avoids
        # producing noisy outlines inside a single smooth color cluster.
        boundary = np.zeros((h, w), dtype=bool)

        left_labels = np.full_like(labels, -2)
        left_labels[:, 1:] = labels[:, :-1]

        right_labels = np.full_like(labels, -2)
        right_labels[:, :-1] = labels[:, 1:]

        up_labels = np.full_like(labels, -2)
        up_labels[1:, :] = labels[:-1, :]

        down_labels = np.full_like(labels, -2)
        down_labels[:-1, :] = labels[1:, :]

        valid = labels >= 0

        boundary |= valid & (left_labels >= 0) & (left_labels != labels)
        boundary |= valid & (right_labels >= 0) & (right_labels != labels)
        boundary |= valid & (up_labels >= 0) & (up_labels != labels)
        boundary |= valid & (down_labels >= 0) & (down_labels != labels)

        # Expand the internal outline by the requested pixel width.
        for _ in range(max(1, width) - 1):
            p = np.pad(
                boundary,
                ((1, 1), (1, 1)),
                mode="constant",
                constant_values=False
            )
            boundary = (
                p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
                p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
                p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
            )

        rgb[boundary] = border_color
        alpha[boundary] = 255

        return Image.fromarray(
            np.dstack((rgb, alpha)).astype(np.uint8),
            "RGBA"
        )

    @staticmethod
    def hex_to_rgb(value):
        value = value.lstrip("#")

        if len(value) != 6:
            return np.array(
                [0, 0, 0],
                dtype=np.uint8
            )

        return np.array(
            [
                int(value[0:2], 16),
                int(value[2:4], 16),
                int(value[4:6], 16)
            ],
            dtype=np.uint8
        )

    # ---------------------------------------------------------
    # Full pipeline
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
            img = self.reconstruct_pixel_art(
                img
            )

        img = self.apply_color_effects(img)

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

        img = self.add_outer_border(
            img
        )

        img = self.add_object_borders(
            img
        )

        if self.vars["alpha"].get():
            return img.convert("RGBA")

        # If transparency is disabled, flatten transparent
        # pixels to black instead of producing accidental
        # alpha artifacts.
        background = Image.new(
            "RGB",
            img.size,
            (0, 0, 0)
        )
        background.paste(
            img,
            mask=img.getchannel("A")
        )

        return background

    # ---------------------------------------------------------
    # Preview / export
    # ---------------------------------------------------------

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

            ext = Path(
                path
            ).suffix.lower()

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
