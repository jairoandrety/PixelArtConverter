
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path

from pixel_pipeline import Pipeline, Settings


class Tooltip:
    """Lightweight hover help for Tk widgets."""

    DELAY_MS = 450

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        self.job = None

        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _event=None):
        self.cancel()
        self.job = self.widget.after(self.DELAY_MS, self.show)

    def cancel(self):
        if self.job is not None:
            try:
                self.widget.after_cancel(self.job)
            except Exception:
                pass
            self.job = None

    def show(self):
        self.job = None

        if self.window is not None:
            return

        try:
            x = self.widget.winfo_rootx() + 14
            y = (
                self.widget.winfo_rooty()
                + self.widget.winfo_height()
                + 6
            )
        except Exception:
            return

        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")

        tk.Label(
            self.window,
            text=self.text,
            justify="left",
            wraplength=270,
            background="#ffffe0",
            foreground="#333333",
            relief="solid",
            borderwidth=1,
            padx=7,
            pady=5
        ).pack()

    def hide(self, _event=None):
        self.cancel()

        if self.window is not None:
            self.window.destroy()
            self.window = None


class PixelArtConverter:
    # Typed numeric fields are validated before a preview is scheduled,
    # so a partially typed value never reaches the pipeline.
    NUMERIC_LIMITS = {
        "width": (0, 4096, "Width"),
        "height": (0, 4096, "Height"),
        "palette": (0, 256, "Palette"),
    }

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

        "noise_cleanup": False,
        "noise_strength": 30,
        "noise_tolerance": 20,
        "noise_group_tolerance": 2,
        "noise_max_area": 12,
        "noise_protect": True,

        "selection_mode": "Rectangle",
        "brush_size": 3,
        "local_effects": False,
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Converter v2.8")
        self.root.geometry("1370x900")
        self.root.minsize(1150, 760)

        self.src = None
        self.preview_source = None
        self.preview_img = None
        self.render_job = None
        self.outer_border_margin_warning = False
        self.undo_stack = []
        self.redo_stack = []
        self.restoring_history = False
        self.history_hold = False
        self.selection = None
        self.selection_shape = None
        self.selection_anchor = None
        self.last_render_seconds = 0.0
        self.noise_pixels_changed = 0

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

        self.history_state = (dict(self.DEFAULTS), None)
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
                  integer=True, decimals=0, help_text=None):
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

        scale = ttk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            variable=self.vars[key],
            orient="horizontal",
            command=lambda _: self.schedule_preview()
        )
        scale.pack(fill="x")
        scale.bind("<ButtonPress-1>", self.hold_history, add="+")
        scale.bind("<ButtonRelease-1>", self.release_history, add="+")

        if help_text:
            Tooltip(row, help_text)
            Tooltip(scale, help_text)

        return scale

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
        slider.bind("<ButtonPress-1>", self.hold_history, add="+")
        slider.bind("<ButtonRelease-1>", self.release_history, add="+")

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

        # Wheel bindings are attached to every child at the end of
        # build_ui so the pointer scrolls the panel from any control.
        self.left_canvas = left_canvas
        self.left_outer = left_outer
        self.reconstruction_widgets = []

        right = ttk.Frame(self.root, padding=12)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(
            left,
            text="PIXEL ART CONVERTER",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w")

        ttk.Label(
            left,
            text="v2.8 • Local Brush and Fast Segmentation",
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

        history_row = ttk.Frame(left)
        history_row.pack(fill="x", pady=3)

        self.undo_button = ttk.Button(
            history_row,
            text="Undo",
            command=self.undo
        )
        self.undo_button.pack(side="left", fill="x", expand=True)

        self.redo_button = ttk.Button(
            history_row,
            text="Redo",
            command=self.redo
        )
        self.redo_button.pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        self.history_label = ttk.Label(
            left,
            text="No changes yet",
            foreground="#666666",
            wraplength=310
        )
        self.history_label.pack(anchor="w", pady=(0, 4))

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
            lambda e: (
                self.update_mode_state(),
                self.schedule_preview()
            )
        )

        self.section(left, "OUTPUT RESOLUTION")

        # Width and Height are now side-by-side.
        resolution = ttk.Frame(left)
        resolution.pack(fill="x", pady=3)

        ttk.Label(resolution, text="Width").grid(
            row=0, column=0, sticky="w"
        )
        width_entry = ttk.Entry(
            resolution,
            textvariable=self.vars["width"],
            width=9
        )
        width_entry.grid(row=0, column=1, padx=(4, 10))
        Tooltip(
            width_entry,
            "Target width in pixels. 0 means the width is derived "
            "from the height. Used by the width and exact fit modes."
        )

        ttk.Label(resolution, text="Height").grid(
            row=0, column=2, sticky="w"
        )
        height_entry = ttk.Entry(
            resolution,
            textvariable=self.vars["height"],
            width=9
        )
        height_entry.grid(row=0, column=3, padx=4)
        Tooltip(
            height_entry,
            "Target height in pixels. 0 means the height is derived "
            "from the width. Used by the height and exact fit modes."
        )

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
            ("Preserve color clusters", "color_clusters"),
        ]:
            check = ttk.Checkbutton(
                left, text=text,
                variable=self.vars[key],
                command=self.schedule_preview
            )
            check.pack(anchor="w", pady=2)
            self.reconstruction_widgets.append(check)

        self.reconstruction_widgets.append(self.add_scale(
            left, "Cluster strength",
            "cluster_strength", 0, 100
        ))
        self.reconstruction_widgets.append(self.add_scale(
            left, "Detail simplification",
            "detail_strength", 0, 100
        ))
        self.reconstruction_widgets.append(self.add_scale(
            left, "Edge optimization",
            "edge_strength", 0, 100
        ))
        self.reconstruction_widgets.append(self.add_scale(
            left, "Color cluster strength",
            "color_strength", 0, 100
        ))

        self.section(left, "COLOR")

        row = ttk.Frame(left)
        row.pack(fill="x", pady=3)

        ttk.Label(
            row, text="Palette (0 = original)"
        ).pack(side="left")

        palette_spin = ttk.Spinbox(
            row,
            from_=0,
            to=256,
            textvariable=self.vars["palette"],
            width=7,
            command=self.schedule_preview
        )
        palette_spin.pack(side="right")
        Tooltip(
            palette_spin,
            "Reduce the image to this many colors. 0 keeps the "
            "original palette."
        )

        ttk.Checkbutton(
            left,
            text="Invert colors",
            variable=self.vars["invert"],
            command=self.schedule_preview
        ).pack(anchor="w", pady=2)

        mono = ttk.Combobox(
            left,
            textvariable=self.vars["monochrome"],
            values=["Off", "Grayscale"],
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

        alpha_check = ttk.Checkbutton(
            left,
            text="Preserve transparency",
            variable=self.vars["alpha"],
            command=self.schedule_preview
        )
        alpha_check.pack(anchor="w", pady=2)
        Tooltip(
            alpha_check,
            "Keeps the original alpha channel in the export. When it "
            "is off, transparent pixels are flattened to black; no "
            "background is detected or removed."
        )

        self.section(left, "OUTLINE")

        border_check = ttk.Checkbutton(
            left,
            text="Add outer border",
            variable=self.vars["border"],
            command=self.schedule_preview
        )
        border_check.pack(anchor="w", pady=2)

        border_hint = ttk.Label(
            left,
            text="Output size stays fixed; borders need transparent margin.",
            foreground="#666666",
            wraplength=310,
        )
        border_hint.pack(anchor="w", pady=(0, 2))

        for widget in (border_check, border_hint):
            Tooltip(
                widget,
                "The border grows outwards from the visible silhouette "
                "but the canvas is never enlarged, so it can only fill "
                "transparent pixels that already exist. If content "
                "touches an edge, the status bar reports a clipped "
                "border; add margin or reduce the target size."
            )

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
            "object_threshold", 1, 60,
            help_text=(
                "RGB distance allowed between a pixel and the seed "
                "color of its region. Because the comparison uses the "
                "seed color, a gradual gradient splits into several "
                "regions instead of one."
            )
        )

        self.add_scale(
            left, "Minimum object area",
            "object_min_area", 1, 100,
            help_text=(
                "Connected regions smaller than this pixel count are "
                "discarded and never receive an internal outline."
            )
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

        self.section(left, "NOISE CLEANUP")

        noise_check = ttk.Checkbutton(
            left,
            text="Clean small color noise",
            variable=self.vars["noise_cleanup"],
            command=self.schedule_preview
        )
        noise_check.pack(anchor="w", pady=2)
        Tooltip(
            noise_check,
            "Replaces small stray color fragments with the closest "
            "adjacent region color. Runs at the final resolution, "
            "after palette reduction and before any outline."
        )

        self.add_scale(
            left, "Cleanup strength",
            "noise_strength", 0, 100,
            help_text=(
                "Higher values remove larger fragments. Low values only "
                "reach single-pixel speckles."
            )
        )

        self.advanced_noise_widgets = []

        self.noise_advanced = tk.BooleanVar(value=False)
        advanced_frame = ttk.Frame(left)

        ttk.Checkbutton(
            left,
            text="Advanced cleanup options",
            variable=self.noise_advanced,
            command=lambda: (
                advanced_frame.pack(fill="x")
                if self.noise_advanced.get()
                else advanced_frame.pack_forget()
            )
        ).pack(anchor="w", pady=2)

        self.add_scale(
            advanced_frame, "Cleanup color tolerance",
            "noise_tolerance", 0, 60,
            help_text=(
                "How different an adjacent color may be before it can "
                "absorb a fragment. Fragments are always exact-color "
                "groups, so raising this never widens the grouping."
            )
        )

        self.add_scale(
            advanced_frame, "Fragment grouping tolerance",
            "noise_group_tolerance", 0, 30,
            help_text=(
                "How similar two neighboring pixels must be to count as "
                "the same fragment. At 0 only identical colors group, "
                "which means nothing is cleaned unless the palette has "
                "been reduced first. Keep it below the merge tolerance."
            )
        )

        self.add_scale(
            advanced_frame, "Maximum region area",
            "noise_max_area", 1, 64,
            help_text=(
                "Hard ceiling in pixels. Cleanup strength scales up to "
                "this value and never past it."
            )
        )

        protect_check = ttk.Checkbutton(
            advanced_frame,
            text="Protect silhouette",
            variable=self.vars["noise_protect"],
            command=self.schedule_preview
        )
        protect_check.pack(anchor="w", pady=2)
        Tooltip(
            protect_check,
            "Leaves fragments that touch transparency or the canvas "
            "edge untouched, so the outline of the sprite is preserved."
        )

        self.section(left, "SELECTION")

        selection_combo = ttk.Combobox(
            left,
            textvariable=self.vars["selection_mode"],
            values=[
                "Rectangle",
                "Connected region",
                "Brush (add)",
                "Brush (subtract)"
            ],
            state="readonly"
        )
        selection_combo.pack(fill="x", pady=2)
        selection_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self.clear_selection()
        )
        Tooltip(
            selection_combo,
            "Drag on the preview to select a rectangle, click to "
            "select a connected color region using the internal "
            "outline tolerance, or paint with the brush to add and "
            "subtract pixels. Selections mark pixels only; they never "
            "modify the image."
        )

        self.add_scale(
            left, "Brush size",
            "brush_size", 1, 32,
            help_text=(
                "Diameter of the brush dab, in output pixels. Each "
                "completed stroke is one undoable action."
            )
        )

        local_check = ttk.Checkbutton(
            left,
            text="Apply color effects to selection only",
            variable=self.vars["local_effects"],
            command=self.schedule_preview
        )
        local_check.pack(anchor="w", pady=2)
        Tooltip(
            local_check,
            "Restricts invert, grayscale, RGB tints, and noise cleanup "
            "to the selected pixels. Contrast runs before the resize, "
            "so it stays global."
        )

        self.selection_label = ttk.Label(
            left,
            text="Selection: none",
            foreground="#666666",
            wraplength=310
        )
        self.selection_label.pack(anchor="w", pady=(2, 0))

        ttk.Button(
            left,
            text="Clear selection",
            command=self.clear_selection
        ).pack(fill="x", pady=3)

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
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(sequence, self.preview_wheel)

        self.canvas.bind("<ButtonPress-1>", self.selection_press)
        self.canvas.bind("<B1-Motion>", self.selection_drag)
        self.canvas.bind("<ButtonRelease-1>", self.selection_release)

        # Typed numeric fields refresh the preview on their own, with a
        # longer debounce so intermediate keystrokes are not rendered.
        for key in self.NUMERIC_LIMITS:
            self.vars[key].trace_add(
                "write",
                lambda *_: self.schedule_preview(delay=350)
            )

        self.bind_wheel(self.left_outer, self.config_wheel)
        self.update_mode_state()

        self.root.bind_all(
            "<Control-z>", lambda e: self.undo()
        )
        self.root.bind_all(
            "<Control-y>", lambda e: self.redo()
        )
        self.root.bind_all(
            "<Control-Shift-Z>", lambda e: self.redo()
        )

        self.update_history_buttons()

    # ---------------------------------------------------------
    # Non-destructive selection
    # ---------------------------------------------------------

    PREVIEW_ORIGIN = 10

    def canvas_to_image(self, event):
        """
        Map a canvas click to final-image coordinates.

        Zoom and scroll offsets are removed, so the result addresses the
        processed output rather than what happens to be on screen.
        """
        if self.preview_source is None:
            return None

        scale = float(self.zoom.get()) / 100.0

        if scale <= 0:
            return None

        x = (
            self.canvas.canvasx(event.x) - self.PREVIEW_ORIGIN
        ) / scale
        y = (
            self.canvas.canvasy(event.y) - self.PREVIEW_ORIGIN
        ) / scale

        x, y = int(np.floor(x)), int(np.floor(y))

        if not (
            0 <= x < self.preview_source.width
            and 0 <= y < self.preview_source.height
        ):
            return None

        return x, y

    @staticmethod
    def rectangle_mask(shape, start, end):
        """Binary mask for an inclusive rectangle in image coordinates."""
        height, width = shape
        x0, y0 = start
        x1, y1 = end

        left, right = sorted((int(x0), int(x1)))
        top, bottom = sorted((int(y0), int(y1)))

        left = max(0, left)
        top = max(0, top)
        right = min(width - 1, right)
        bottom = min(height - 1, bottom)

        mask = np.zeros((height, width), dtype=bool)

        if left <= right and top <= bottom:
            mask[top:bottom + 1, left:right + 1] = True

        return mask

    @staticmethod
    def region_mask(rgb, alpha, point, tolerance):
        """
        Binary mask for the connected color region under a point.

        This reuses the same segmentation the internal outlines use, so
        clicking selects exactly the region that would be outlined.
        """
        labels = Pipeline.segment_color_regions(
            rgb, alpha, tolerance, 1
        )

        x, y = point
        label = labels[y, x]

        if label < 0:
            return np.zeros(labels.shape, dtype=bool)

        return labels == label

    @staticmethod
    def brush_mask(shape, center, diameter):
        """Circular stroke dab, in image coordinates."""
        height, width = shape
        x, y = center
        radius = max(1, int(round(diameter))) / 2.0

        rows = np.arange(height)[:, None] - y
        columns = np.arange(width)[None, :] - x

        return (columns ** 2 + rows ** 2) <= radius ** 2

    @staticmethod
    def stroke_points(start, end):
        """
        Integer points along a segment.

        Pointer motion is sampled, so a fast drag would otherwise leave
        gaps between dabs.
        """
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0))

        if steps == 0:
            return [(x0, y0)]

        return [
            (
                round(x0 + (x1 - x0) * step / steps),
                round(y0 + (y1 - y0) * step / steps)
            )
            for step in range(steps + 1)
        ]

    def paint_stroke(self, start, end, additive):
        """Add or subtract one segment of a brush stroke."""
        shape = (
            self.preview_source.height,
            self.preview_source.width
        )

        if self.selection is None or self.selection.shape != shape:
            self.selection = np.zeros(shape, dtype=bool)

        diameter = self.vars["brush_size"].get()

        for point in self.stroke_points(start, end):
            dab = self.brush_mask(shape, point, diameter)

            if additive:
                self.selection |= dab
            else:
                self.selection &= ~dab

        self.selection_shape = self.selection_signature()

        count = int(self.selection.sum())

        if not count:
            self.clear_selection()
            return

        self.selection_label.config(
            text=f"Selection: brush ({count} px)"
        )
        self.draw_selection()

    def selection_signature(self):
        """Output geometry a selection belongs to."""
        if self.preview_source is None:
            return None

        return (self.preview_source.width, self.preview_source.height)

    def set_selection(self, mask, description):
        self.selection = mask
        self.selection_shape = self.selection_signature()

        count = int(mask.sum()) if mask is not None else 0

        if not count:
            self.clear_selection()
            return

        self.selection_label.config(
            text=f"Selection: {description} ({count} px)"
        )
        self.draw_selection()

    def clear_selection(self, *_):
        had_selection = self.selection is not None

        self.selection = None
        self.selection_shape = None
        self.selection_label.config(text="Selection: none")
        self.canvas.delete("selection")

        if had_selection:
            self.record_history("Clear selection")

    def validate_selection(self):
        """
        Drop a selection whose coordinates no longer address the output.

        Remapping across a crop or resize would be ambiguous, so the
        selection is cleared and reported instead of silently moving.
        """
        if self.selection is None:
            return

        if self.selection_shape != self.selection_signature():
            self.clear_selection()
            self.selection_label.config(
                text="Selection cleared: output size changed"
            )

    def draw_selection(self):
        """Overlay the selection outline without altering any pixel."""
        self.canvas.delete("selection")

        if self.selection is None or self.preview_source is None:
            return

        scale = float(self.zoom.get()) / 100.0
        rows = np.flatnonzero(self.selection.any(axis=1))
        cols = np.flatnonzero(self.selection.any(axis=0))

        if not rows.size or not cols.size:
            return

        self.canvas.create_rectangle(
            self.PREVIEW_ORIGIN + cols[0] * scale,
            self.PREVIEW_ORIGIN + rows[0] * scale,
            self.PREVIEW_ORIGIN + (cols[-1] + 1) * scale,
            self.PREVIEW_ORIGIN + (rows[-1] + 1) * scale,
            outline="#00d0ff",
            dash=(4, 3),
            width=2,
            tags="selection"
        )

    def selection_press(self, event):
        point = self.canvas_to_image(event)

        if point is None:
            return

        mode = self.vars["selection_mode"].get()

        if mode.startswith("Brush"):
            # One stroke is one history action, so recording is held
            # until the button is released.
            self.hold_history()
            self.paint_stroke(point, point, mode == "Brush (add)")
            self.selection_anchor = point
            return

        if self.vars["selection_mode"].get() == "Connected region":
            rgba = np.asarray(self.preview_source.convert("RGBA"))

            self.set_selection(
                self.region_mask(
                    np.array(rgba[:, :, :3], copy=True),
                    np.array(rgba[:, :, 3], copy=True),
                    point,
                    int(round(self.vars["object_threshold"].get()))
                ),
                "connected region"
            )
            self.selection_anchor = None
            return

        self.selection_anchor = point

    def selection_drag(self, event):
        if self.selection_anchor is None:
            return

        point = self.canvas_to_image(event)

        if point is None:
            return

        mode = self.vars["selection_mode"].get()

        if mode.startswith("Brush"):
            self.paint_stroke(
                self.selection_anchor,
                point,
                mode == "Brush (add)"
            )
            self.selection_anchor = point
            return

        self.set_selection(
            self.rectangle_mask(
                (
                    self.preview_source.height,
                    self.preview_source.width
                ),
                self.selection_anchor,
                point
            ),
            "rectangle"
        )

    def selection_release(self, _event=None):
        was_stroke = (
            self.selection_anchor is not None
            and self.vars["selection_mode"].get().startswith("Brush")
        )

        self.selection_anchor = None

        if was_stroke:
            self.history_hold = False
            self.record_history("Brush stroke")

    # ---------------------------------------------------------
    # Edit history
    # ---------------------------------------------------------

    HISTORY_LIMIT = 40

    # A render slower than the debounce window gets a busy status.
    PREVIEW_BUSY_MS = 250

    def snapshot(self):
        """Compact record of every setting; never a rendered image."""
        return {
            key: var.get()
            for key, var in self.vars.items()
        }

    def packed_selection(self):
        """
        The active mask as 1-bit data, or None.

        Packing keeps a 512 px mask at 32 KB, so a bounded history of
        strokes stays small next to the image it describes.
        """
        if self.selection is None:
            return None

        return (
            self.selection.shape,
            np.packbits(self.selection).tobytes()
        )

    @staticmethod
    def unpack_selection(packed):
        if packed is None:
            return None

        shape, data = packed
        bits = np.unpackbits(
            np.frombuffer(data, dtype=np.uint8)
        )

        return bits[:shape[0] * shape[1]].astype(bool).reshape(shape)

    def current_state(self):
        """Settings plus selection metadata; never a rendered image."""
        return (self.snapshot(), self.packed_selection())

    @staticmethod
    def describe_change(before, after):
        """Human-readable label for the difference between two snapshots."""
        changed = [
            key
            for key in after
            if before.get(key) != after[key]
        ]

        if not changed:
            return None

        if len(changed) > 3:
            return f"{len(changed)} settings"

        parts = []

        for key in changed:
            label = key.replace("_", " ").capitalize()
            value = after[key]

            if isinstance(value, bool):
                parts.append(f"{label} {'on' if value else 'off'}")
            elif isinstance(value, float):
                parts.append(f"{label} {value:.2f}")
            else:
                parts.append(f"{label} {value}")

        return ", ".join(parts)

    def hold_history(self, _event=None):
        """Suspend recording while a slider is being dragged."""
        self.history_hold = True

    def release_history(self, _event=None):
        """Close a slider drag as a single history entry."""
        self.history_hold = False
        self.record_history()

    def record_history(self, label=None):
        """
        Store one history entry when the settings actually changed.

        Entries hold setting values only, so the stack stays small no
        matter how large the rendered image is. Nothing is recorded
        mid-drag or while a typed value is still invalid.
        """
        if self.restoring_history or self.history_hold:
            return

        if self.numeric_error() is not None:
            return

        current = self.current_state()
        description = self.describe_change(
            self.history_state[0], current[0]
        )

        if description is None:
            if self.history_state[1] == current[1]:
                return

            description = "Selection"

        self.undo_stack.append((self.history_state, label or description))
        del self.undo_stack[:-self.HISTORY_LIMIT]

        self.redo_stack.clear()
        self.history_state = current
        self.update_history_buttons()

    def apply_snapshot(self, state):
        """Restore settings and selection without recording an entry."""
        settings, packed = state

        self.restoring_history = True

        try:
            for key, value in settings.items():
                if self.vars[key].get() != value:
                    self.vars[key].set(value)
        finally:
            self.restoring_history = False

        self.selection = self.unpack_selection(packed)
        self.selection_shape = (
            self.selection_signature()
            if self.selection is not None
            else None
        )
        self.selection_label.config(
            text=(
                "Selection: none"
                if self.selection is None
                else f"Selection: restored "
                     f"({int(self.selection.sum())} px)"
            )
        )

        self.history_state = state

        self.border_color_preview.config(
            bg=self.vars["border_color"].get()
        )
        self.object_color_preview.config(
            bg=self.vars["object_border_color"].get()
        )

        self.update_mode_state()
        self.update_history_buttons()
        self.schedule_preview()

    def undo(self):
        if not self.undo_stack or self.numeric_error() is not None:
            return

        state, label = self.undo_stack.pop()
        self.redo_stack.append((self.current_state(), label))
        self.apply_snapshot(state)

    def redo(self):
        if not self.redo_stack or self.numeric_error() is not None:
            return

        state, label = self.redo_stack.pop()
        self.undo_stack.append((self.current_state(), label))
        self.apply_snapshot(state)

    def update_history_buttons(self):
        self.undo_button.config(
            state="normal" if self.undo_stack else "disabled"
        )
        self.redo_button.config(
            state="normal" if self.redo_stack else "disabled"
        )

        if self.undo_stack:
            text = f"Last change: {self.undo_stack[-1][1]}"
        else:
            text = "No changes yet"

        self.history_label.config(text=text)

    # ---------------------------------------------------------
    # Wheel / panel state
    # ---------------------------------------------------------

    @staticmethod
    def wheel_steps(event):
        """
        Normalize wheel input across platforms.

        Returns the number of scroll units; negative means up or left.
        Windows and macOS report ``delta``, X11 reports button 4/5.
        """
        num = getattr(event, "num", 0)

        if num == 4:
            return -1
        if num == 5:
            return 1

        delta = int(getattr(event, "delta", 0) or 0)

        if delta == 0:
            return 0

        # Windows reports multiples of 120; macOS reports small values.
        if abs(delta) >= 120:
            return -delta // 120

        return -1 if delta > 0 else 1

    def preview_wheel(self, event):
        """Ctrl + wheel zooms, Shift + wheel pans, plain wheel scrolls."""
        steps = self.wheel_steps(event)

        if steps == 0 or self.preview_source is None:
            return "break"

        if event.state & 0x0004:
            current = float(self.zoom.get())
            self.zoom.set(
                max(
                    25,
                    min(800, current * (1.15 ** -steps))
                )
            )
            self.render_preview_image()

        elif event.state & 0x0001:
            self.canvas.xview_scroll(steps, "units")

        else:
            self.canvas.yview_scroll(steps, "units")

        return "break"

    def config_wheel(self, event):
        steps = self.wheel_steps(event)

        if steps:
            self.left_canvas.yview_scroll(steps, "units")

        return "break"

    def bind_wheel(self, widget, handler):
        """Bind wheel events on a widget and all of its descendants."""
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, handler, add="+")

        for child in widget.winfo_children():
            self.bind_wheel(child, handler)

    def update_mode_state(self):
        """
        Disable reconstruction-only controls in Pixel Perfect Resize.

        Values are preserved so switching back restores the settings.
        """
        state = (
            "normal"
            if self.vars["mode"].get() == "Pixel Art Reconstruction"
            else "disabled"
        )

        for widget in self.reconstruction_widgets:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    # ---------------------------------------------------------
    # Numeric validation
    # ---------------------------------------------------------

    @staticmethod
    def validate_numeric(text, low, high, label):
        """Return an error message for a typed field, or None if valid."""
        text = str(text).strip()

        if not text:
            return f"{label} is empty."

        try:
            value = int(text)
        except ValueError:
            return f"{label} must be a whole number."

        if not low <= value <= high:
            return f"{label} must be between {low} and {high}."

        return None

    def numeric_error(self):
        """First invalid typed numeric field, or None when all are valid."""
        for key, (low, high, label) in self.NUMERIC_LIMITS.items():
            # Reading the raw Tcl text avoids the TclError an IntVar
            # raises while a value is still being typed.
            text = self.root.getvar(str(self.vars[key]))
            error = self.validate_numeric(text, low, high, label)

            if error is not None:
                return error

        return None

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
            self.record_history(f"Border color {color}")
            self.schedule_preview()

    def choose_object_color(self):
        color = colorchooser.askcolor(
            initialcolor=self.vars["object_border_color"].get(),
            title="Choose object border color"
        )[1]

        if color:
            self.vars["object_border_color"].set(color)
            self.object_color_preview.config(bg=color)
            self.record_history(f"Object border color {color}")
            self.schedule_preview()

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):
        self.restoring_history = True

        try:
            for key, value in self.DEFAULTS.items():
                self.vars[key].set(value)
        finally:
            self.restoring_history = False

        self.zoom.set(100)

        self.border_color_preview.config(
            bg=self.vars["border_color"].get()
        )
        self.object_color_preview.config(
            bg=self.vars["object_border_color"].get()
        )

        self.record_history("Reset to defaults")
        self.update_mode_state()
        self.schedule_preview()

    # ---------------------------------------------------------
    # Zoom
    # ---------------------------------------------------------

    def reset_preview_scroll(self):
        """Return the preview to its top-left corner after a zoom change."""
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def set_zoom(self, value):
        self.zoom.set(float(value))
        self.render_preview_image()
        self.reset_preview_scroll()

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
        self.reset_preview_scroll()

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

            self.draw_selection()

        except Exception as e:
            self.status.config(
                text=f"Preview display error: {e}"
            )

    # ---------------------------------------------------------
    # Open / crop / resize
    # ---------------------------------------------------------

    def clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.history_state = self.current_state()
        self.update_history_buttons()

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

            # A new source image starts a new document, so earlier
            # settings history no longer applies to what is on screen.
            self.clear_history()
            self.schedule_preview()

        except Exception as e:
            messagebox.showerror(
                "Open error",
                str(e)
            )


    # ---------------------------------------------------------
    # Processing
    # ---------------------------------------------------------

    def build_settings(self):
        """Normalized snapshot of every processing setting."""
        return Settings.from_mapping(self.snapshot())

    def process(self):
        """Run the pipeline on the loaded image and collect its warnings."""
        pipeline = Pipeline(
            self.build_settings(),
            mask=self.selection
        )
        result = pipeline.run(self.src)

        self.outer_border_margin_warning = pipeline.outer_border_clipped
        self.noise_pixels_changed = pipeline.noise_pixels_changed

        return result

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

        error = self.numeric_error()

        if error is not None:
            self.status.config(text=error)
            return

        self.record_history()

        # A render that overran the debounce window last time is very
        # likely to overrun it again, so say so before starting.
        slow = self.last_render_seconds * 1000 > self.PREVIEW_BUSY_MS

        if slow:
            self.status.config(text="Processing preview...")
            self.status.update_idletasks()

        try:
            started = time.perf_counter()
            img = self.process()
            self.last_render_seconds = time.perf_counter() - started

            self.preview_source = img
            self.validate_selection()

            status = (
                f"Preview: {img.width}×{img.height} px | "
                f"{self.vars['mode'].get()} | "
                f"Zoom {self.zoom.get():.0f}%"
            )

            if self.vars["noise_cleanup"].get():
                # Without this the controls give no feedback at all when
                # they happen to clean nothing.
                status += f" | cleanup {self.noise_pixels_changed} px"

            if self.last_render_seconds * 1000 > self.PREVIEW_BUSY_MS:
                status += f" | {self.last_render_seconds:.1f} s"

            if getattr(self, "outer_border_margin_warning", False):
                status += (
                    " | Outer border clipped: content touches the canvas edge"
                )

            self.status.config(text=status)

            self.render_preview_image()

        except Exception as e:
            self.status.config(
                text=f"Preview error: {e}"
            )

    def schedule_preview(self, delay=100):
        if self.render_job is not None:
            try:
                self.root.after_cancel(
                    self.render_job
                )
            except Exception:
                pass

        self.render_job = self.root.after(
            delay,
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
