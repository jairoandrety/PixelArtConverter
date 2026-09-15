"""
Image pipeline for Pixel Art Converter.

This module is independent of Tkinter: it takes a source image plus a
normalized :class:`Settings` record and returns a processed PIL image.
Keeping it separate makes the pipeline unit-testable and reusable for
batch conversion or a command-line front end, without touching the
desktop experience.

    source image + normalized settings
      -> crop / resize / reconstruction / color effects / cleanup / outlines
      -> PIL output image
"""

from bisect import bisect_left
from dataclasses import dataclass, fields

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


@dataclass
class Settings:
    """Normalized processing settings, free of any Tk variable."""

    width: int = 0
    height: int = 256
    fit_mode: str = "height"
    mode: str = "Pixel Art Reconstruction"
    crop: bool = True
    crop_threshold: int = 8
    cleanup: bool = False
    contrast: float = 1.06
    progressive: bool = True
    pixel_final: bool = True
    palette: int = 0
    dither: bool = False
    alpha: bool = False
    clusters: bool = True
    detail: bool = True
    edges: bool = True
    color_clusters: bool = True
    cluster_strength: int = 35
    detail_strength: int = 25
    edge_strength: int = 40
    color_strength: int = 30
    invert: bool = False
    monochrome: str = "Off"
    tint_r: bool = False
    tint_g: bool = False
    tint_b: bool = False
    tint_r_amount: int = 0
    tint_g_amount: int = 0
    tint_b_amount: int = 0
    border: bool = False
    border_width: int = 1
    border_color: str = "#000000"
    object_borders: bool = False
    object_border_width: int = 1
    object_threshold: int = 18
    object_min_area: int = 3
    object_border_color: str = "#000000"
    noise_cleanup: bool = False
    noise_method: str = "Local dominant color"
    noise_window: int = 3
    noise_min_isolation: int = 1
    noise_strength: int = 30
    noise_tolerance: int = 20
    noise_group_tolerance: int = 2
    noise_max_area: int = 12
    noise_protect: bool = True
    local_effects: bool = False
    average_blend: int = 0
    average_source: str = "Selection"

    @classmethod
    def from_mapping(cls, data):
        """
        Build settings from a plain mapping, ignoring unrelated keys.

        The GUI stores presentation-only values (such as the selection
        mode) alongside processing ones; those are not pipeline inputs.
        """
        known = {field.name for field in fields(cls)}

        return cls(**{
            key: value
            for key, value in data.items()
            if key in known
        })


class Pipeline:
    """Stateless-per-run image pipeline."""

    def __init__(self, settings, mask=None, class_colors=None,
                 class_overrides=None):
        """
        ``mask`` is an optional boolean array in output coordinates.

        When it is present and ``settings.local_effects`` is set, the
        final-resolution stages are restricted to the marked pixels.
        """
        self.settings = settings
        self.mask = mask
        # Detected class centers plus the replacement colors chosen for
        # some of them. Pixels are matched to a center at render time,
        # so a recolor survives changes to the other settings instead of
        # being a one-off edit to a rendered image.
        self.class_colors = (
            None if class_colors is None
            else np.asarray(class_colors, dtype=np.float64)
        )
        self.class_overrides = dict(class_overrides or {})
        self.outer_border_clipped = False
        self.noise_pixels_changed = 0
        self.recolored_pixels = 0
        self.average_blended_pixels = 0

    def restrict(self, before, after):
        """
        Keep a stage's result only where the active mask is set.

        A mask that does not match the image is ignored rather than
        applied to the wrong pixels; the GUI clears stale selections,
        and this is the pipeline-side guard for the same rule.
        """
        if not self.settings.local_effects or self.mask is None:
            return after

        if before is after:
            return after

        if self.mask.shape != (before.height, before.width):
            return after

        original = np.asarray(before.convert("RGBA"))
        updated = np.asarray(after.convert("RGBA"))

        if original.shape != updated.shape:
            return after

        return Image.fromarray(
            np.where(
                self.mask[:, :, None], updated, original
            ).astype(np.uint8),
            "RGBA"
        )

    def crop_empty(self, img):
        rgba = np.array(img, copy=True)

        rgb = rgba[:, :, :3].astype(
            np.int16
        )

        threshold = int(
            round(
                self.settings.crop_threshold
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
        mode = self.settings.fit_mode

        tw = int(self.settings.width)
        th = int(self.settings.height)

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

    # Intermediate sizes for the progressive ladder, largest first.
    PROGRESSIVE_STEPS = (768, 512, 384, 320)

    @staticmethod
    def dominant_resize_axis(size, target):
        """
        Axis that has to shrink the most on the way to the target.

        The progressive ladder follows this axis so that a very wide or
        very tall source receives the same intermediate treatment as a
        portrait one. For a proportional resize both ratios match and
        the height is used, which is the historical behavior.
        """
        width, height = size
        target_width, target_height = target

        if width / max(1, target_width) > height / max(1, target_height):
            return "width"

        return "height"

    def progressive_resize(self, img, target):
        """
        Step down through intermediate sizes before the final resize.

        Going straight from a large illustration to a small sprite loses
        structure, so the image is reduced in LANCZOS stages first.
        """
        if not self.settings.progressive:
            return img

        axis = self.dominant_resize_axis(
            (img.width, img.height), target
        )
        limit = target[0] if axis == "width" else target[1]

        current = img

        for step in self.PROGRESSIVE_STEPS:
            size = current.width if axis == "width" else current.height

            if size <= step or step <= limit:
                continue

            scale = step / size

            if axis == "width":
                size = (
                    step,
                    max(1, round(current.height * scale))
                )
            else:
                size = (
                    max(1, round(current.width * scale)),
                    step
                )

            current = current.resize(
                size,
                Image.Resampling.LANCZOS
            )

        return current

    # ---------------------------------------------------------
    # Pixel-art reconstruction
    # ---------------------------------------------------------

    def reconstruct_pixel_art(self, img):
        rgba = np.array(
            img.convert("RGBA"),
            dtype=np.float32,
            copy=True
        )

        rgb = np.array(rgba[:, :, :3], copy=True)
        alpha = np.array(rgba[:, :, 3], copy=True)

        h, w = rgb.shape[:2]

        if h < 3 or w < 3:
            return img

        # Color cluster preservation.
        cs = (
            float(
                self.settings.color_strength
            ) / 100.0
        )

        if (
            self.settings.color_clusters
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
                self.settings.cluster_strength
            ) / 100.0
        )

        if (
            self.settings.clusters
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
                self.settings.detail_strength
            ) / 100.0
        )

        if (
            self.settings.detail
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
                self.settings.edge_strength
            ) / 100.0
        )

        if (
            self.settings.edges
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
        rgba = np.array(
            img.convert("RGBA"),
            dtype=np.float32,
            copy=True
        )

        rgb = np.array(rgba[:, :, :3], copy=True)
        alpha = np.array(rgba[:, :, 3], copy=True)

        if self.settings.invert:
            rgb = 255.0 - rgb

        mono = self.settings.monochrome

        if mono != "Off":
            luminance = (
                0.299 * rgb[:, :, 0]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            )

            # "Black" and "White" are retained for compatibility with
            # pre-v2.5 sessions; the UI exposes the accurate Grayscale name.
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
            if getattr(self.settings, enabled_key):
                amount = float(
                    getattr(self.settings, amount_key)
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
    # Element detection
    # ---------------------------------------------------------

    # Pixels compared against centers in one go; chunking keeps the
    # distance matrix bounded on large outputs.
    ASSIGN_CHUNK = 65536

    @staticmethod
    def assign_classes(pixels, centers):
        """Nearest center for every pixel, in bounded chunks."""
        out = np.empty(len(pixels), dtype=np.int32)

        for start in range(0, len(pixels), Pipeline.ASSIGN_CHUNK):
            block = pixels[start:start + Pipeline.ASSIGN_CHUNK]
            distances = (
                (block[:, None, :] - centers[None, :, :]) ** 2
            ).sum(axis=2)
            out[start:start + len(block)] = distances.argmin(axis=1)

        return out

    @staticmethod
    def cluster_colors(rgb, alpha, classes, sample_size=6000,
                       iterations=20, seed=0):
        """
        Group visible pixels into color classes with k-means.

        A class is a material rather than an object: lit brick, shaded
        brick, glass, foliage, balcony slab. Clustering runs on RGB
        because dropping luminance was measured to be worse here --
        value is most of what separates glass from render from stone in
        pixel art.

        The seed is fixed so a preview does not change between renders,
        and the fit runs on a subsample so cost does not grow with the
        image.
        """
        height, width = rgb.shape[:2]
        labels = np.full((height, width), -1, dtype=np.int32)
        visible = alpha > 5

        if not visible.any() or classes < 1:
            return labels, np.zeros((0, 3), dtype=np.uint8)

        pixels = rgb[visible].astype(np.float64)
        distinct = np.unique(pixels, axis=0)
        classes = max(1, min(int(classes), len(distinct)))

        rng = np.random.default_rng(seed)

        if len(pixels) > sample_size:
            sample = pixels[
                rng.choice(len(pixels), sample_size, replace=False)
            ]
        else:
            sample = pixels

        # k-means++ seeding, so the result does not depend on luck.
        centers = np.empty((classes, 3), dtype=np.float64)
        centers[0] = sample[rng.integers(len(sample))]
        closest = ((sample - centers[0]) ** 2).sum(axis=1)

        for index in range(1, classes):
            total = float(closest.sum())

            if total <= 0:
                centers[index] = sample[rng.integers(len(sample))]
            else:
                centers[index] = sample[
                    rng.choice(len(sample), p=closest / total)
                ]

            closest = np.minimum(
                closest, ((sample - centers[index]) ** 2).sum(axis=1)
            )

        for _ in range(iterations):
            assignment = Pipeline.assign_classes(sample, centers)
            moved = False

            for index in range(classes):
                members = sample[assignment == index]

                if not len(members):
                    continue

                updated = members.mean(axis=0)

                if not np.allclose(updated, centers[index]):
                    moved = True

                centers[index] = updated

            if not moved:
                break

        labels[visible] = Pipeline.assign_classes(pixels, centers)

        return labels, np.clip(np.rint(centers), 0, 255).astype(np.uint8)

    @staticmethod
    def class_instances(mask, min_area=1):
        """
        Connected components of one class mask.

        A class is the material; an instance is one object of it, such
        as a single window within the glass class. This reuses the same
        run-based segmentation the outlines use by handing it a uniform
        color, so only connectivity decides the result.
        """
        height, width = mask.shape
        uniform = np.zeros((height, width, 3), dtype=np.uint8)
        alpha = np.where(mask, 255, 0).astype(np.uint8)

        return Pipeline.segment_color_regions(uniform, alpha, 0, min_area)

    @staticmethod
    def describe_classes(labels, centers, min_area=1):
        """
        One entry per detected class, largest first.

        Each entry carries the class id, its representative color, its
        pixel count, and how many separate objects it forms.
        """
        described = []
        visible = labels >= 0
        total = int(visible.sum())

        if not total:
            return described

        counts = np.bincount(labels[visible], minlength=len(centers))

        for index, count in enumerate(counts):
            if not count:
                continue

            instances = Pipeline.class_instances(
                labels == index, min_area
            )
            described.append({
                "id": int(index),
                "color": tuple(int(v) for v in centers[index]),
                "pixels": int(count),
                "share": float(count) / total,
                "instances": int(instances.max()) + 1,
            })

        described.sort(key=lambda entry: -entry["pixels"])

        return described

    def detect_elements(self, img, classes=10, min_area=1):
        """Cluster an output image into classes and count their objects."""
        data = np.asarray(img.convert("RGBA"))
        labels, centers = self.cluster_colors(
            np.array(data[:, :, :3], copy=True),
            np.array(data[:, :, 3], copy=True),
            classes
        )

        return labels, centers, self.describe_classes(
            labels, centers, min_area
        )

    # ---------------------------------------------------------
    # Region-aware noise cleanup
    # ---------------------------------------------------------

    @staticmethod
    def cleanup_max_area(strength, ceiling):
        """
        Map a 0-100 strength to a maximum removable region area.

        The mapping is deliberately conservative: low strengths only
        reach one-pixel speckles, and the ceiling is never exceeded.
        """
        ceiling = max(1, int(round(ceiling)))
        strength = max(0, min(100, float(strength)))

        return max(1, int(round(strength / 100.0 * ceiling)))

    @staticmethod
    def clean_region_noise(rgb, alpha, tolerance, max_area,
                           protect_silhouette=True, group_tolerance=0):
        """
        Replace small stray color fragments with an adjacent region color.

        Groups no larger than ``max_area`` are candidates, and each one
        is merged into the closest strictly larger adjacent group, and
        only when that color is within ``tolerance``. Requiring a larger
        neighbor is what stops two speckles from merging into each
        other. Alpha is never modified.

        ``tolerance`` controls how different a neighbor may be before it
        can absorb a fragment. ``group_tolerance`` controls what counts
        as one fragment in the first place, and must be smaller than
        ``tolerance`` for anything to be merged at all: at 0 only
        identical pixels group together, so on continuous-tone artwork
        every pixel becomes its own fragment, no neighbor is ever large
        enough to absorb one, and nothing is cleaned.
        """
        rgb = np.array(rgb, copy=True)
        valid = alpha > 5

        if not valid.any():
            return rgb

        labels = Pipeline.segment_color_regions(
            rgb, alpha, group_tolerance, 1
        )

        count = int(labels.max()) + 1

        if count < 2:
            return rgb

        flat = labels.ravel()
        inside = flat >= 0
        indices = flat[inside]

        sizes = np.bincount(indices, minlength=count)

        # Mean color per region; pixel-art regions are near-uniform, so
        # the mean is a faithful representative color.
        means = np.stack(
            [
                np.bincount(
                    indices,
                    weights=rgb[:, :, channel].ravel()[inside].astype(
                        np.float64
                    ),
                    minlength=count
                )
                for channel in range(3)
            ],
            axis=1
        ) / np.maximum(sizes, 1)[:, None]

        candidates = np.flatnonzero(
            (sizes > 0) & (sizes <= max_area)
        )

        if candidates.size == 0:
            return rgb

        if protect_silhouette:
            # A fragment touching transparency (or the canvas edge) is
            # part of the outline and must not be repainted.
            outside = np.pad(
                ~valid,
                1,
                mode="constant",
                constant_values=True
            )
            touches = (
                outside[:-2, 1:-1]
                | outside[2:, 1:-1]
                | outside[1:-1, :-2]
                | outside[1:-1, 2:]
            )
            exposed = np.bincount(
                indices,
                weights=touches.ravel()[inside].astype(np.float64),
                minlength=count
            ) > 0
            candidates = candidates[~exposed[candidates]]

            if candidates.size == 0:
                return rgb

        # Undirected 4-neighbor adjacency between different regions.
        first = np.concatenate(
            (labels[:, :-1].ravel(), labels[:-1, :].ravel())
        )
        second = np.concatenate(
            (labels[:, 1:].ravel(), labels[1:, :].ravel())
        )

        linked = (first >= 0) & (second >= 0) & (first != second)
        first, second = first[linked], second[linked]

        sources = np.concatenate((first, second))
        neighbors = np.concatenate((second, first))

        order = np.argsort(sources, kind="stable")
        sources = sources[order]
        neighbors = neighbors[order]

        starts = np.searchsorted(sources, candidates, side="left")
        ends = np.searchsorted(sources, candidates, side="right")

        target = np.full(count, -1, dtype=np.int64)
        limit_sq = float(tolerance) ** 2

        for region, start, end in zip(candidates, starts, ends):
            if start == end:
                continue

            adjacent = np.unique(neighbors[start:end])
            # A fragment may only be absorbed by a strictly larger
            # neighbor, so two speckles never merge into each other.
            # Comparing against the fragment rather than against
            # max_area keeps the strength control monotonic: raising it
            # can only ever clean more, never less.
            adjacent = adjacent[sizes[adjacent] > sizes[region]]

            if adjacent.size == 0:
                continue

            deltas = means[adjacent] - means[region]
            distances = np.einsum("ij,ij->i", deltas, deltas)
            best = int(np.argmin(distances))

            if distances[best] <= limit_sq:
                target[region] = adjacent[best]

        if not (target >= 0).any():
            return rgb

        safe = np.where(labels >= 0, labels, 0)
        mapped = target[safe]
        replace = (labels >= 0) & (mapped >= 0)

        if replace.any():
            palette = np.clip(
                np.rint(means), 0, 255
            ).astype(np.uint8)
            rgb[replace] = palette[mapped[replace]]

        return rgb

    @staticmethod
    def dominant_tolerance(strength, ceiling):
        """
        Map a 0-100 strength to the local color-cluster radius.

        For this method the strength drives the tolerance rather than an
        area: the radius is what decides how much tonal variation counts
        as "the same color", and it is the control that actually changes
        the result on dirty artwork.
        """
        ceiling = max(1, int(round(ceiling)))
        strength = max(0, min(100, float(strength)))

        return max(1, int(round(strength / 100.0 * ceiling)))

    @staticmethod
    def clean_dominant_color(rgb, alpha, tolerance, window=3,
                             max_similar=1, protect_silhouette=True):
        """
        Snap isolated pixels to the dominant color around them.

        Region merging needs regions to exist. On a dirty conversion,
        where almost every pixel is a slightly different tone, there are
        none, so this works on a local window instead: colors are
        bucketed at ``tolerance`` (which is what makes "most common
        color" meaningful when every pixel is unique), and a pixel is
        replaced by the mean of the window's dominant bucket only when
        at most ``max_similar`` window pixels share its own bucket.

        The isolation test compares colors by distance rather than by
        shared bucket, which is what protects intentional detail: a
        one-pixel window frame has near-identical pixels along its
        length and survives even when anti-aliasing puts each of them in
        a different bucket. A lone stray tone has no such neighbors.

        ``max_similar`` therefore doubles as a stylization dial: 1 keeps
        structure faithful, while 3 or 4 flattens organic texture such
        as foliage into larger blocks. Alpha is never modified.
        """
        rgb = np.array(rgb, copy=True)
        height, width = rgb.shape[:2]
        valid = alpha > 5

        if not valid.any():
            return rgb

        radius = max(1, int(window) // 2)
        step = max(1, int(tolerance))

        buckets = rgb.astype(np.int32) // step
        keys = (
            (buckets[:, :, 0] << 20)
            | (buckets[:, :, 1] << 10)
            | buckets[:, :, 2]
        ).astype(np.int32)
        keys = np.where(valid, keys, -1)

        offsets = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        count = len(offsets)
        center = count // 2

        def shifted(source, dy, dx, fill):
            out = np.full(source.shape, fill, dtype=source.dtype)
            rows = slice(max(0, dy), height + min(0, dy))
            columns = slice(max(0, dx), width + min(0, dx))
            into_rows = slice(max(0, -dy), height + min(0, -dy))
            into_columns = slice(max(0, -dx), width + min(0, -dx))
            out[into_rows, into_columns] = source[rows, columns]
            return out

        window_keys = np.stack(
            [shifted(keys, dy, dx, -1) for dy, dx in offsets]
        )
        window_valid = window_keys >= 0

        # How many window pixels share each candidate's bucket.
        tallies = np.zeros((count, height, width), dtype=np.int16)

        for index in range(count):
            same = (window_keys == window_keys[index]) & window_valid
            tallies[index] = same.sum(axis=0) * window_valid[index]

        best = tallies.argmax(axis=0)
        best_count = np.take_along_axis(tallies, best[None], axis=0)[0]
        dominant = np.take_along_axis(window_keys, best[None], axis=0)[0]

        # Isolation is measured by color distance, never by shared
        # bucket. Bucket boundaries fall arbitrarily, so on an
        # anti-aliased edge every pixel of a thin dark line lands in a
        # different bucket and the line reads as a row of isolated
        # pixels. That is what erased window frames and railings.
        values = rgb.astype(np.int32)
        limit = float(tolerance) ** 2
        own_count = np.zeros((height, width), dtype=np.int16)

        for dy, dx in offsets:
            neighbor = shifted(values, dy, dx, 0)
            reachable = shifted(valid, dy, dx, False)
            close = ((neighbor - values) ** 2).sum(axis=2) <= limit
            own_count += close & reachable

        replace = (
            valid
            & (own_count <= max_similar)
            & (best_count > own_count)
            & (dominant != keys)
        )

        if protect_silhouette:
            outside = np.pad(
                ~valid, 1, mode="constant", constant_values=True
            )
            touches = (
                outside[:-2, 1:-1]
                | outside[2:, 1:-1]
                | outside[1:-1, :-2]
                | outside[1:-1, 2:]
            )
            replace &= ~touches

        if not replace.any():
            return rgb

        # Accumulated one offset at a time; stacking the shifted colors
        # would cost hundreds of megabytes at 512 px with a 5 px window.
        totals = np.zeros((height, width), dtype=np.int32)
        accumulated = np.zeros((height, width, 3), dtype=np.float32)

        for index, (dy, dx) in enumerate(offsets):
            member = (window_keys[index] == dominant) & window_valid[index]

            if not member.any():
                continue

            totals += member
            accumulated += (
                shifted(rgb, dy, dx, 0).astype(np.float32)
                * member[:, :, None]
            )

        mean = accumulated / np.maximum(totals, 1)[:, :, None]
        rgb[replace] = np.clip(
            np.rint(mean[replace]), 0, 255
        ).astype(np.uint8)

        return rgb

    def clean_color_noise(self, img):
        self.noise_pixels_changed = 0

        if not self.settings.noise_cleanup:
            return img

        rgba = np.array(img.convert("RGBA"), copy=True)
        alpha = np.array(rgba[:, :, 3], copy=True)

        tolerance = int(round(self.settings.noise_tolerance))
        max_area = self.cleanup_max_area(
            self.settings.noise_strength,
            self.settings.noise_max_area
        )

        original = np.array(rgba[:, :, :3], copy=True)
        rgb = original
        method = self.settings.noise_method
        protect = bool(self.settings.noise_protect)

        if method in ("Local dominant color", "Both"):
            rgb = self.clean_dominant_color(
                rgb,
                alpha,
                self.dominant_tolerance(
                    self.settings.noise_strength, tolerance
                ),
                int(self.settings.noise_window),
                max(1, int(self.settings.noise_min_isolation)),
                protect
            )

        if method in ("Merge small regions", "Both"):
            # Grouping must stay below the merge tolerance, otherwise a
            # fragment is absorbed while grouping and never becomes a
            # candidate, and the controls appear to do nothing.
            group_tolerance = min(
                int(round(self.settings.noise_group_tolerance)),
                max(0, tolerance - 1)
            )

            rgb = self.clean_region_noise(
                rgb,
                alpha,
                tolerance,
                max_area,
                protect,
                group_tolerance
            )

        self.noise_pixels_changed = int(
            (rgb != original).any(axis=2).sum()
        )

        return Image.fromarray(
            np.dstack((rgb, alpha)).astype(np.uint8),
            "RGBA"
        )

    def apply_class_overrides(self, img):
        """
        Repaint whole material classes with the colors chosen for them.

        Each visible pixel is matched to its nearest detected class
        center; the ones whose class has a replacement color take it.
        Alpha is untouched, and classes without a replacement keep their
        original per-pixel colors rather than being flattened.
        """
        self.recolored_pixels = 0

        if self.class_colors is None or not self.class_overrides:
            return img

        if not len(self.class_colors):
            return img

        data = np.array(img.convert("RGBA"), copy=True)
        alpha = data[:, :, 3]
        visible = alpha > 5

        if not visible.any():
            return img

        pixels = data[:, :, :3][visible].astype(np.float64)
        assigned = self.assign_classes(pixels, self.class_colors)

        replacement = np.zeros((len(self.class_colors), 3), dtype=np.uint8)
        overridden = np.zeros(len(self.class_colors), dtype=bool)

        for index, colour in self.class_overrides.items():
            if 0 <= int(index) < len(replacement):
                replacement[int(index)] = colour
                overridden[int(index)] = True

        chosen = overridden[assigned]

        if not chosen.any():
            return img

        updated = data[:, :, :3][visible]
        updated[chosen] = replacement[assigned[chosen]]
        data[:, :, :3][visible] = updated

        self.recolored_pixels = int(chosen.sum())

        return Image.fromarray(data, "RGBA")

    def reduce_palette(self, img):
        """Quantize to the requested number of colors, alpha untouched."""
        colors = int(self.settings.palette)

        if colors <= 0:
            return img

        colors = max(2, min(256, colors))

        dither = (
            Image.Dither.FLOYDSTEINBERG
            if self.settings.dither
            else Image.Dither.NONE
        )

        rgb = img.convert(
            "RGB"
        ).quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=dither
        ).convert("RGB")

        return Image.merge(
            "RGBA",
            (
                *rgb.split(),
                img.getchannel("A")
            )
        )

    # ---------------------------------------------------------
    # Average color blend
    # ---------------------------------------------------------

    @staticmethod
    def average_color(rgb, mask):
        """Mean color of the masked pixels, or None when none are set."""
        if not mask.any():
            return None

        return rgb[mask].astype(np.float64).mean(axis=0)

    def apply_average_blend(self, img):
        """
        Pull colors towards one average, by a sliding amount.

        This is the blunt counterpart to recoloring a class: instead of
        replacing a color outright it drags every pixel towards a single
        average, so 0 leaves the image alone and 100 flattens the area
        to one flat tone. The average is taken either from the whole
        visible image or from the active selection, which is what makes
        it useful for unifying one material without sampling the rest.
        """
        self.average_blended_pixels = 0

        amount = max(0.0, min(100.0, float(self.settings.average_blend)))

        if amount <= 0:
            return img

        amount /= 100.0

        data = np.array(img.convert("RGBA"), copy=True)
        rgb = data[:, :, :3]
        visible = data[:, :, 3] > 5

        if not visible.any():
            return img

        sampled = visible

        if (
            self.settings.average_source == "Selection"
            and self.mask is not None
            and self.mask.shape == visible.shape
        ):
            selected = visible & self.mask

            # Falling back to the whole image keeps the slider useful
            # before any selection has been made.
            if selected.any():
                sampled = selected

        average = self.average_color(rgb, sampled)

        if average is None:
            return img

        blended = rgb.astype(np.float64)
        blended[visible] = (
            blended[visible] * (1.0 - amount) + average * amount
        )

        data[:, :, :3] = np.clip(
            np.rint(blended), 0, 255
        ).astype(np.uint8)

        self.average_blended_pixels = int(visible.sum())

        return Image.fromarray(data, "RGBA")

    # ---------------------------------------------------------
    # Outer border
    # ---------------------------------------------------------

    @staticmethod
    def outer_border_needs_margin(img):
        """Return whether visible content touches an edge of a fixed canvas."""
        alpha = np.asarray(img.convert("RGBA"))[:, :, 3]
        visible = alpha > 5
        return bool(
            visible[0, :].any()
            or visible[-1, :].any()
            or visible[:, 0].any()
            or visible[:, -1].any()
        )

    def add_outer_border(self, img):
        if not self.settings.border:
            return img

        # The canvas size is fixed, so an outside border can only occupy
        # transparent pixels that already exist inside the image.
        self.outer_border_clipped = self.outer_border_needs_margin(
            img
        )

        rgba = np.array(
            img.convert("RGBA"),
            copy=True
        )

        rgb = np.array(rgba[:, :, :3], copy=True)
        alpha = np.array(rgba[:, :, 3], copy=True)

        width = int(
            round(
                self.settings.border_width
            )
        )

        color = self.hex_to_rgb(
            self.settings.border_color
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

    @staticmethod
    def segment_color_regions(rgb, alpha, threshold, min_area):
        """
        Connected-component segmentation for pixel-art objects.

        Pixels are considered part of the same region when their RGB
        distance is <= threshold and they are 4-connected. This is more
        faithful than simply drawing every color boundary because broad
        areas of a similar color become one object while genuinely
        different regions remain separate.

        The flood fill walks horizontal runs of identical color rather
        than individual pixels. That is exactly equivalent — two
        identical adjacent pixels are at distance 0 and so always share
        a region, whatever the threshold — but pixel art is made of long
        uniform runs, so the graph is orders of magnitude smaller than
        the image. Runs are found with vectorized NumPy comparisons, and
        the fill itself uses plain Python integers because NumPy scalar
        indexing dominates the runtime otherwise.

        Memory is bounded by the image rather than by the number of
        regions: one reusable buffer holds the component being built,
        and no per-region pixel list is retained.
        """
        h, w = rgb.shape[:2]
        labels = np.full((h, w), -1, dtype=np.int32)

        if h == 0 or w == 0:
            return labels

        visible = alpha > 5

        if not visible.any():
            return labels

        channels = rgb.reshape(h, w, 3).astype(np.int32)

        # Squared RGB distances reach 195075, so comparisons are done in
        # int32; identity is compared through a single packed key.
        packed = (
            (channels[:, :, 0] << 16)
            | (channels[:, :, 1] << 8)
            | channels[:, :, 2]
        )

        # A pixel continues the run to its left when both are visible
        # and their colors are identical.
        continues = np.zeros((h, w), dtype=bool)
        continues[:, 1:] = (
            visible[:, 1:]
            & visible[:, :-1]
            & (packed[:, 1:] == packed[:, :-1])
        )

        begins = visible & ~continues

        finishes = np.zeros((h, w), dtype=bool)
        finishes[:, :-1] = visible[:, :-1] & ~continues[:, 1:]
        finishes[:, -1] = visible[:, -1]

        # Column 0 never continues a run, so no run spans two rows and
        # starts and ends pair up in order.
        run_start = np.flatnonzero(begins.ravel())
        run_end = np.flatnonzero(finishes.ravel())

        count = run_start.size

        if count == 0:
            return labels

        flat_start = run_start.tolist()
        flat_end = run_end.tolist()
        first_col = (run_start % w).tolist()
        last_col = (run_end % w).tolist()

        flat_channels = channels.reshape(-1, 3)
        reds = flat_channels[run_start, 0].tolist()
        greens = flat_channels[run_start, 1].tolist()
        blues = flat_channels[run_start, 2].tolist()

        # First run index of every row, plus a sentinel, for neighbors.
        row_begin = np.searchsorted(
            run_start, np.arange(h + 1) * w
        ).tolist()

        visited = bytearray(count)
        component = [0] * count
        stack = [0] * count

        labels_flat = labels.ravel()
        label = 0
        limit = float(threshold) ** 2

        for origin in range(count):
            if visited[origin]:
                continue

            seed_r = reds[origin]
            seed_g = greens[origin]
            seed_b = blues[origin]

            visited[origin] = 1
            stack[0] = origin
            top = 1
            found = 0
            area = 0

            while top:
                top -= 1
                current = stack[top]
                component[found] = current
                found += 1

                low = first_col[current]
                high = last_col[current]
                area += high - low + 1

                # Derived instead of stored: one list of large row
                # indices per run costs more than the arithmetic.
                row = flat_start[current] // w

                # Runs are maximal, so a same-row neighbor is a
                # different color and must pass the threshold test.
                if current > row_begin[row]:
                    neighbor = current - 1
                    if (
                        not visited[neighbor]
                        and last_col[neighbor] + 1 == low
                    ):
                        dr = reds[neighbor] - seed_r
                        dg = greens[neighbor] - seed_g
                        db = blues[neighbor] - seed_b
                        if dr * dr + dg * dg + db * db <= limit:
                            visited[neighbor] = 1
                            stack[top] = neighbor
                            top += 1

                neighbor = current + 1
                if neighbor < row_begin[row + 1]:
                    if (
                        not visited[neighbor]
                        and first_col[neighbor] == high + 1
                    ):
                        dr = reds[neighbor] - seed_r
                        dg = greens[neighbor] - seed_g
                        db = blues[neighbor] - seed_b
                        if dr * dr + dg * dg + db * db <= limit:
                            visited[neighbor] = 1
                            stack[top] = neighbor
                            top += 1

                # Vertical neighbors are the runs of the adjacent rows
                # whose column span overlaps this one.
                for other in (row - 1, row + 1):
                    if other < 0 or other >= h:
                        continue

                    scan = row_begin[other]
                    stop = row_begin[other + 1]

                    if scan == stop:
                        continue

                    scan = bisect_left(last_col, low, scan, stop)

                    while scan < stop and first_col[scan] <= high:
                        if not visited[scan]:
                            dr = reds[scan] - seed_r
                            dg = greens[scan] - seed_g
                            db = blues[scan] - seed_b
                            if dr * dr + dg * dg + db * db <= limit:
                                visited[scan] = 1
                                stack[top] = scan
                                top += 1

                        scan += 1

            if area >= min_area:
                for index in range(found):
                    run = component[index]
                    labels_flat[flat_start[run]:flat_end[run] + 1] = label

                label += 1

        return labels

    def add_object_borders(self, img):
        if not self.settings.object_borders:
            return img

        rgba = np.array(img.convert("RGBA"), copy=True)
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3].copy()

        h, w = rgb.shape[:2]
        if h < 2 or w < 2:
            return img

        threshold = int(round(self.settings.object_threshold))
        width = int(round(self.settings.object_border_width))
        min_area = int(round(self.settings.object_min_area))

        border_color = self.hex_to_rgb(
            self.settings.object_border_color
        )

        labels = self.segment_color_regions(
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

    def run(self, source):
        """
        Run the full pipeline and return the processed image.

        ``source`` is never modified, and stage-level warnings such as
        :attr:`outer_border_clipped` are refreshed on every call.
        """
        if source is None:
            raise ValueError(
                "Open an image first."
            )

        self.outer_border_clipped = False
        self.recolored_pixels = 0
        self.average_blended_pixels = 0
        img = source.copy()

        if self.settings.crop:
            img = self.crop_empty(img)

        if self.settings.cleanup:
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
            self.settings.contrast
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
            if self.settings.pixel_final
            else Image.Resampling.LANCZOS
        )

        img = img.resize(
            target,
            resample
        )

        if (
            self.settings.mode
            == "Pixel Art Reconstruction"
        ):
            img = self.restrict(
                img,
                self.reconstruct_pixel_art(img)
            )

        img = self.restrict(
            img,
            self.apply_color_effects(img)
        )

        img = self.restrict(
            img,
            self.reduce_palette(img)
        )

        # Cleanup runs on the final resolution, after palette reduction
        # and before any outline is drawn.
        img = self.restrict(
            img,
            self.clean_color_noise(img)
        )

        img = self.apply_class_overrides(
            img
        )

        img = self.restrict(
            img,
            self.apply_average_blend(img)
        )

        img = self.restrict(
            img,
            self.add_outer_border(img)
        )

        img = self.restrict(
            img,
            self.add_object_borders(img)
        )

        if self.settings.alpha:
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
