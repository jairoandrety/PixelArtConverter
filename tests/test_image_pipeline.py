import unittest
from dataclasses import fields

import numpy as np
from PIL import Image

from pixel_art_converter import PixelArtConverter
from pixel_pipeline import Pipeline, Settings


def pipeline(**overrides):
    """A pipeline for one run; no Tkinter and no stand-in variables."""
    return Pipeline(Settings(**overrides))


class SelfCallTests(unittest.TestCase):
    """
    Every ``self.name(...)`` in the GUI must resolve to something.

    The refactor moved the pipeline helpers out of the Tk class, and a
    call left behind only failed when a user clicked that one button.
    """

    @staticmethod
    def unresolved(path):
        import ast

        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        problems = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            known = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            known |= {
                item.target.attr
                for item in ast.walk(node)
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Attribute)
            }

            for item in ast.walk(node):
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            known.add(target.attr)

            for item in ast.walk(node):
                if not isinstance(item, ast.Call):
                    continue

                func = item.func

                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                    and func.attr not in known
                ):
                    problems.append(
                        f"{node.name}.{func.attr} (line {item.lineno})"
                    )

        return problems

    def test_the_interface_calls_only_methods_it_has(self):
        import pixel_art_converter

        self.assertEqual(
            self.unresolved(pixel_art_converter.__file__), []
        )

    def test_the_pipeline_calls_only_methods_it_has(self):
        import pixel_pipeline

        self.assertEqual(self.unresolved(pixel_pipeline.__file__), [])


class OverrideNormalizationTests(unittest.TestCase):
    """
    History entries are compared with ``==``.

    Class recolors used to be stored as the NumPy arrays hex_to_rgb
    returns, so comparing two states raised "truth value of an array is
    ambiguous" instead of comparing. It surfaced only when nothing else
    had changed, which is exactly what stepping between the objects of a
    class does.
    """

    def test_arrays_become_plain_tuples(self):
        overrides = {1: Pipeline.hex_to_rgb("#14c85a")}

        normalized = PixelArtConverter.normalize_overrides(overrides)

        self.assertEqual(normalized, {1: (20, 200, 90)})
        self.assertIsInstance(normalized[1], tuple)
        self.assertTrue(all(isinstance(v, int) for v in normalized[1]))

    def test_normalized_states_compare_without_raising(self):
        left = PixelArtConverter.normalize_overrides(
            {1: Pipeline.hex_to_rgb("#14c85a")}
        )
        right = PixelArtConverter.normalize_overrides(
            {1: Pipeline.hex_to_rgb("#14c85a")}
        )

        self.assertEqual((None, left), (None, right))

    def test_raw_arrays_would_have_raised(self):
        raw = {1: Pipeline.hex_to_rgb("#14c85a")}

        with self.assertRaises(ValueError):
            bool((None, raw) == (None, {1: Pipeline.hex_to_rgb("#14c85a")}))

    def test_differing_recolors_are_not_equal(self):
        left = PixelArtConverter.normalize_overrides({1: (1, 2, 3)})
        right = PixelArtConverter.normalize_overrides({1: (4, 5, 6)})

        self.assertNotEqual(left, right)

    def test_plain_tuples_pass_through_unchanged(self):
        self.assertEqual(
            PixelArtConverter.normalize_overrides({2: (7, 8, 9)}),
            {2: (7, 8, 9)},
        )


class SettingsTests(unittest.TestCase):
    def test_defaults_match_the_interface_defaults(self):
        defaults = Settings()

        for field in fields(Settings):
            with self.subTest(field=field.name):
                self.assertEqual(
                    PixelArtConverter.DEFAULTS[field.name],
                    getattr(defaults, field.name),
                )

    def test_presentation_only_keys_are_ignored(self):
        settings = Settings.from_mapping(
            PixelArtConverter.DEFAULTS | {"selection_mode": "Rectangle"}
        )

        self.assertFalse(hasattr(settings, "selection_mode"))
        self.assertEqual(settings.height, 256)

    def test_a_mapping_round_trips_through_settings(self):
        source = PixelArtConverter.DEFAULTS | {"palette": 32, "invert": True}
        settings = Settings.from_mapping(source)

        self.assertEqual(settings.palette, 32)
        self.assertTrue(settings.invert)

    def test_the_pipeline_never_imports_tkinter(self):
        import pixel_pipeline

        with open(pixel_pipeline.__file__, encoding="utf-8") as handle:
            self.assertNotIn("tkinter", handle.read())


class RegionSegmentationTests(unittest.TestCase):
    def test_minimum_area_discards_small_regions(self):
        rgb = np.array(
            [
                [[255, 0, 0], [0, 0, 255], [0, 0, 255]],
                [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
            ],
            dtype=np.uint8,
        )
        alpha = np.full((2, 3), 255, dtype=np.uint8)

        labels = Pipeline.segment_color_regions(
            rgb, alpha, threshold=0, min_area=2
        )

        self.assertEqual(labels[0, 0], -1)
        self.assertTrue(np.all(labels[:, 1:] == 0))

    def test_transparent_pixels_do_not_create_regions(self):
        rgb = np.full((2, 2, 3), 255, dtype=np.uint8)
        alpha = np.array([[255, 0], [0, 255]], dtype=np.uint8)

        labels = Pipeline.segment_color_regions(
            rgb, alpha, threshold=0, min_area=1
        )

        self.assertEqual(labels[0, 1], -1)
        self.assertEqual(labels[1, 0], -1)
        self.assertNotEqual(labels[0, 0], labels[1, 1])

    def test_region_at_the_exact_minimum_area_is_retained(self):
        rgb = np.full((2, 2, 3), 80, dtype=np.uint8)
        alpha = np.full((2, 2), 255, dtype=np.uint8)

        labels = Pipeline.segment_color_regions(
            rgb, alpha, threshold=0, min_area=4
        )

        self.assertTrue(np.all(labels == 0))

    def test_diagonal_pixels_are_separate_under_four_connectivity(self):
        rgb = np.full((2, 2, 3), 50, dtype=np.uint8)
        alpha = np.array([[255, 0], [0, 255]], dtype=np.uint8)

        labels = Pipeline.segment_color_regions(
            rgb, alpha, threshold=0, min_area=1
        )

        self.assertNotEqual(labels[0, 0], labels[1, 1])

    def test_color_tolerance_uses_seed_color_distance(self):
        rgb = np.array([[[10, 10, 10], [20, 10, 10]]], dtype=np.uint8)
        alpha = np.full((1, 2), 255, dtype=np.uint8)

        labels = Pipeline.segment_color_regions(
            rgb, alpha, threshold=9, min_area=1
        )
        self.assertNotEqual(labels[0, 0], labels[0, 1])

        labels = Pipeline.segment_color_regions(
            rgb, alpha, threshold=10, min_area=1
        )
        self.assertEqual(labels[0, 0], labels[0, 1])


class WheelEvent:
    def __init__(self, delta=0, num=0, state=0):
        self.delta = delta
        self.num = num
        self.state = state


class WheelNormalizationTests(unittest.TestCase):
    def test_windows_delta_maps_to_one_unit_per_notch(self):
        self.assertEqual(
            PixelArtConverter.wheel_steps(WheelEvent(delta=120)), -1
        )
        self.assertEqual(
            PixelArtConverter.wheel_steps(WheelEvent(delta=-120)), 1
        )
        self.assertEqual(
            PixelArtConverter.wheel_steps(WheelEvent(delta=-240)), 2
        )

    def test_x11_buttons_map_to_single_units(self):
        self.assertEqual(
            PixelArtConverter.wheel_steps(WheelEvent(num=4)), -1
        )
        self.assertEqual(
            PixelArtConverter.wheel_steps(WheelEvent(num=5)), 1
        )

    def test_small_trackpad_deltas_still_scroll(self):
        self.assertEqual(
            PixelArtConverter.wheel_steps(WheelEvent(delta=3)), -1
        )
        self.assertEqual(
            PixelArtConverter.wheel_steps(WheelEvent(delta=-3)), 1
        )

    def test_empty_event_produces_no_movement(self):
        self.assertEqual(PixelArtConverter.wheel_steps(WheelEvent()), 0)


class NumericValidationTests(unittest.TestCase):
    def test_valid_value_reports_no_error(self):
        self.assertIsNone(
            PixelArtConverter.validate_numeric(" 256 ", 0, 4096, "Height")
        )

    def test_empty_text_is_rejected(self):
        self.assertEqual(
            PixelArtConverter.validate_numeric("", 0, 4096, "Height"),
            "Height is empty.",
        )

    def test_non_numeric_text_is_rejected(self):
        self.assertEqual(
            PixelArtConverter.validate_numeric("12a", 0, 256, "Palette"),
            "Palette must be a whole number.",
        )

    def test_out_of_range_value_is_rejected(self):
        self.assertEqual(
            PixelArtConverter.validate_numeric("300", 0, 256, "Palette"),
            "Palette must be between 0 and 256.",
        )

    def test_bounds_are_inclusive(self):
        self.assertIsNone(
            PixelArtConverter.validate_numeric("0", 0, 256, "Palette")
        )
        self.assertIsNone(
            PixelArtConverter.validate_numeric("256", 0, 256, "Palette")
        )

    def test_every_limit_covers_a_real_setting(self):
        for key in PixelArtConverter.NUMERIC_LIMITS:
            self.assertIn(key, PixelArtConverter.DEFAULTS)


class NoiseCleanupTests(unittest.TestCase):
    @staticmethod
    def wall_with_speckle():
        rgb = np.full((9, 9, 3), 200, dtype=np.uint8)
        rgb[4, 4] = (206, 200, 200)
        alpha = np.full((9, 9), 255, dtype=np.uint8)
        return rgb, alpha

    def test_speckle_is_replaced_by_the_surrounding_color(self):
        rgb, alpha = self.wall_with_speckle()

        result = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=3, protect_silhouette=False
        )

        np.testing.assert_array_equal(result[4, 4], (200, 200, 200))

    def test_a_distant_color_is_never_merged(self):
        rgb, alpha = self.wall_with_speckle()
        rgb[2, 2] = (10, 10, 10)

        result = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=3, protect_silhouette=False
        )

        np.testing.assert_array_equal(result[2, 2], (10, 10, 10))

    def test_a_region_larger_than_the_limit_is_kept(self):
        rgb = np.full((9, 9, 3), 200, dtype=np.uint8)
        rgb[3:6, 3:6] = (206, 200, 200)
        alpha = np.full((9, 9), 255, dtype=np.uint8)

        result = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=4, protect_silhouette=False
        )

        np.testing.assert_array_equal(result, rgb)

    def test_the_input_array_is_not_modified(self):
        rgb, alpha = self.wall_with_speckle()
        original = rgb.copy()

        Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=3, protect_silhouette=False
        )

        np.testing.assert_array_equal(rgb, original)

    def test_silhouette_fragments_are_protected_by_default(self):
        rgb = np.full((7, 7, 3), 100, dtype=np.uint8)
        rgb[1, 3] = (106, 100, 100)
        alpha = np.full((7, 7), 255, dtype=np.uint8)
        alpha[0, :] = 0

        protected = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=3, protect_silhouette=True
        )
        merged = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=3, protect_silhouette=False
        )

        np.testing.assert_array_equal(protected[1, 3], (106, 100, 100))
        np.testing.assert_array_equal(merged[1, 3], (100, 100, 100))

    def test_two_speckles_never_merge_into_each_other(self):
        rgb = np.array(
            [[[10, 10, 10], [12, 10, 10]]], dtype=np.uint8
        )
        alpha = np.full((1, 2), 255, dtype=np.uint8)

        result = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=1, protect_silhouette=False
        )

        np.testing.assert_array_equal(result, rgb)

    def test_a_fully_transparent_image_is_returned_untouched(self):
        rgb = np.full((4, 4, 3), 50, dtype=np.uint8)
        alpha = np.zeros((4, 4), dtype=np.uint8)

        result = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=3
        )

        np.testing.assert_array_equal(result, rgb)

    def test_alpha_is_preserved_through_the_pipeline_stage(self):
        rgb, alpha = self.wall_with_speckle()
        alpha[0, 0] = 0
        image = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
        app = pipeline(noise_cleanup=True, noise_protect=False)

        result = app.clean_color_noise(image)

        np.testing.assert_array_equal(
            np.asarray(result)[:, :, 3], alpha
        )

    def test_cleanup_is_skipped_when_the_toggle_is_off(self):
        rgb, alpha = self.wall_with_speckle()
        image = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
        app = pipeline(noise_cleanup=False)

        self.assertIs(app.clean_color_noise(image), image)


class CleanupResponsivenessTests(unittest.TestCase):
    """Regressions for controls that silently did nothing."""

    @staticmethod
    def continuous_art(size=64):
        rows, columns = np.mgrid[0:size, 0:size]
        rgb = np.stack(
            [
                120 + columns * 0.5,
                90 + rows * 0.5,
                160 - columns * 0.3,
            ],
            axis=2,
        )
        rgb += np.random.default_rng(1).normal(0, 3, rgb.shape)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        alpha = np.full((size, size), 255, dtype=np.uint8)
        return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

    def run_cleanup(self, **overrides):
        overrides.setdefault("noise_method", "Merge small regions")
        app = pipeline(
            crop=False, progressive=False, mode="Pixel Perfect Resize",
            fit_mode="exact", width=64, height=64, alpha=True,
            noise_cleanup=True, **overrides
        )
        app.run(self.continuous_art())
        return app.noise_pixels_changed

    def test_continuous_art_is_cleaned_with_the_default_grouping(self):
        self.assertGreater(self.run_cleanup(), 0)

    def test_grouping_zero_barely_touches_continuous_art(self):
        # The defect this suite exists for: with exact-color grouping,
        # almost every pixel is its own fragment on continuous-tone art,
        # so the controls looked inert. Grouping must stay adjustable.
        exact = self.run_cleanup(noise_group_tolerance=0)
        grouped = self.run_cleanup(noise_group_tolerance=2)

        self.assertLess(exact * 10, grouped)

    def test_strength_is_monotonic(self):
        counts = [
            self.run_cleanup(noise_strength=strength)
            for strength in (0, 10, 30, 50, 75, 100)
        ]

        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[-1], 0)

    def test_grouping_at_or_above_the_merge_tolerance_is_clamped(self):
        # Without the clamp a fragment is absorbed while grouping and
        # never becomes a candidate, so the controls appear inert.
        self.assertGreater(
            self.run_cleanup(
                noise_group_tolerance=30, noise_tolerance=5
            ),
            0,
        )

    def test_a_disabled_toggle_reports_no_change(self):
        app = pipeline(
            crop=False, progressive=False, mode="Pixel Perfect Resize",
            fit_mode="exact", width=64, height=64, alpha=True,
            noise_cleanup=False,
        )
        app.run(self.continuous_art())

        self.assertEqual(app.noise_pixels_changed, 0)


class ElementDetectionTests(unittest.TestCase):
    @staticmethod
    def scene():
        """Two wall tones, four windows, and a patch of foliage."""
        rgb = np.full((40, 40, 3), (150, 70, 45), dtype=np.uint8)
        rgb[:, 20:] = (240, 130, 55)

        for y in (5, 25):
            for x in (5, 25):
                rgb[y:y + 6, x:x + 6] = (20, 50, 110)

        rgb[34:, :] = (60, 110, 40)
        alpha = np.full((40, 40), 255, dtype=np.uint8)
        return rgb, alpha

    def test_clustering_is_deterministic(self):
        rgb, alpha = self.scene()

        first, centers_a = Pipeline.cluster_colors(rgb, alpha, 6)
        second, centers_b = Pipeline.cluster_colors(rgb, alpha, 6)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(centers_a, centers_b)

    def test_distinct_materials_land_in_distinct_classes(self):
        rgb, alpha = self.scene()
        labels, _ = Pipeline.cluster_colors(rgb, alpha, 6)

        wall_left = labels[15, 2]
        wall_right = labels[15, 38]
        window = labels[7, 7]
        foliage = labels[38, 20]

        self.assertEqual(
            len({wall_left, wall_right, window, foliage}), 4
        )

    def test_every_window_joins_the_same_class(self):
        rgb, alpha = self.scene()
        labels, _ = Pipeline.cluster_colors(rgb, alpha, 6)

        corners = {labels[y + 2, x + 2] for y in (5, 25) for x in (5, 25)}

        self.assertEqual(len(corners), 1)

    def test_transparent_pixels_are_unclassified(self):
        rgb, alpha = self.scene()
        alpha[0, :] = 0

        labels, _ = Pipeline.cluster_colors(rgb, alpha, 6)

        self.assertTrue((labels[0, :] == -1).all())
        self.assertTrue((labels[1:] >= 0).all())

    def test_class_count_never_exceeds_the_distinct_colors(self):
        rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
        rgb[:, 4:] = 200
        alpha = np.full((8, 8), 255, dtype=np.uint8)

        labels, centers = Pipeline.cluster_colors(rgb, alpha, 12)

        self.assertLessEqual(len(centers), 2)
        self.assertEqual(len(np.unique(labels[labels >= 0])), 2)

    def test_a_fully_transparent_image_yields_nothing(self):
        rgb, alpha = self.scene()
        labels, centers = Pipeline.cluster_colors(
            rgb, np.zeros_like(alpha), 6
        )

        self.assertTrue((labels == -1).all())
        self.assertEqual(len(centers), 0)

    def test_instances_are_the_connected_objects_of_a_class(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:5, 2:5] = True
        mask[12:16, 12:16] = True

        instances = Pipeline.class_instances(mask)

        self.assertEqual(int(instances.max()) + 1, 2)
        self.assertNotEqual(instances[3, 3], instances[13, 13])
        self.assertEqual(instances[10, 10], -1)

    def test_small_objects_are_filtered_out(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:8, 2:8] = True
        mask[15, 15] = True

        self.assertEqual(int(Pipeline.class_instances(mask, 1).max()) + 1, 2)
        self.assertEqual(int(Pipeline.class_instances(mask, 4).max()) + 1, 1)

    def test_classes_are_described_largest_first(self):
        rgb, alpha = self.scene()
        labels, centers = Pipeline.cluster_colors(rgb, alpha, 6)

        described = Pipeline.describe_classes(labels, centers, 1)

        self.assertEqual(
            [entry["pixels"] for entry in described],
            sorted((entry["pixels"] for entry in described), reverse=True),
        )
        self.assertAlmostEqual(
            sum(entry["share"] for entry in described), 1.0, places=5
        )

    def test_the_window_class_reports_four_objects(self):
        rgb, alpha = self.scene()
        labels, centers = Pipeline.cluster_colors(rgb, alpha, 6)
        described = Pipeline.describe_classes(labels, centers, 2)

        window_class = labels[7, 7]
        entry = next(e for e in described if e["id"] == window_class)

        self.assertEqual(entry["instances"], 4)

    def test_detect_elements_accepts_an_image(self):
        rgb, alpha = self.scene()
        image = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

        labels, centers, described = pipeline().detect_elements(image, 6, 2)

        self.assertEqual(labels.shape, (40, 40))
        self.assertTrue(described)
        self.assertLessEqual(len(centers), 6)


class AverageBlendTests(unittest.TestCase):
    SIZE = 12

    @staticmethod
    def gradient():
        columns = np.mgrid[0:12, 0:12][1]
        rgb = np.stack(
            [columns * 20, np.full((12, 12), 80), 255 - columns * 20], axis=2
        ).astype(np.uint8)
        alpha = np.full((12, 12), 255, dtype=np.uint8)
        return rgb, alpha

    def image(self):
        rgb, alpha = self.gradient()
        return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

    def build(self, amount, mask=None, source="Whole image", local=False):
        return Pipeline(
            Settings(
                crop=False, progressive=False, mode="Pixel Perfect Resize",
                fit_mode="exact", width=12, height=12, alpha=True,
                contrast=1.0, average_blend=amount, average_source=source,
                local_effects=local,
            ),
            mask=mask,
        )

    def test_zero_leaves_the_image_alone(self):
        original = np.asarray(self.image())
        result = np.asarray(self.build(0).run(self.image()))

        np.testing.assert_array_equal(result, original)

    def test_full_blend_flattens_to_one_tone(self):
        result = np.asarray(self.build(100).run(self.image()))
        visible = result[:, :, 3] > 5

        self.assertEqual(len(np.unique(result[:, :, :3][visible], axis=0)), 1)

    def test_the_blend_is_monotonic(self):
        spreads = [
            float(
                np.asarray(self.build(amount).run(self.image()))[:, :, :3]
                .astype(float).std()
            )
            for amount in (0, 25, 50, 75, 100)
        ]

        self.assertEqual(spreads, sorted(spreads, reverse=True))

    def test_the_average_is_the_mean_of_the_sampled_pixels(self):
        rgb, _ = self.gradient()
        expected = np.rint(rgb.reshape(-1, 3).mean(axis=0)).astype(int)

        result = np.asarray(self.build(100).run(self.image()))

        np.testing.assert_allclose(result[0, 0][:3], expected, atol=1)

    def test_a_selection_source_samples_only_the_selection(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[:, :4] = True

        rgb, _ = self.gradient()
        expected = np.rint(rgb[mask].mean(axis=0)).astype(int)

        result = np.asarray(
            self.build(100, mask=mask, source="Selection").run(self.image())
        )

        np.testing.assert_allclose(result[0, 0][:3], expected, atol=1)

    def test_an_empty_selection_falls_back_to_the_whole_image(self):
        mask = np.zeros((12, 12), dtype=bool)

        selected = np.asarray(
            self.build(100, mask=mask, source="Selection").run(self.image())
        )
        whole = np.asarray(self.build(100).run(self.image()))

        np.testing.assert_array_equal(selected, whole)

    def test_local_effects_confine_the_blend(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[:, :4] = True

        original = np.asarray(self.image())[:, :, :3]
        result = np.asarray(
            self.build(
                100, mask=mask, source="Selection", local=True
            ).run(self.image())
        )[:, :, :3]

        self.assertEqual(len(np.unique(result[mask], axis=0)), 1)
        np.testing.assert_array_equal(result[~mask], original[~mask])

    def test_alpha_is_never_modified(self):
        rgb, alpha = self.gradient()
        alpha[0, 0] = 0
        image = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

        result = np.asarray(self.build(100).run(image))

        np.testing.assert_array_equal(result[:, :, 3], alpha)


class RestrictedStageTests(unittest.TestCase):
    """Every final-resolution stage must honour the selection."""

    @staticmethod
    def scene():
        """Shaded wall, a window, and a speckle inside the masked half."""
        rows = np.mgrid[0:16, 0:16][0]
        rgb = np.stack(
            [150 + rows, 70 + rows // 2, 45 + rows // 3], axis=2
        ).astype(float)
        # Reconstruction only acts on texture, so the wall needs some.
        rgb += np.random.default_rng(3).normal(0, 15, rgb.shape)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb[2:6, 4:8] = (30, 60, 140)
        rgb[3, 10] = (200, 90, 60)
        alpha = np.full((16, 16), 255, dtype=np.uint8)
        alpha[0, :] = 0
        return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

    def run_with(self, mask, **overrides):
        settings = dict(
            crop=False, progressive=False,
            mode="Pixel Perfect Resize", fit_mode="exact",
            width=16, height=16, alpha=True, contrast=1.0,
            local_effects=True,
        )
        settings.update(overrides)

        return np.asarray(
            Pipeline(Settings(**settings), mask=mask).run(self.scene())
        )[:, :, :3]

    def assert_confined(self, **overrides):
        mask = np.zeros((16, 16), dtype=bool)
        mask[:8] = True

        before = self.run_with(mask)
        after = self.run_with(mask, **overrides)
        changed = (after != before).any(axis=2)

        self.assertTrue(changed.any(), f"{overrides} changed nothing")
        self.assertEqual(int((changed & ~mask).sum()), 0)

    def test_color_effects_are_confined(self):
        self.assert_confined(invert=True)

    def test_palette_reduction_is_confined(self):
        self.assert_confined(palette=2)

    def test_noise_cleanup_is_confined(self):
        self.assert_confined(noise_cleanup=True, noise_strength=100)

    def test_internal_outlines_are_confined(self):
        self.assert_confined(object_borders=True, object_threshold=30)

    def test_the_average_blend_is_confined(self):
        self.assert_confined(average_blend=100)

    def test_reconstruction_is_confined(self):
        self.assert_confined(mode="Pixel Art Reconstruction")


class ClassRecolorTests(unittest.TestCase):
    @staticmethod
    def two_tone():
        rgb = np.full((10, 10, 3), (200, 60, 40), dtype=np.uint8)
        rgb[:, 5:] = (40, 60, 200)
        alpha = np.full((10, 10), 255, dtype=np.uint8)
        return rgb, alpha

    def build(self, overrides, centers=((200, 60, 40), (40, 60, 200))):
        return Pipeline(
            Settings(
                crop=False, progressive=False, mode="Pixel Perfect Resize",
                fit_mode="exact", width=10, height=10, alpha=True,
                contrast=1.0,
            ),
            class_colors=np.array(centers, dtype=np.float64),
            class_overrides=overrides,
        )

    def image(self):
        rgb, alpha = self.two_tone()
        return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

    def test_one_class_takes_its_replacement_color(self):
        app = self.build({1: (10, 220, 90)})

        result = np.asarray(app.run(self.image()))

        np.testing.assert_array_equal(result[0, 7][:3], (10, 220, 90))
        np.testing.assert_array_equal(result[0, 2][:3], (200, 60, 40))

    def test_several_classes_are_recolored_at_once(self):
        app = self.build({0: (1, 2, 3), 1: (4, 5, 6)})

        result = np.asarray(app.run(self.image()))

        np.testing.assert_array_equal(result[0, 2][:3], (1, 2, 3))
        np.testing.assert_array_equal(result[0, 7][:3], (4, 5, 6))
        self.assertEqual(app.recolored_pixels, 100)

    def test_classes_without_a_replacement_keep_their_own_pixels(self):
        rgb, alpha = self.two_tone()
        rgb[3, 1] = (210, 70, 50)
        image = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

        result = np.asarray(self.build({1: (10, 220, 90)}).run(image))

        np.testing.assert_array_equal(result[3, 1][:3], (210, 70, 50))

    def test_alpha_is_never_modified(self):
        rgb, alpha = self.two_tone()
        alpha[0, 0] = 0
        image = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

        result = np.asarray(self.build({1: (10, 220, 90)}).run(image))

        np.testing.assert_array_equal(result[:, :, 3], alpha)

    def test_no_overrides_leaves_the_image_alone(self):
        plain = np.asarray(self.build({}).run(self.image()))
        none = np.asarray(
            Pipeline(
                Settings(
                    crop=False, progressive=False,
                    mode="Pixel Perfect Resize", fit_mode="exact",
                    width=10, height=10, alpha=True, contrast=1.0,
                )
            ).run(self.image())
        )

        np.testing.assert_array_equal(plain, none)
        self.assertEqual(self.build({}).recolored_pixels, 0)

    def test_an_out_of_range_class_is_ignored(self):
        app = self.build({99: (1, 2, 3)})

        result = np.asarray(app.run(self.image()))

        np.testing.assert_array_equal(result[0, 2][:3], (200, 60, 40))
        self.assertEqual(app.recolored_pixels, 0)

    def test_a_recolor_survives_other_settings(self):
        # The point of keeping a class map rather than editing pixels:
        # the replacement is reapplied on every render.
        app = Pipeline(
            Settings(
                crop=False, progressive=False, mode="Pixel Perfect Resize",
                fit_mode="exact", width=10, height=10, alpha=True,
                contrast=1.0, border=True, noise_cleanup=True,
            ),
            class_colors=np.array(
                [(200, 60, 40), (40, 60, 200)], dtype=np.float64
            ),
            class_overrides={1: (10, 220, 90)},
        )

        result = np.asarray(app.run(self.image()))

        np.testing.assert_array_equal(result[5, 7][:3], (10, 220, 90))


class DominantColorTests(unittest.TestCase):
    """The method for dirty artwork, where no real regions exist."""

    SIZE = 64

    @classmethod
    def facade(cls, sigma=0):
        """Brick wall with one-pixel mortar joints and a window."""
        size = cls.SIZE
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[:, :] = (150, 70, 45)
        rgb[(np.mgrid[0:size, 0:size][0] % 6) < 1] = (120, 55, 35)
        rgb[10:20, 10:22] = (40, 60, 110)

        if sigma:
            noise = np.random.default_rng(2).normal(0, sigma, rgb.shape)
            rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)

        return rgb

    @staticmethod
    def alpha(size):
        return np.full((size, size), 255, dtype=np.uint8)

    @staticmethod
    def local_variance(img):
        values = img.astype(np.float64)
        padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
        window = np.stack([
            padded[i:i + img.shape[0], j:j + img.shape[1]]
            for i in range(3) for j in range(3)
        ])
        return float(window.var(axis=0).mean())

    def clean(self, rgb, strength, **kwargs):
        return Pipeline.clean_dominant_color(
            rgb,
            self.alpha(self.SIZE),
            Pipeline.dominant_tolerance(strength, 20),
            protect_silhouette=False,
            **kwargs
        )

    def test_flat_artwork_is_never_touched(self):
        # The guarantee that makes this safe as the default: a clean
        # image must come out identical at every strength.
        flat = self.facade()

        for strength in (0, 10, 30, 50, 75, 100):
            with self.subTest(strength=strength):
                np.testing.assert_array_equal(
                    self.clean(flat, strength), flat
                )

    def test_dirty_artwork_gets_less_noisy_as_strength_rises(self):
        dirty = self.facade(sigma=12)

        variances = [
            self.local_variance(self.clean(dirty, strength))
            for strength in (0, 30, 50, 75, 100)
        ]

        self.assertEqual(variances, sorted(variances, reverse=True))
        self.assertLess(variances[-1], self.local_variance(dirty) * 0.8)

    def test_one_pixel_lines_survive_at_the_default_isolation(self):
        dirty = self.facade(sigma=12)
        cleaned = self.clean(dirty, 100)

        # Mortar rows must stay darker than the brick rows around them.
        mortar = cleaned[24, 30:60].astype(int).mean()
        brick = cleaned[26, 30:60].astype(int).mean()

        self.assertLess(mortar, brick)

    def test_an_antialiased_thin_line_survives(self):
        # The defect this guards: a one-pixel window frame softened by
        # downscaling put every one of its pixels in a different bucket,
        # so the line read as isolated pixels and was erased.
        size = 32
        rgb = np.full((size, size, 3), 200, dtype=np.uint8)
        rgb[:, 15] = 40
        rgb[:, 14] = 120
        rgb[:, 16] = 120

        # Give the line a per-row wobble, as anti-aliasing would.
        wobble = np.random.default_rng(9).integers(-6, 7, size)
        for row in range(size):
            rgb[row, 14:17] = np.clip(
                rgb[row, 14:17].astype(int) + wobble[row], 0, 255
            )

        alpha = np.full((size, size), 255, dtype=np.uint8)
        cleaned = Pipeline.clean_dominant_color(
            rgb, alpha, tolerance=20, protect_silhouette=False
        )

        # The line must stay clearly darker than the field beside it.
        line = cleaned[:, 15].astype(int).mean()
        field = cleaned[:, 5].astype(int).mean()

        self.assertLess(line, field - 100)

    def test_isolated_pixels_are_still_cleaned_beside_a_line(self):
        size = 32
        rgb = np.full((size, size, 3), 200, dtype=np.uint8)
        rgb[:, 15] = 40
        rgb[7, 5] = (170, 200, 200)

        alpha = np.full((size, size), 255, dtype=np.uint8)
        cleaned = Pipeline.clean_dominant_color(
            rgb, alpha, tolerance=20, protect_silhouette=False
        )

        np.testing.assert_array_equal(cleaned[7, 5], (200, 200, 200))
        np.testing.assert_array_equal(cleaned[:, 15], rgb[:, 15])

    def test_raising_isolation_eats_thin_detail(self):
        flat = self.facade()

        eaten = self.clean(flat, 60, max_similar=3)

        self.assertFalse(np.array_equal(eaten, flat))

    def test_a_larger_window_cleans_more(self):
        dirty = self.facade(sigma=12)

        narrow = self.local_variance(self.clean(dirty, 60, window=3))
        wide = self.local_variance(self.clean(dirty, 60, window=5))

        self.assertLess(wide, narrow)

    def test_alpha_is_never_modified(self):
        dirty = self.facade(sigma=12)
        alpha = self.alpha(self.SIZE)
        alpha[0, :] = 0

        result = Pipeline.clean_dominant_color(dirty, alpha, 10)

        self.assertEqual(result.shape, dirty.shape)

    def test_a_fully_transparent_image_is_returned_untouched(self):
        dirty = self.facade(sigma=12)
        alpha = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)

        np.testing.assert_array_equal(
            Pipeline.clean_dominant_color(dirty, alpha, 10), dirty
        )

    def test_the_input_array_is_not_modified(self):
        dirty = self.facade(sigma=12)
        original = dirty.copy()

        self.clean(dirty, 100)

        np.testing.assert_array_equal(dirty, original)

    def test_both_methods_run_in_sequence(self):
        dirty = self.facade(sigma=12)
        image = Image.fromarray(
            np.dstack((dirty, self.alpha(self.SIZE))), "RGBA"
        )

        counts = {}
        for method in ("Local dominant color", "Merge small regions", "Both"):
            app = pipeline(
                noise_cleanup=True, noise_method=method,
                noise_strength=100, noise_protect=False,
            )
            app.clean_color_noise(image)
            counts[method] = app.noise_pixels_changed

        self.assertGreater(counts["Local dominant color"], 0)
        self.assertGreaterEqual(
            counts["Both"], counts["Local dominant color"]
        )


class NeighborRuleTests(unittest.TestCase):
    def test_a_fragment_needs_a_strictly_larger_neighbor(self):
        rgb = np.array([[[10, 10, 10], [12, 10, 10]]], dtype=np.uint8)
        alpha = np.full((1, 2), 255, dtype=np.uint8)

        result = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=1, protect_silhouette=False
        )

        np.testing.assert_array_equal(result, rgb)

    def test_a_larger_neighbor_absorbs_the_fragment(self):
        rgb = np.full((5, 5, 3), 100, dtype=np.uint8)
        rgb[2, 2] = (104, 100, 100)
        alpha = np.full((5, 5), 255, dtype=np.uint8)

        result = Pipeline.clean_region_noise(
            rgb, alpha, tolerance=10, max_area=1, protect_silhouette=False
        )

        np.testing.assert_array_equal(result[2, 2], (100, 100, 100))


class CleanupStrengthTests(unittest.TestCase):
    def test_strength_scales_up_to_the_ceiling(self):
        self.assertEqual(Pipeline.cleanup_max_area(100, 12), 12)
        self.assertEqual(Pipeline.cleanup_max_area(50, 12), 6)

    def test_low_strength_only_reaches_single_pixels(self):
        self.assertEqual(Pipeline.cleanup_max_area(0, 12), 1)
        self.assertEqual(Pipeline.cleanup_max_area(5, 12), 1)

    def test_the_ceiling_is_never_exceeded(self):
        for strength in range(0, 201, 25):
            self.assertLessEqual(
                Pipeline.cleanup_max_area(strength, 4), 4
            )


class SelectionTests(unittest.TestCase):
    def test_rectangle_covers_an_inclusive_range(self):
        mask = PixelArtConverter.rectangle_mask((10, 10), (2, 3), (5, 7))

        self.assertEqual(int(mask.sum()), 20)
        self.assertTrue(mask[3, 2])
        self.assertTrue(mask[7, 5])
        self.assertFalse(mask[2, 2])
        self.assertFalse(mask[8, 5])

    def test_drag_direction_does_not_matter(self):
        forward = PixelArtConverter.rectangle_mask((8, 8), (1, 1), (4, 4))
        backward = PixelArtConverter.rectangle_mask((8, 8), (4, 4), (1, 1))

        np.testing.assert_array_equal(forward, backward)

    def test_rectangle_is_clipped_to_the_image(self):
        mask = PixelArtConverter.rectangle_mask((4, 4), (-5, -5), (99, 99))

        self.assertTrue(mask.all())

    def test_region_selection_matches_the_outline_segmentation(self):
        rgb = np.full((5, 5, 3), 90, dtype=np.uint8)
        rgb[1:3, 1:3] = (200, 30, 30)
        alpha = np.full((5, 5), 255, dtype=np.uint8)

        mask = PixelArtConverter.region_mask(rgb, alpha, (1, 1), 0)

        self.assertEqual(int(mask.sum()), 4)
        self.assertTrue(mask[1:3, 1:3].all())

    def test_clicking_a_transparent_pixel_selects_nothing(self):
        rgb = np.full((3, 3, 3), 90, dtype=np.uint8)
        alpha = np.full((3, 3), 255, dtype=np.uint8)
        alpha[0, 0] = 0

        mask = PixelArtConverter.region_mask(rgb, alpha, (0, 0), 0)

        self.assertEqual(int(mask.sum()), 0)


class HistoryTests(unittest.TestCase):
    def test_a_changed_setting_is_described(self):
        before = {"invert": False, "border_width": 1}
        after = {"invert": True, "border_width": 1}

        self.assertEqual(
            PixelArtConverter.describe_change(before, after), "Invert on"
        )

    def test_an_unchanged_snapshot_produces_no_entry(self):
        state = {"invert": False, "border_width": 1}

        self.assertIsNone(PixelArtConverter.describe_change(state, state))

    def test_many_changes_collapse_into_a_count(self):
        before = {f"k{i}": 0 for i in range(6)}
        after = {f"k{i}": 1 for i in range(6)}

        self.assertEqual(
            PixelArtConverter.describe_change(before, after), "6 settings"
        )

    def test_float_settings_are_rounded_for_display(self):
        self.assertEqual(
            PixelArtConverter.describe_change(
                {"contrast": 1.0}, {"contrast": 1.0625}
            ),
            "Contrast 1.06",
        )


class ProgressiveResizeTests(unittest.TestCase):
    def test_a_wide_source_is_driven_by_its_width(self):
        self.assertEqual(
            Pipeline.dominant_resize_axis((4000, 200), (256, 13)), "width"
        )

    def test_a_tall_source_is_driven_by_its_height(self):
        self.assertEqual(
            Pipeline.dominant_resize_axis((200, 4000), (13, 256)), "height"
        )

    def test_a_proportional_resize_keeps_the_historical_axis(self):
        self.assertEqual(
            Pipeline.dominant_resize_axis((1000, 1000), (256, 256)), "height"
        )

    def test_a_very_wide_source_still_steps_down(self):
        image = Image.new("RGBA", (4000, 200), (10, 20, 30, 255))
        app = pipeline(progressive=True, fit_mode="width", width=256)

        stepped = app.progressive_resize(
            image, app.calculate_target(image)
        )

        self.assertLess(stepped.width, image.width)
        self.assertEqual(stepped.width, 320)

    def test_wide_and_tall_sources_are_treated_equivalently(self):
        wide = Image.new("RGBA", (4000, 200), (10, 20, 30, 255))
        tall = Image.new("RGBA", (200, 4000), (10, 20, 30, 255))

        wide_app = pipeline(progressive=True, fit_mode="width", width=256)
        tall_app = pipeline(progressive=True, fit_mode="height", height=256)

        stepped_wide = wide_app.progressive_resize(
            wide, wide_app.calculate_target(wide)
        )
        stepped_tall = tall_app.progressive_resize(
            tall, tall_app.calculate_target(tall)
        )

        self.assertEqual(stepped_wide.size, stepped_tall.size[::-1])

    def test_the_ladder_is_skipped_when_disabled(self):
        image = Image.new("RGBA", (4000, 200), (10, 20, 30, 255))
        app = pipeline(progressive=False, fit_mode="width", width=256)

        self.assertIs(
            app.progressive_resize(image, (256, 13)), image
        )


class LocalEffectTests(unittest.TestCase):
    @staticmethod
    def flat_source():
        rgb = np.full((6, 6, 3), 200, dtype=np.uint8)
        alpha = np.full((6, 6), 255, dtype=np.uint8)
        return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

    @staticmethod
    def corner_mask():
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:3, 0:3] = True
        return mask

    def settings(self, **overrides):
        return Settings(
            crop=False, progressive=False, mode="Pixel Perfect Resize",
            fit_mode="exact", width=6, height=6, alpha=True, invert=True,
            **overrides
        )

    def test_an_effect_is_confined_to_the_mask(self):
        result = np.asarray(
            Pipeline(
                self.settings(local_effects=True), mask=self.corner_mask()
            ).run(self.flat_source())
        )

        np.testing.assert_array_equal(result[0, 0][:3], (55, 55, 55))
        np.testing.assert_array_equal(result[5, 5][:3], (200, 200, 200))

    def test_without_the_toggle_the_effect_stays_global(self):
        result = np.asarray(
            Pipeline(
                self.settings(local_effects=False), mask=self.corner_mask()
            ).run(self.flat_source())
        )

        np.testing.assert_array_equal(result[5, 5][:3], (55, 55, 55))

    def test_a_mask_of_the_wrong_shape_is_ignored(self):
        stale = np.ones((3, 3), dtype=bool)

        guarded = np.asarray(
            Pipeline(self.settings(local_effects=True), mask=stale).run(
                self.flat_source()
            )
        )
        globalized = np.asarray(
            Pipeline(self.settings()).run(self.flat_source())
        )

        np.testing.assert_array_equal(guarded, globalized)

    def test_alpha_is_preserved_under_a_local_effect(self):
        rgb = np.full((6, 6, 3), 200, dtype=np.uint8)
        alpha = np.full((6, 6), 255, dtype=np.uint8)
        alpha[0, 0] = 0
        source = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")

        result = np.asarray(
            Pipeline(
                self.settings(local_effects=True), mask=self.corner_mask()
            ).run(source)
        )

        self.assertEqual(result[0, 0][3], 0)


class BrushTests(unittest.TestCase):
    def test_a_single_pixel_dab(self):
        mask = PixelArtConverter.brush_mask((9, 9), (4, 4), 1)

        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(mask[4, 4])

    def test_larger_diameters_cover_more(self):
        sizes = [
            int(PixelArtConverter.brush_mask((15, 15), (7, 7), d).sum())
            for d in (1, 3, 5, 9)
        ]

        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(set(sizes)), len(sizes))

    def test_a_dab_is_clipped_to_the_image(self):
        mask = PixelArtConverter.brush_mask((9, 9), (0, 0), 5)

        self.assertEqual(mask.shape, (9, 9))
        self.assertTrue(mask[0, 0])

    def test_a_stroke_is_continuous(self):
        points = PixelArtConverter.stroke_points((0, 0), (3, 3))

        self.assertEqual(points, [(0, 0), (1, 1), (2, 2), (3, 3)])

    def test_a_stroke_that_did_not_move_is_one_point(self):
        self.assertEqual(
            PixelArtConverter.stroke_points((2, 2), (2, 2)), [(2, 2)]
        )

    def test_a_fast_drag_leaves_no_gap(self):
        points = PixelArtConverter.stroke_points((0, 0), (0, 9))

        self.assertEqual(len(points), 10)
        self.assertEqual(
            sorted({y for _, y in points}), list(range(10))
        )


class SelectionPackingTests(unittest.TestCase):
    def test_a_mask_round_trips_through_packing(self):
        mask = np.zeros((7, 5), dtype=bool)
        mask[1, 1] = True
        mask[6, 4] = True

        packed = (mask.shape, np.packbits(mask).tobytes())
        restored = PixelArtConverter.unpack_selection(packed)

        np.testing.assert_array_equal(restored, mask)

    def test_packing_is_one_bit_per_pixel(self):
        mask = np.ones((64, 64), dtype=bool)

        packed = np.packbits(mask).tobytes()

        self.assertEqual(len(packed), 64 * 64 // 8)

    def test_an_absent_mask_stays_absent(self):
        self.assertIsNone(PixelArtConverter.unpack_selection(None))


class OuterBorderMarginTests(unittest.TestCase):
    def test_margin_is_reported_when_content_touches_the_edge(self):
        image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        image.putpixel((0, 1), (10, 20, 30, 255))

        self.assertTrue(Pipeline.outer_border_needs_margin(image))

    def test_margin_is_not_reported_for_inset_content(self):
        image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        image.putpixel((1, 1), (10, 20, 30, 255))

        self.assertFalse(Pipeline.outer_border_needs_margin(image))

    def test_add_outer_border_flags_a_clipped_border(self):
        image = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
        app = pipeline(border=True, border_width=1)

        app.add_outer_border(image)

        self.assertTrue(app.outer_border_clipped)

    def test_add_outer_border_clears_the_flag_when_margin_exists(self):
        image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        image.putpixel((2, 2), (10, 20, 30, 255))
        app = pipeline(border=True, border_width=1)

        app.add_outer_border(image)

        self.assertFalse(app.outer_border_clipped)


class PipelineTests(unittest.TestCase):
    def test_small_region_does_not_create_an_internal_outline(self):
        image = Image.fromarray(
            np.array(
                [
                    [[255, 0, 0, 255], [0, 0, 255, 255], [0, 0, 255, 255]],
                    [[0, 0, 255, 255], [0, 0, 255, 255], [0, 0, 255, 255]],
                ],
                dtype=np.uint8,
            ),
            "RGBA",
        )
        app = pipeline(object_borders=True, object_min_area=2)

        result = np.array(app.add_object_borders(image))

        np.testing.assert_array_equal(result, np.array(image))

    def test_outer_border_preserves_original_pixels_and_dimensions(self):
        image = Image.new("RGBA", (5, 5), (0, 0, 0, 0))
        image.putpixel((2, 2), (10, 20, 30, 128))
        app = pipeline(border=True, border_width=1, border_color="#010203")

        result = app.add_outer_border(image)

        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((2, 2)), (10, 20, 30, 128))
        self.assertEqual(result.getpixel((1, 2)), (1, 2, 3, 255))

    def test_outer_border_honors_requested_width(self):
        image = Image.new("RGBA", (7, 7), (0, 0, 0, 0))
        image.putpixel((3, 3), (10, 20, 30, 255))
        app = pipeline(border=True, border_width=2, border_color="#010203")

        result = app.add_outer_border(image)

        self.assertEqual(result.getpixel((1, 3)), (1, 2, 3, 255))
        self.assertEqual(result.getpixel((5, 3)), (1, 2, 3, 255))
        self.assertEqual(result.getpixel((0, 3)), (0, 0, 0, 0))

    def test_mutating_stages_accept_pillow_images(self):
        image = Image.new("RGBA", (4, 4), (30, 60, 90, 255))
        image.putpixel((0, 0), (0, 0, 0, 0))
        app = pipeline(
            border=True,
            object_borders=True,
            invert=True,
            monochrome="Black",
        )

        result = app.reconstruct_pixel_art(image)
        result = app.apply_color_effects(result)
        self.assertEqual(result.getpixel((0, 0))[3], 0)
        result = app.add_outer_border(result)
        result = app.add_object_borders(result)

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, image.size)

    def test_process_keeps_requested_dimensions_and_alpha_when_enabled(self):
        image = Image.new("RGBA", (6, 3), (20, 40, 60, 255))
        image.putpixel((0, 0), (10, 20, 30, 0))
        app = pipeline(
            crop=False,
            progressive=False,
            mode="Pixel Perfect Resize",
            fit_mode="exact",
            width=8,
            height=4,
            alpha=True,
        )

        result = app.run(image)

        self.assertEqual(result.size, (8, 4))
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getchannel("A").getextrema(), (0, 255))


if __name__ == "__main__":
    unittest.main()
