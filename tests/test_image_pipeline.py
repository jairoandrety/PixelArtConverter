import unittest
from dataclasses import fields

import numpy as np
from PIL import Image

from pixel_art_converter import PixelArtConverter
from pixel_pipeline import Pipeline, Settings


def pipeline(**overrides):
    """A pipeline for one run; no Tkinter and no stand-in variables."""
    return Pipeline(Settings(**overrides))


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
