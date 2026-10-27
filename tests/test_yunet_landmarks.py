"""Validate YuNet's public output contract and saved 20-image results.

These tests never run detector inference or modify dataset files. The synthetic
rows below test parsing only and are never saved as image annotations.
"""

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/yunet_lfw20"
spec = importlib.util.spec_from_file_location(
    "check_yunet_landmarks", ROOT / "scripts/check_yunet_landmarks.py"
)
yunet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yunet)

EXPECTED_NAMES = (
    "Edmund_Stoiber_0006.jpg", "Angelina_Jolie_0003.jpg",
    "Jesse_Jackson_0001.jpg", "Jacques_Chirac_0021.jpg",
    "Howard_Dean_0008.jpg", "George_W_Bush_0153.jpg",
    "Donald_Rumsfeld_0054.jpg", "David_Nalbandian_0001.jpg",
    "Ricardo_Lagos_0017.jpg", "Ariel_Sharon_0013.jpg",
    "Ari_Fleischer_0010.jpg", "Dick_Cheney_0011.jpg",
    "Heizo_Takenaka_0001.jpg", "Hugo_Chavez_0058.jpg",
    "Paradorn_Srichaphan_0005.jpg", "Susan_Sarandon_0003.jpg",
    "Benjamin_Netanyahu_0003.jpg", "Jeremy_Greenstock_0017.jpg",
    "Colin_Powell_0055.jpg", "Paul_Burrell_0010.jpg",
)
EXPECTED_HASH = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


class YuNetContractTests(unittest.TestCase):
    def test_official_landmark_offsets(self):
        # OpenCV output: bbox, right eye, left eye, nose, right mouth,
        # left mouth, confidence. Left/right are the subject's anatomy.
        raw = [0, 0, 240, 240, 80, 90, 160, 91, 120, 130,
               85, 170, 155, 171, 0.98]
        expected = [[160, 91], [80, 90], [120, 130], [155, 171], [85, 170]]
        self.assertEqual(tuple(yunet.ORDER), (1, 0, 2, 4, 3))
        self.assertEqual(tuple(yunet.NAMES),
                         ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth"))
        np.testing.assert_array_equal(yunet.convert_landmarks(raw), expected)

    def test_invalid_detector_rows_rejected(self):
        for raw in (np.zeros(14), np.zeros((1, 15)), np.full(15, np.nan),
                    np.full(15, np.inf)):
            with self.subTest(shape=raw.shape):
                with self.assertRaises(ValueError):
                    yunet.convert_landmarks(raw)

    def test_coordinate_boundaries_and_nonfinite_values(self):
        good = np.array([[0, 0], [249.99, 249.99], [80, 90], [110, 130], [120, 160]])
        self.assertTrue(yunet.valid_points(good))
        for value in (-0.01, 250, float("nan"), float("inf")):
            for column in (0, 1):
                bad = good.copy()
                bad[0, column] = value
                with self.subTest(value=value, column=column):
                    self.assertFalse(yunet.valid_points(bad))
        self.assertFalse(yunet.valid_points(np.zeros((4, 2))))
        self.assertFalse(yunet.valid_points(np.zeros(10)))

    def test_face_selection_area_confidence_and_stable_tie(self):
        rows = np.zeros((3, 15))
        rows[:, 2:4] = [[10, 10], [20, 20], [20, 20]]
        rows[:, 14] = [0.99, 0.90, 0.95]
        self.assertEqual(yunet.select_face(rows), 2)
        rows[1, 14] = 0.95
        self.assertEqual(yunet.select_face(rows), 1)

    def test_exact_original_seeded_selection(self):
        selected = yunet.samples()
        self.assertEqual([Path(row["image_path"]).name for row in selected], list(EXPECTED_NAMES))
        self.assertEqual(len({row["image_path"] for row in selected}), 20)
        self.assertEqual(Counter(row["split"] for row in selected),
                         {"train": 14, "validation": 3, "test": 3})
        self.assertEqual(selected, yunet.samples())


class YuNetSavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing results is an error: unit tests alone cannot validate a run.
        cls.report = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
        cls.results = cls.report["results"]

    def test_model_cpu_configuration_and_sample_coverage(self):
        model = ROOT / "models/yunet/face_detection_yunet_2026may.onnx"
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), EXPECTED_HASH)
        self.assertEqual(self.report["model_sha256"], EXPECTED_HASH)
        self.assertEqual(self.report["target"], "DNN_TARGET_CPU")
        self.assertEqual(self.report["backend"], "DNN_BACKEND_OPENCV")
        self.assertEqual(self.report["seed"], 42)
        self.assertEqual(self.report["mapping"], [1, 0, 2, 4, 3])
        selected = yunet.samples()
        self.assertEqual(len(self.results), 20)
        self.assertEqual([row["image_path"] for row in self.results],
                         [row["image_path"] for row in selected])
        self.assertEqual([row["split"] for row in self.results],
                         [row["split"] for row in selected])
        self.assertEqual([row["sample_index"] for row in self.results], list(range(20)))

    def test_actual_raw_output_mapping_and_valid_coordinates(self):
        for row in self.results:
            with self.subTest(image=row["image_path"]):
                raw = row["raw_detections"]
                self.assertEqual(len(raw), row["face_count"])
                if not raw:
                    self.assertEqual(row["status"], "no_detection")
                    self.assertIsNone(row["selected_index"])
                    self.assertIsNone(row["confidence"])
                    self.assertIsNone(row["landmarks"])
                    continue
                for face in raw:
                    self.assertEqual(len(face), 15)
                    self.assertTrue(all(math.isfinite(v) for v in face))
                    self.assertGreater(face[2], 0)
                    self.assertGreater(face[3], 0)
                face = raw[row["selected_index"]]
                # Independent, explicit offsets from the public API contract.
                expected = [face[6:8], face[4:6], face[8:10], face[12:14], face[10:12]]
                self.assertEqual(row["landmarks"], expected)
                self.assertEqual(row["confidence"], face[14])
                self.assertGreaterEqual(face[14], self.report["score_threshold"])
                self.assertLessEqual(face[14], 1)
                in_bounds = all(0 <= v < 250 for point in expected for v in point)
                self.assertEqual(row["status"], "ok" if in_bounds else "invalid_coordinates")
                if row["status"] == "ok":
                    self.assertTrue(yunet.valid_points(row["landmarks"]))

    def test_selected_face_policy_and_multiple_face_flag(self):
        for row in self.results:
            raw = row["raw_detections"]
            if raw:
                best = sorted(range(len(raw)),
                              key=lambda i: (-raw[i][2] * raw[i][3], -raw[i][14], i))[0]
                self.assertEqual(row["selected_index"], best)
            self.assertEqual("multiple_faces" in row["warnings"], len(raw) > 1)

    def test_original_size_visualizations_and_protected_hashes(self):
        expected_protected = {"README.md", *(
            f"data/lfw/manifests/{name}" for name in (
                "train.csv", "validation.csv", "test.csv",
                "train_label_mapping.csv", "split_summary.json")),
            *(row["image_path"] for row in self.results)}
        self.assertEqual(set(self.report["protected_input_hashes"]), expected_protected)
        for relative, digest in self.report["protected_input_hashes"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)
        for row in self.results:
            with self.subTest(image=row["image_path"]):
                with Image.open(ROOT / row["image_path"]) as source:
                    self.assertEqual(source.size, (250, 250))
                    self.assertEqual(source.mode, "RGB")
                self.assertEqual(row["image_shape"], [250, 250, 3])
                self.assertEqual(row["input_sha256"],
                                 self.report["protected_input_hashes"][row["image_path"]])
                with Image.open(OUTPUT / row["visualization"]) as rendered:
                    self.assertEqual(rendered.size, (250, 365))
                    rendered.verify()
        with Image.open(OUTPUT / "contact_sheet.png") as sheet:
            self.assertEqual(sheet.size, (1250, 1460))
            sheet.verify()

    def test_report_aggregates_match_actual_results(self):
        self.assertEqual(self.report["processed"], len(self.results))
        self.assertEqual(self.report["detected"], sum(row["face_count"] > 0 for row in self.results))
        self.assertEqual(self.report["valid_landmarks"], sum(row["status"] == "ok" for row in self.results))
        times = [row["detection_ms"] for row in self.results]
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in times))
        expected_timing = {"total": sum(times), "mean": statistics.mean(times),
                           "median": statistics.median(times), "min": min(times),
                           "max": max(times), "first": times[0]}
        for name, value in expected_timing.items():
            self.assertAlmostEqual(self.report["timing_ms"][name], value)
        scores = [row["confidence"] for row in self.results if row["confidence"] is not None]
        if scores:
            for name, value in {"min": min(scores), "max": max(scores),
                                "mean": statistics.mean(scores)}.items():
                self.assertAlmostEqual(self.report["confidence"][name], value)
        else:
            self.assertIsNone(self.report["confidence"])


if __name__ == "__main__":
    unittest.main()
