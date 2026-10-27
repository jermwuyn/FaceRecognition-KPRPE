"""Inspect exactly 20 seeded LFW samples using CPU YuNet; never change inputs."""
import csv
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
ORDER = (1, 0, 2, 4, 3)
NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
# Anatomical sides (subject's perspective), as documented by OpenCV.
SOURCE = "https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html"
COLORS = ((0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 220, 220), (220, 0, 220))


def samples():
    rng = random.Random(42)
    result = []
    for split, count in (("train", 14), ("validation", 3), ("test", 3)):
        with (ROOT / f"data/lfw/manifests/{split}.csv").open(newline="", encoding="utf-8") as stream:
            rows = sorted(csv.DictReader(stream), key=lambda row: row["image_path"])
        result.extend(rng.sample(rows, count))
    return result


def convert_landmarks(face):
    face = np.asarray(face)
    if face.shape != (15,) or not np.isfinite(face).all():
        raise ValueError("Expected 15 finite YuNet values")
    return face[4:14].reshape(5, 2)[list(ORDER)].copy()


def valid_points(points, width=250, height=250):
    points = np.asarray(points)
    return bool(points.shape == (5, 2) and np.isfinite(points).all()
                and (points[:, 0] >= 0).all() and (points[:, 0] < width).all()
                and (points[:, 1] >= 0).all() and (points[:, 1] < height).all())


def select_face(faces):
    # Largest bbox; confidence breaks ties, original output index breaks exact ties.
    return max(range(len(faces)), key=lambda i: (float(faces[i][2]) * float(faces[i][3]), float(faces[i][14]), -i))


def protected_hashes(selected):
    paths = [ROOT / "README.md", *(ROOT / f"data/lfw/manifests/{name}" for name in
             ("train.csv", "validation.csv", "test.csv", "train_label_mapping.csv", "split_summary.json")),
             *(ROOT / row["image_path"] for row in selected)]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    model = ROOT / "models/yunet/face_detection_yunet_2026may.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Model SHA-256 mismatch; stopping")
    output = ROOT / "outputs/yunet_lfw20"
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing run")
    selected = samples()
    before = protected_hashes(selected)
    start = time.perf_counter()
    detector = cv2.FaceDetectorYN.create(str(model), "", (250, 250), 0.9, 0.3, 5000,
                                       cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    initialization_ms = (time.perf_counter() - start) * 1000
    output.mkdir(parents=True)
    results, panels = [], []
    for index, row in enumerate(selected):
        path = ROOT / row["image_path"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (250, 250, 3):
            raise ValueError(f"Unexpected image size: {path}")
        tick = time.perf_counter()
        _, faces = detector.detect(image)
        elapsed = (time.perf_counter() - tick) * 1000
        faces = np.empty((0, 15), dtype=np.float32) if faces is None else faces
        record = {"sample_index": index, "image_path": row["image_path"], "split": row["split"],
                  "image_shape": list(image.shape), "input_sha256": before[row["image_path"]],
                  "detection_ms": elapsed, "face_count": len(faces), "raw_detections": faces.tolist(),
                  "selected_index": None, "confidence": None, "landmarks": None, "status": "no_detection",
                  "warnings": []}
        annotated = image.copy()
        if len(faces):
            chosen = select_face(faces)
            face = faces[chosen]
            points = convert_landmarks(face)
            record.update(selected_index=chosen, confidence=float(face[14]), landmarks=points.tolist(),
                          status="ok" if valid_points(points) else "invalid_coordinates")
            if len(faces) > 1:
                record["warnings"].append("multiple_faces")
            # Anatomical right usually lies to the viewer's left; flag only, never reorder by x.
            if not (points[0, 0] > points[1, 0] and points[3, 0] > points[4, 0]):
                record["warnings"].append("unusual_anatomical_side_geometry")
            x, y, w, h = face[:4]
            if x < 0 or y < 0 or x + w > 250 or y + h > 250:
                record["warnings"].append("bbox_exceeds_image")
            cv2.rectangle(annotated, (round(float(x)), round(float(y))),
                          (round(float(x+w)), round(float(y+h))), (0, 180, 0), 1)
            for point_id, (point, color) in enumerate(zip(points, COLORS), start=1):
                if np.isfinite(point).all():
                    p = tuple(round(float(v)) for v in point)
                    cv2.circle(annotated, p, 3, color, -1)
                    cv2.putText(annotated, str(point_id), (p[0]+4, p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1)
        # Preserve 250x250 image area; captions occupy a separate panel below it.
        panel = np.full((365, 250, 3), 255, dtype=np.uint8)
        panel[:250] = annotated
        lines = [f"{index+1:02d} {row['split']} {record['status']}", path.stem,
                 f"faces={len(faces)} score={record['confidence']:.4f}" if len(faces) else "faces=0; no landmarks",
                 "1 left eye; 2 right eye; 3 nose", "4 left mouth; 5 right mouth", "Left/right = subject's anatomy"]
        for line_id, line in enumerate(lines):
            cv2.putText(panel, line, (4, 265+line_id*17), cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 0, 0), 1)
        filename = f"{index+1:02d}_{path.stem}.png"
        if not cv2.imwrite(str(output / filename), panel):
            raise OSError("Could not save visualization")
        record["visualization"] = filename
        results.append(record)
        panels.append(panel)
        print(f"{index+1:02d} {path.name}: {record['status']} faces={len(faces)} score={record['confidence']} {elapsed:.2f}ms", flush=True)
    if protected_hashes(selected) != before:
        raise RuntimeError("Protected input changed")
    contact = np.vstack([np.hstack(panels[i:i+5]) for i in range(0,20,5)])
    if not cv2.imwrite(str(output / "contact_sheet.png"), contact):
        raise OSError("Could not save contact sheet")
    times = [r["detection_ms"] for r in results]
    scores = [r["confidence"] for r in results if r["confidence"] is not None]
    report = {"opencv_version": cv2.__version__, "model_sha256": SHA256, "seed": 42,
              "backend": "DNN_BACKEND_OPENCV", "target": "DNN_TARGET_CPU", "score_threshold": .9,
              "nms_threshold": .3, "top_k": 5000, "initialization_ms": initialization_ms,
              "landmark_order": list(NAMES), "raw_order": ["right_eye","left_eye","nose","right_mouth","left_mouth"],
              "mapping": list(ORDER), "order_reference": SOURCE,
              "selection_rule": "largest bbox area, then highest confidence, then first output index",
              "processed": len(results), "detected": sum(r["face_count"] > 0 for r in results),
              "valid_landmarks": sum(r["status"] == "ok" for r in results),
              "confidence": {"min": min(scores), "max": max(scores), "mean": statistics.mean(scores)} if scores else None,
              "timing_ms": {"total": sum(times), "mean": statistics.mean(times), "median": statistics.median(times),
                            "min": min(times), "max": max(times), "first": times[0],
                            "note": "detect() only; no warm-up; excludes image IO, visualization and model creation"},
              "protected_input_hashes": before, "results": results}
    with (output / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("results","protected_input_hashes")},indent=2))


if __name__ == "__main__":
    main()
