"""Extract CPU YuNet annotations for the existing LFW split, without editing inputs.

Run from the workspace: python -B -m scripts.extract_lfw_landmarks
Use --output with a new directory inside this workspace to reproduce a run.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import logging
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

from scripts.check_yunet_landmarks import SHA256, ORDER, NAMES, convert_landmarks, valid_points, select_face
from scripts.prepare_lfw_manifest import read_manifests, validate

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"train": 4161, "validation": 836, "test": 988}
COORD_FIELDS = tuple(f"{name}_{axis}" for name in NAMES for axis in ("x", "y"))
FIELDS = ("image_path", "identity_id", "identity_name", "split", "train_label",
          "image_sha256", "width", "height", "status", "detection_count",
          "selected_detection_index", "confidence", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
          *COORD_FIELDS, "landmarks_available", "detection_ms", "warnings", "error")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def analyze_detections(faces, width=250, height=250):
    result = {"status": "no_detection", "detection_count": 0,
              "selected_detection_index": None, "confidence": None, "bbox": None,
              "landmarks": None, "warnings": [], "error": ""}
    if faces is None:
        return result
    raw = np.asarray(faces)
    if raw.shape == (0, 15):
        return result
    if raw.ndim != 2 or raw.shape[1] != 15 or not np.issubdtype(raw.dtype, np.number):
        result.update(status="invalid_detector_output", error=f"Unexpected detection shape/dtype: {raw.shape}/{raw.dtype}")
        return result
    result["detection_count"] = len(raw)
    if len(raw) > 1:
        result["warnings"].append("multiple_faces")
    if not np.isfinite(raw[:, :4]).all() or (raw[:, 2:4] <= 0).any():
        result.update(status="invalid_detection", error="Nonfinite or nonpositive bbox; cannot rank areas")
        return result
    if not np.isfinite(raw[:, 14]).all() or ((raw[:, 14] < 0) | (raw[:, 14] > 1)).any():
        result.update(status="invalid_detection", error="Invalid confidence; cannot rank ties")
        return result
    index = select_face(raw)
    face = raw[index]
    result.update(selected_detection_index=index, confidence=float(face[14]), bbox=face[:4].tolist())
    if not np.isfinite(face[4:14]).all():
        result.update(status="invalid_coordinates", error="Nonfinite selected landmarks")
        return result
    points = convert_landmarks(face)
    if not valid_points(points, width, height):
        result.update(status="invalid_coordinates", error="Selected landmarks outside original image bounds")
        return result
    result.update(status="ok", landmarks=points.tolist())
    if not (points[0, 0] > points[1, 0] and points[3, 0] > points[4, 0]):
        result["warnings"].append("unusual_anatomical_side_geometry")
    x, y, w, h = face[:4]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        result["warnings"].append("bbox_exceeds_image")
    return result


def duplicate_report(records):
    by_hash = defaultdict(list)
    for row in records:
        if row["image_sha256"]:
            by_hash[row["image_sha256"]].append({key: row[key] for key in ("image_path", "split", "identity_id")})
    groups = []
    for digest, members in sorted(by_hash.items()):
        if len(members) > 1:
            splits = sorted({m["split"] for m in members})
            groups.append({"sha256": digest, "members": members, "splits": splits,
                           "cross_split": len(splits) > 1,
                           "identity_conflict": len({m["identity_id"] for m in members}) > 1})
    pairs = {}
    for left, right in itertools.combinations(SPLITS, 2):
        shared = [g for g in groups if left in g["splits"] and right in g["splits"]]
        pairs[f"{left}__{right}"] = {
            "groups": len(shared),
            "image_pairs": sum(sum(m["split"] == left for m in g["members"]) *
                               sum(m["split"] == right for m in g["members"]) for g in shared),
        }
    within = {}
    for split in SPLITS:
        counts = [sum(m["split"] == split for m in g["members"]) for g in groups]
        counts = [n for n in counts if n > 1]
        within[split] = {"groups": len(counts), "images": sum(counts), "extra_copies": sum(n-1 for n in counts)}
    return {"method": "SHA-256 of original file bytes; exact binary duplicates only, not perceptual similarity",
            "hashed_images": sum(len(v) for v in by_hash.values()), "unique_hashes": len(by_hash),
            "duplicate_groups": len(groups), "duplicate_images": sum(len(g["members"]) for g in groups),
            "extra_copies": sum(len(g["members"])-1 for g in groups),
            "cross_split_groups": sum(g["cross_split"] for g in groups),
            "cross_split_images": sum(len(g["members"]) for g in groups if g["cross_split"]),
            "within_split": within, "split_pairs": pairs, "groups": groups}


def distribution(values):
    return {"min": min(values), "max": max(values), "mean": statistics.mean(values),
            "median": statistics.median(values)} if values else None


def summarize(records):
    counts = Counter(row["status"] for row in records)
    detected = sum(row["detection_count"] > 0 for row in records)
    success = counts["ok"]
    times = [row["detection_ms"] for row in records if row["detection_ms"] != ""]
    scores = [row["confidence"] for row in records if row["confidence"] != ""]
    valid_ids = {row["identity_id"] for row in records if row["status"] == "ok"}
    return {"images": len(records), "identities": len({r["identity_id"] for r in records}),
            "detected_images": detected, "detection_rate": detected/len(records),
            "success_images": success, "success_rate": success/len(records),
            "failed_images": len(records)-success, "status_counts": dict(counts),
            "identities_with_valid_landmarks": len(valid_ids),
            "multi_face_images": sum(row["detection_count"] > 1 for row in records),
            "warning_counts": dict(Counter(w for r in records for w in r["warnings"].split(";") if w)),
            "confidence": distribution(scores), "detection_ms": distribution(times),
            "total_detection_ms": sum(times)}


def json_safe(value):
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(output, summary, duplicates):
    lines = ["# Báo cáo trích xuất YuNet trên LFW", "",
             "Ảnh và cách chia danh tính giữ nguyên. Annotation được liên kết bằng image_path tương đối với workspace.", "",
             "| Tập | Ảnh | Có detection | Landmarks hợp lệ | Thành công | Thất bại | Nhiều mặt |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in (*SPLITS, "all"):
        s = summary["splits"][split] if split != "all" else summary["overall"]
        lines.append(f"| {split} | {s['images']} | {s['detected_images']} | {s['success_images']} | {s['success_rate']:.2%} | {s['failed_images']} | {s['multi_face_images']} |")
    lines += ["", "## Quy ước và phạm vi", "",
              "- Mapping [1, 0, 2, 4, 3]: mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải; trái/phải theo người trong ảnh.",
              "- Tọa độ float theo ảnh gốc 250×250, gốc tọa độ ở góc trên trái; x sang phải, y xuống dưới. Không crop, căn chỉnh, resize hoặc clip tọa độ.",
              "- Nhiều mặt: chọn bbox lớn nhất, sau đó confidence cao nhất, sau đó index nhỏ nhất. Đây là heuristic, không xác nhận danh tính.",
              "- Thất bại: cột landmarks để trống, landmarks_available=false; output thô chỉ giữ trong detections.jsonl để kiểm tra.",
              "- Ngưỡng confidence=0.9, NMS=0.3, top_k=5000. Không hạ ngưỡng hoặc thử lại ảnh thất bại.",
              "- Confidence là score detector, không phải độ chính xác landmarks. Không có ground truth để đo sai số định vị.",
              "", "## Trùng nội dung", "",
              f"SHA-256 trên byte file gốc: {duplicates['unique_hashes']} hash khác nhau; {duplicates['duplicate_groups']} nhóm trùng; {duplicates['cross_split_groups']} nhóm trùng giữa các tập.",
              "Không phát hiện ảnh gần giống hay ảnh được nén lại khác byte bằng phương pháp này. Không xóa hoặc chuyển ảnh.",
              "", "## Thời gian và môi trường", "",
              f"OpenCV {summary['opencv_version']}; CPU. Tổng detect(): {summary['overall']['total_detection_ms']/1000:.3f} giây; tổng pipeline: {summary['pipeline_seconds']:.3f} giây.",
              "Thời gian detect() không bao gồm đọc ảnh, hashing, ghi file; không warm-up. Xem summary.json cho thống kê chi tiết.",
              "OpenCV 5.0 có thể cảnh báo target chưa được hỗ trợ bởi graph engine mới; target mặc định vẫn là CPU. Không thay đổi môi trường.",
              "", "## Kiểm tra toàn vẹn và tái tạo", "",
              f"Hash kiểm tra lại sau xử lý: {summary['integrity']['images_checked']} ảnh; số file thay đổi: {len(summary['integrity']['changed_files'])}.",
              "Chạy từ workspace với Python của dự án; chỉ định thư mục mới nếu đầu ra đã tồn tại:", "",
              "```powershell", "& 'D:\\CondaEnvs\\ai\\python.exe' -B -m scripts.extract_lfw_landmarks --output data/yunet/annotations/yunet_2026may_repeat", "```", "",
              "Danh sách mọi ảnh lỗi/cảnh báo nằm trong failures.csv và processing.log; output thô và index mặt đã chọn nằm trong detections.jsonl.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/yunet/annotations/yunet_2026may"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("Output must be a directory inside this project")
    if output.exists():
        raise FileExistsError("Output exists; inspect it and choose a new --output directory")
    model = ROOT / "models/yunet/face_detection_yunet_2026may.onnx"
    if file_hash(model) != SHA256:
        raise ValueError("Model SHA-256 mismatch; stopping before inference")
    manifest_dir = ROOT / "data/lfw/manifests"
    manifests = read_manifests(manifest_dir)
    validate(ROOT, manifests)
    if {s: len(manifests[s]) for s in SPLITS} != EXPECTED_COUNTS:
        raise ValueError("Manifest counts differ from the approved 5985-image split")
    protected = [ROOT / "README.md", model, model.parent / "LICENSE", *sorted(manifest_dir.iterdir())]
    protected_before = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in protected if p.is_file()}
    pipeline_start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    detector = cv2.FaceDetectorYN.create(str(model), "", (250, 250), .9, .3, 5000,
                                       cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    initialization_ms = (time.perf_counter()-tick)*1000
    output.mkdir(parents=True)
    logger = logging.getLogger("lfw_landmarks")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "processing.log", mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("START model=%s sha256=%s CPU OpenCV=%s score=0.9 NMS=0.3", model.name, SHA256, cv2.__version__)
    all_records = []
    try:
        with (output / "detections.jsonl").open("x", encoding="utf-8") as raw_stream:
            for split in SPLITS:
                with (output / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    consecutive_detector_errors = 0
                    for source in manifests[split]:
                        record = {key: "" for key in FIELDS}
                        record.update({key: source[key] for key in ("image_path", "identity_id", "identity_name", "split", "train_label")})
                        record.update(landmarks_available="false", detection_count=0)
                        path = ROOT / source["image_path"]
                        raw = []
                        try:
                            contents = path.read_bytes()
                            record["image_sha256"] = hashlib.sha256(contents).hexdigest()
                            image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
                            del contents
                        except (OSError, cv2.error) as error:
                            image = None
                            record["error"] = str(error)
                        if image is None:
                            record.update(status="image_read_error")
                        else:
                            record.update(width=image.shape[1], height=image.shape[0])
                            if image.shape != (250, 250, 3):
                                record.update(status="invalid_image_size", error=str(image.shape))
                            else:
                                tick = time.perf_counter()
                                try:
                                    _, faces = detector.detect(image)
                                    record["detection_ms"] = (time.perf_counter()-tick)*1000
                                    raw = [] if faces is None else json_safe(faces.tolist())
                                    analysis = analyze_detections(faces)
                                    consecutive_detector_errors = 0
                                    for key in ("status", "detection_count", "selected_detection_index", "confidence", "error"):
                                        record[key] = "" if analysis[key] is None else analysis[key]
                                    record["warnings"] = ";".join(analysis["warnings"])
                                    if analysis["bbox"] is not None:
                                        record.update(zip(("bbox_x", "bbox_y", "bbox_width", "bbox_height"), analysis["bbox"]))
                                    if analysis["status"] == "ok":
                                        record.update(zip(COORD_FIELDS, itertools.chain.from_iterable(analysis["landmarks"])))
                                        record["landmarks_available"] = "true"
                                except cv2.error as error:
                                    record.update(status="detector_error", error=str(error), detection_ms=(time.perf_counter()-tick)*1000)
                                    consecutive_detector_errors += 1
                        writer.writerow(record)
                        raw_stream.write(json.dumps({"image_path": record["image_path"], "raw_detections": raw,
                                                     "selected_detection_index": record["selected_detection_index"],
                                                     "status": record["status"]}, allow_nan=False) + "\n")
                        all_records.append(record)
                        logger.info("%d split=%s path=%s status=%s faces=%s confidence=%s ms=%s warnings=%s error=%s",
                                    len(all_records), split, record["image_path"], record["status"], record["detection_count"],
                                    record["confidence"], record["detection_ms"], record["warnings"], record["error"])
                        if len(all_records) % 250 == 0:
                            print(f"Processed {len(all_records)}/5985; valid={sum(r['status']=='ok' for r in all_records)}", flush=True)
                        if consecutive_detector_errors >= 3:
                            raise RuntimeError("Three consecutive detector errors; stopped without modifying environment")
        duplicates = duplicate_report(all_records)
        changed = [p for p, digest in protected_before.items() if file_hash(ROOT / p) != digest]
        for row in all_records:
            if row["image_sha256"] and file_hash(ROOT / row["image_path"]) != row["image_sha256"]:
                changed.append(row["image_path"])
        summary = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "opencv_version": cv2.__version__, "python_version": sys.version,
                   "model_sha256": SHA256, "model_source": "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
                   "backend": "DNN_BACKEND_OPENCV", "target": "DNN_TARGET_CPU", "score_threshold": .9, "nms_threshold": .3,
                   "top_k": 5000, "mapping": list(ORDER), "landmark_order": list(NAMES),
                   "coordinate_system": "original 250x250 image pixels, top-left origin; subject anatomical left/right",
                   "initialization_ms": initialization_ms, "pipeline_seconds": time.perf_counter()-pipeline_start,
                   "splits": {s: summarize([r for r in all_records if r['split'] == s]) for s in SPLITS},
                   "overall": summarize(all_records), "duplicate_statistics": {k:v for k,v in duplicates.items() if k != "groups"},
                   "protected_input_hashes": protected_before,
                   "integrity": {"images_checked": sum(bool(r['image_sha256']) for r in all_records), "changed_files": changed}}
        for name, value in (("summary.json", summary), ("duplicates.json", duplicates)):
            with (output / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write("\n")
        with (output / "failures.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(r for r in all_records if r["status"] != "ok" or r["warnings"])
        write_report(output, summary, duplicates)
        logger.info("COMPLETE processed=%d successful=%d failed=%d cross_split_duplicate_groups=%d changed_files=%d",
                    len(all_records), summary["overall"]["success_images"], summary["overall"]["failed_images"],
                    duplicates["cross_split_groups"], len(changed))
        if changed:
            raise RuntimeError("Input integrity check failed; see summary.json")
        print(json.dumps({"splits": summary["splits"], "overall": summary["overall"],
                          "duplicates": summary["duplicate_statistics"], "pipeline_seconds": summary["pipeline_seconds"]}, indent=2), flush=True)
    except BaseException:
        logger.exception("ABORTED: outputs may be partial; do not use as a completed run")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    main()
