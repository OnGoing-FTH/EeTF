#!/usr/bin/env python3
"""Recursively build a complete 256x256 tiled edge dataset from Labelme data.

Every Labelme JSON below ``--src`` is paired with its source image. The edge
label is rasterized at the original image resolution, converted to a soft edge
map, and then image/label are padded and split with identical coordinates.
Output tile names are deterministically shuffled and numbered from 000001.

The source dataset is read-only. Generation uses a temporary directory and only
replaces ``--out`` after every output file has been written successfully.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SourcePair:
    pair_index: int
    image_path: Path
    annotation_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class TileRecord:
    pair_index: int
    tile_row: int
    tile_col: int
    top: int
    left: int
    valid_height: int
    valid_width: int


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_annotation(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read Labelme JSON {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("shapes", []), list):
        raise ValueError(f"Invalid Labelme annotation: {path}")
    return data


def annotation_image_path(annotation_path: Path, source_root: Path, annotation: dict[str, Any]) -> Path:
    """Resolve imagePath first, then fall back to a same-directory same-stem image."""
    candidates: list[Path] = []
    image_name = annotation.get("imagePath")
    if isinstance(image_name, str) and image_name.strip():
        candidates.extend((annotation_path.parent / image_name, source_root / image_name))
    for suffix in sorted(IMAGE_SUFFIXES):
        candidates.append(annotation_path.with_suffix(suffix))
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    raise FileNotFoundError(f"Image not found for annotation: {annotation_path}")


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    height, width = image.shape[:2]
    return width, height


def collect_source_pairs(source_root: Path) -> tuple[list[SourcePair], dict[str, Any]]:
    annotation_files = sorted(source_root.rglob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No Labelme JSON files found under {source_root}")

    pairs: list[SourcePair] = []
    paired_images: set[Path] = set()
    duplicate_images: list[str] = []
    for annotation_path in annotation_files:
        annotation = load_annotation(annotation_path)
        image_path = annotation_image_path(annotation_path, source_root, annotation)
        if image_path in paired_images:
            duplicate_images.append(str(image_path))
            continue
        paired_images.add(image_path)
        width, height = image_size(image_path)
        json_width, json_height = annotation.get("imageWidth"), annotation.get("imageHeight")
        if json_width not in (None, width) or json_height not in (None, height):
            raise ValueError(
                f"Annotation/image size mismatch: {annotation_path}; "
                f"JSON={json_width}x{json_height}, image={width}x{height}"
            )
        pairs.append(SourcePair(len(pairs), image_path, annotation_path, width, height))

    if duplicate_images:
        preview = ", ".join(duplicate_images[:5])
        raise ValueError(f"Multiple JSON files resolve to the same image: {preview}")

    all_images = {
        path.resolve() for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    orphan_images = sorted(str(path.relative_to(source_root)) for path in all_images - paired_images)
    report = {
        "json_count": len(annotation_files),
        "paired_source_count": len(pairs),
        "orphan_image_count": len(orphan_images),
        "orphan_images": orphan_images,
    }
    return pairs, report


def draw_labelme_edges(edge_map: np.ndarray, shapes: list[dict[str, Any]], line_width: int) -> Counter:
    """Rasterize supported Labelme shapes as outlines without filling interiors."""
    shape_counts: Counter = Counter()
    height, width = edge_map.shape
    for shape in shapes:
        shape_type = str(shape.get("shape_type", "polygon")).lower()
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        shape_counts[shape_type] += 1
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 1:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
        points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
        integer_points = np.rint(points).astype(np.int32)
        contour = integer_points.reshape(-1, 1, 2)

        if shape_type == "polygon" and len(points) >= 3:
            cv2.polylines(edge_map, [contour], True, 255, line_width, cv2.LINE_8)
        elif shape_type in {"line", "linestrip", "polyline"} and len(points) >= 2:
            cv2.polylines(edge_map, [contour], False, 255, line_width, cv2.LINE_8)
        elif shape_type == "rectangle" and len(points) >= 2:
            cv2.rectangle(edge_map, tuple(integer_points[0]), tuple(integer_points[1]), 255, line_width, cv2.LINE_8)
        elif shape_type == "circle" and len(points) >= 2:
            radius = max(1, int(round(np.linalg.norm(points[1] - points[0]))))
            cv2.circle(edge_map, tuple(integer_points[0]), radius, 255, line_width, cv2.LINE_8)
        elif shape_type == "point":
            cv2.circle(edge_map, tuple(integer_points[0]), max(1, line_width // 2), 255, -1, cv2.LINE_8)
        else:
            shape_counts[f"unsupported:{shape_type}"] += 1
    return shape_counts


def make_soft_edge_map(hard_edges: np.ndarray, sigma: float, radius: float) -> np.ndarray:
    """Convert binary outlines to a bounded Gaussian distance-field label."""
    if not np.any(hard_edges):
        return hard_edges.copy()
    distance = cv2.distanceTransform((hard_edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    soft = np.exp(-(distance * distance) / (2.0 * sigma * sigma))
    soft[distance > radius] = 0.0
    return np.rint(np.clip(soft, 0.0, 1.0) * 255.0).astype(np.uint8)


def build_tile_records(pairs: list[SourcePair], tile_size: int) -> list[TileRecord]:
    records: list[TileRecord] = []
    for pair in pairs:
        rows = math.ceil(pair.height / tile_size)
        cols = math.ceil(pair.width / tile_size)
        for row in range(rows):
            for col in range(cols):
                top, left = row * tile_size, col * tile_size
                records.append(
                    TileRecord(
                        pair_index=pair.pair_index,
                        tile_row=row,
                        tile_col=col,
                        top=top,
                        left=left,
                        valid_height=min(tile_size, pair.height - top),
                        valid_width=min(tile_size, pair.width - left),
                    )
                )
    return records


def prepare_output(output_root: Path, overwrite: bool) -> Path:
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
    staging = output_root.parent / f".{output_root.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    (staging / "edge_maps").mkdir(parents=True)
    return staging


def pad_to_tile(array: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = array.shape[:2]
    padded_height = math.ceil(height / tile_size) * tile_size
    padded_width = math.ceil(width / tile_size) * tile_size
    if array.ndim == 3:
        output = np.zeros((padded_height, padded_width, array.shape[2]), dtype=array.dtype)
    else:
        output = np.zeros((padded_height, padded_width), dtype=array.dtype)
    output[:height, :width] = array
    return output


def write_dataset(
    pairs: list[SourcePair],
    tile_records: list[TileRecord],
    assigned_names: dict[TileRecord, str],
    source_root: Path,
    staging: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_pair: dict[int, list[TileRecord]] = defaultdict(list)
    for record in tile_records:
        by_pair[record.pair_index].append(record)

    manifest_rows: list[dict[str, Any]] = []
    shape_counts: Counter = Counter()
    foreground_tiles = 0
    pairs_by_index = {pair.pair_index: pair for pair in pairs}

    for pair_index in sorted(by_pair):
        pair = pairs_by_index[pair_index]
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {pair.image_path}")
        annotation = load_annotation(pair.annotation_path)
        hard_edges = np.zeros((pair.height, pair.width), dtype=np.uint8)
        shape_counts.update(draw_labelme_edges(hard_edges, annotation.get("shapes", []), args.line_width))
        label = make_soft_edge_map(hard_edges, args.soft_sigma, args.soft_radius) if args.soft_labels else hard_edges
        padded_image = pad_to_tile(image, args.tile_size)
        padded_label = pad_to_tile(label, args.tile_size)

        for record in by_pair[pair_index]:
            name = assigned_names[record]
            top, left = record.top, record.left
            image_tile = padded_image[top:top + args.tile_size, left:left + args.tile_size]
            label_tile = padded_label[top:top + args.tile_size, left:left + args.tile_size]
            if image_tile.shape[:2] != (args.tile_size, args.tile_size) or label_tile.shape != (args.tile_size, args.tile_size):
                raise RuntimeError(f"Internal tile shape error for {pair.image_path}")
            image_rel = Path("images") / f"{name}.png"
            label_rel = Path("edge_maps") / f"{name}.png"
            if not cv2.imwrite(str(staging / image_rel), image_tile):
                raise IOError(f"Unable to write image tile: {staging / image_rel}")
            if not cv2.imwrite(str(staging / label_rel), label_tile):
                raise IOError(f"Unable to write label tile: {staging / label_rel}")
            has_edge = bool(np.any(label_tile))
            foreground_tiles += int(has_edge)
            padded_height, padded_width = padded_image.shape[:2]
            manifest_rows.append(
                {
                    "name": name,
                    "image": image_rel.as_posix(),
                    "label": label_rel.as_posix(),
                    "source_pair_id": f"source_{pair.pair_index + 1:06d}",
                    "source_image": pair.image_path.relative_to(source_root).as_posix(),
                    "source_annotation": pair.annotation_path.relative_to(source_root).as_posix(),
                    "source_width": pair.width,
                    "source_height": pair.height,
                    "padded_width": padded_width,
                    "padded_height": padded_height,
                    "tile_row": record.tile_row,
                    "tile_col": record.tile_col,
                    "tile_top": top,
                    "tile_left": left,
                    "valid_width": record.valid_width,
                    "valid_height": record.valid_height,
                    "has_edge": has_edge,
                }
            )

    manifest_rows.sort(key=lambda item: item["name"])
    manifest = {
        "format_version": 1,
        "dataset_type": "labelme_edge_tiles",
        "source_root": str(source_root),
        "tile_size": args.tile_size,
        "padding": "right_bottom_zero",
        "resize": False,
        "image_format": "png_bgr_uint8",
        "label_format": "soft_edge_png_uint8" if args.soft_labels else "binary_edge_png_uint8",
        "line_width": args.line_width,
        "soft_sigma": args.soft_sigma if args.soft_labels else None,
        "soft_radius": args.soft_radius if args.soft_labels else None,
        "shuffle_seed": args.seed,
        "numbering": "000001-based contiguous global tile numbering",
        "source_pair_count": len(pairs),
        "tile_count": len(manifest_rows),
        "foreground_tile_count": foreground_tiles,
        "background_tile_count": len(manifest_rows) - foreground_tiles,
        "shape_counts": dict(sorted(shape_counts.items())),
        "samples": manifest_rows,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (staging / "pairs.json").write_text(
        json.dumps([[row["image"], row["label"]] for row in manifest_rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a shuffled 256x256 edge dataset from recursive Labelme image/JSON pairs")
    parser.add_argument("--src", type=Path, default=Path("/home/fth/EdTF/DATA"), help="Recursive source root")
    parser.add_argument("--out", type=Path, required=True, help="New dataset output directory")
    parser.add_argument("--tile-size", type=int, default=256, help="Fixed image/label tile side")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic global tile-name shuffle seed")
    parser.add_argument("--line-width", type=int, default=1, help="Hard Labelme outline width before softening")
    parser.add_argument("--soft-labels", action=argparse.BooleanOptionalAction, default=True, help="Generate Gaussian soft-edge labels")
    parser.add_argument("--soft-sigma", type=float, default=0.5)
    parser.add_argument("--soft-radius", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace an existing output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect pairs and report planned tile counts")
    args = parser.parse_args()
    args.src = args.src.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not args.src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {args.src}")
    if args.tile_size != 256:
        raise ValueError("Current EeTF contract requires --tile-size 256")
    if args.line_width < 1 or args.soft_sigma <= 0 or args.soft_radius <= 0:
        raise ValueError("line width, soft sigma, and soft radius must be positive")
    if args.out == args.src or _is_relative_to(args.out, args.src):
        raise ValueError("--out must be outside --src to prevent recursive self-ingestion")
    return args


def main() -> None:
    args = parse_args()
    pairs, source_report = collect_source_pairs(args.src)
    tile_records = build_tile_records(pairs, args.tile_size)
    shuffled = list(tile_records)
    random.Random(args.seed).shuffle(shuffled)
    assigned_names = {record: f"{index:06d}" for index, record in enumerate(shuffled, start=1)}

    print(f"Source root: {args.src}")
    print(f"Paired source images: {len(pairs)}")
    print(f"Planned 256x256 tiles: {len(tile_records)}")
    print(f"Orphan images: {source_report['orphan_image_count']}")
    if source_report["orphan_images"]:
        for path in source_report["orphan_images"][:20]:
            print(f"  orphan: {path}")
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return

    staging = prepare_output(args.out, args.overwrite)
    try:
        manifest = write_dataset(pairs, tile_records, assigned_names, args.src, staging, args)
        summary = {
            **source_report,
            "tile_count": manifest["tile_count"],
            "foreground_tile_count": manifest["foreground_tile_count"],
            "background_tile_count": manifest["background_tile_count"],
            "output": str(args.out),
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.out.exists():
            shutil.rmtree(args.out)
        staging.rename(args.out)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"Generated dataset: {args.out}")
    print(f"Tiles: {manifest['tile_count']} (foreground={manifest['foreground_tile_count']}, background={manifest['background_tile_count']})")
    print(f"Images: {args.out / 'images'}")
    print(f"Labels: {args.out / 'edge_maps'}")
    print(f"Manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
