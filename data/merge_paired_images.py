#!/usr/bin/env python3
"""Move paired images and edge maps into the data-level directories.

The two files in each pair receive the same numeric stem, while preserving
independent suffixes such as .jpg for images and .png for edge maps.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_GROUPS = (ROOT / "top", ROOT / "bottom")
TARGET_IMAGES = ROOT / "images"
TARGET_EDGE_MAPS = ROOT / "edge_maps"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NUMBERED_STEM = re.compile(r"^(\d+)$")


def files_by_stem(directory: Path) -> dict[str, Path]:
    """Return image files indexed by their filename stem."""
    if not directory.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise RuntimeError(f"同一目录存在重复主文件名，无法安全配对: {path}")
        result[path.stem] = path
    return result


def next_number() -> int:
    """Find the first number after every existing numeric target stem."""
    numbers = []
    for directory in (TARGET_IMAGES, TARGET_EDGE_MAPS):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file():
                match = NUMBERED_STEM.fullmatch(path.stem)
                if match:
                    numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def collect_pairs() -> tuple[list[tuple[Path, Path, str]], list[Path]]:
    pairs: list[tuple[Path, Path, str]] = []
    unmatched: list[Path] = []

    for group in SOURCE_GROUPS:
        image_files = files_by_stem(group / "images")
        edge_files = files_by_stem(group / "edge_maps")
        for stem in sorted(image_files.keys() - edge_files.keys()):
            unmatched.append(image_files[stem])
        for stem in sorted(edge_files.keys() - image_files.keys()):
            unmatched.append(edge_files[stem])
        for stem in sorted(image_files.keys() & edge_files.keys()):
            pairs.append((image_files[stem], edge_files[stem], group.name))

    return pairs, unmatched


def build_moves(pairs: list[tuple[Path, Path, str]]) -> list[tuple[Path, Path, Path, Path]]:
    number = next_number()
    moves = []
    reserved: set[Path] = set()
    for image_source, edge_source, _group in pairs:
        while True:
            stem = f"{number:04d}"
            image_target = TARGET_IMAGES / f"{stem}{image_source.suffix.lower()}"
            edge_target = TARGET_EDGE_MAPS / f"{stem}{edge_source.suffix.lower()}"
            number += 1
            if (
                image_target not in reserved
                and edge_target not in reserved
                and not image_target.exists()
                and not edge_target.exists()
            ):
                reserved.update((image_target, edge_target))
                moves.append((image_source, image_target, edge_source, edge_target))
                break
    return moves


def main() -> int:
    parser = argparse.ArgumentParser(description="移动并重新编号配对的 images 和 edge_maps")
    parser.add_argument("--dry-run", action="store_true", help="仅显示操作，不实际移动文件")
    args = parser.parse_args()

    TARGET_IMAGES.mkdir(parents=True, exist_ok=True)
    TARGET_EDGE_MAPS.mkdir(parents=True, exist_ok=True)

    try:
        pairs, unmatched = collect_pairs()
        moves = build_moves(pairs)
    except RuntimeError as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1

    print(f"发现 {len(pairs)} 对可移动数据。")
    if unmatched:
        print(f"发现 {len(unmatched)} 个未配对文件，已保留在原目录:")
        for path in unmatched:
            print(f"  {path}")

    for image_source, image_target, edge_source, edge_target in moves:
        print(f"{image_source} -> {image_target}")
        print(f"{edge_source} -> {edge_target}")

    if args.dry_run:
        print("预览完成，未移动文件。")
        return 0

    moved = 0
    for image_source, image_target, edge_source, edge_target in moves:
        # Both destinations were checked before moving either member of a pair.
        shutil.move(str(image_source), str(image_target))
        try:
            shutil.move(str(edge_source), str(edge_target))
        except Exception:
            # Keep the already moved image visible in the error; never overwrite it.
            print(
                f"错误: 边缘图移动失败，配对可能暂时不完整: {edge_source} -> {edge_target}",
                file=sys.stderr,
            )
            return 1
        moved += 1

    print(f"已移动 {moved} 对数据。目标目录中的每对文件主文件名一致，扩展名保持原样。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
