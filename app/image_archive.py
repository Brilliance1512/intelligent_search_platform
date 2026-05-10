import io
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

SUPPORTED_DATASET_EXTENSIONS = {".csv", ".json", ".jsonl", ".xlsx", ".xls"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SUPPORTED_ARCHIVE_EXTENSIONS = {".zip"}


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text.split(".", 1)[0] if re.fullmatch(r"-?\d+\.0+", text) else text


def extract_zip_files(data: bytes, target_dir: Path, allowed_extensions: set[str]) -> list[Path]:
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename).name
            if info.is_dir() or not name or name.startswith("."):
                continue
            if Path(name).suffix.lower() not in allowed_extensions:
                continue
            path = target_dir / name
            with archive.open(info) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            files.append(path)
    return files


def find_single_dataset_file(paths: list[Path]) -> Path:
    datasets = [p for p in paths if p.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS]
    if len(datasets) != 1:
        raise ValueError("ZIP must contain exactly one dataset file")
    return datasets[0]


def resolve_image_paths_by_id(df: pd.DataFrame, images_dir: Path, id_col: str = "id") -> list[tuple[int, str]]:
    images = {normalize_id(p.stem): p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS}
    return [(int(idx), str(images[normalize_id(row_id)])) for idx, row_id in df[id_col].items() if normalize_id(row_id) in images]
