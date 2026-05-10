import hashlib
import io
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.analyzer import analyze, load_dataframe
from app.config import CAT_THRESHOLD, CLIP_MODEL, CLIP_PRETRAINED, HYBRID_ALPHA, LANGUAGE, MAX_RESULTS, STORAGE_DIR
from app.enricher import enrich_descriptions
from app.filter_engine import apply_filters
from app.image_archive import SUPPORTED_ARCHIVE_EXTENSIONS, SUPPORTED_DATASET_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS, extract_zip_files, find_single_dataset_file, resolve_image_paths_by_id
from app.image_search import ImageSearchEngine
from app.llm_client import chat
from app.redis_store import RedisStore
from app.schema_compiler import build_messages, compile_schema, parse_response


def dataset_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_dir(tenant_id: str, dataset_id: str) -> Path:
    return STORAGE_DIR / "tenants" / tenant_id / "datasets" / dataset_id


def rows(df: pd.DataFrame, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    out = []
    for idx, row in df.head(limit).iterrows():
        item = {"_row_index": int(idx)}
        for col, value in row.items():
            item[col] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
        out.append(item)
    return out


@dataclass
class DatasetRuntime:
    tenant_id: str
    dataset_id: str
    source_path: Path
    dataset_hash: str
    df: pd.DataFrame
    config: dict
    schema: str
    images_dir: Path | None
    index_path: Path
    clip: ImageSearchEngine | None = None

    def get_clip(self) -> ImageSearchEngine:
        if self.clip is None:
            self.clip = ImageSearchEngine(CLIP_MODEL, CLIP_PRETRAINED)
            self.clip.load_index(self.index_path)
        return self.clip


class DatasetService:
    def __init__(self, store: RedisStore | None = None):
        self.store = store or RedisStore()
        self.runtimes: dict[tuple[str, str], DatasetRuntime] = {}
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def register_dataset(self, tenant_id: str, dataset_id: str, filename: str, data: bytes, language: str = LANGUAGE, build_image_index: bool = False, images_archive_filename: str | None = None, images_archive_data: bytes | None = None) -> dict:
        self.store.register_tenant(tenant_id)
        ds_dir = dataset_dir(tenant_id, dataset_id)
        shutil.rmtree(ds_dir, ignore_errors=True)
        ds_dir.mkdir(parents=True)
        images_dir = ds_dir / "images"

        source_path, source_bytes, source_name, package_images = self.save_dataset(ds_dir, filename, data)
        if images_archive_data:
            extract_zip_files(images_archive_data, images_dir, SUPPORTED_IMAGE_EXTENSIONS)
        elif package_images:
            images_dir.mkdir(exist_ok=True)
            for p in package_images:
                shutil.copy2(p, images_dir / p.name)

        digest = dataset_hash(source_bytes)
        df = load_dataframe(str(source_path))
        cached = self.store.get_schema(tenant_id, digest)
        if cached:
            config, schema, cache_hit = cached.config, cached.schema, True
        else:
            config = enrich_descriptions(analyze(df, source_name, CAT_THRESHOLD), language)
            schema = compile_schema(config)
            self.store.set_schema(tenant_id, digest, config, schema)
            cache_hit = False

        index_path = ds_dir / "clip_index.npz"
        images_indexed = self.build_index(df, images_dir, index_path) if build_image_index and images_dir.exists() else 0
        runtime = DatasetRuntime(tenant_id, dataset_id, source_path, digest, df, config, schema, images_dir if images_dir.exists() else None, index_path)
        self.runtimes[(tenant_id, dataset_id)] = runtime

        meta = {
            "tenant_id": tenant_id,
            "dataset_id": dataset_id,
            "filename": source_name,
            "source_path": str(source_path),
            "dataset_hash": digest,
            "rows": len(df),
            "columns": list(df.columns),
            "image_id_col": "id" if "id" in df.columns else None,
            "images_dir": str(runtime.images_dir) if runtime.images_dir else None,
            "image_match_mode": "id_archive" if images_indexed else None,
            "index_path": str(index_path),
            "schema_cache_hit": cache_hit,
            "images_indexed": images_indexed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.store.set_dataset_meta(tenant_id, dataset_id, meta)
        return {**meta, "filterable_fields": self.filterable_fields(config), "schema": schema}

    def save_dataset(self, ds_dir: Path, filename: str, data: bytes) -> tuple[Path, bytes, str, list[Path]]:
        ext = Path(filename).suffix.lower()
        if ext in SUPPORTED_DATASET_EXTENSIONS:
            path = ds_dir / f"source{ext}"
            path.write_bytes(data)
            return path, data, filename, []
        if ext not in SUPPORTED_ARCHIVE_EXTENSIONS:
            raise ValueError("unsupported dataset format")
        package_dir = ds_dir / "package"
        files = extract_zip_files(data, package_dir, SUPPORTED_DATASET_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS)
        dataset_file = find_single_dataset_file(files)
        source_path = ds_dir / f"source{dataset_file.suffix.lower()}"
        source_bytes = dataset_file.read_bytes()
        source_path.write_bytes(source_bytes)
        images = [p for p in files if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
        return source_path, source_bytes, dataset_file.name, images

    def build_index(self, df: pd.DataFrame, images_dir: Path, index_path: Path) -> int:
        pairs = resolve_image_paths_by_id(df, images_dir, "id")
        ids, paths = zip(*pairs)
        clip = ImageSearchEngine(CLIP_MODEL, CLIP_PRETRAINED)
        count = clip.build_index(list(paths), list(ids))
        clip.save_index(index_path)
        return count

    def register_images_archive(self, tenant_id: str, dataset_id: str, filename: str, data: bytes) -> dict:
        runtime = self.get_runtime(tenant_id, dataset_id)
        images_dir = dataset_dir(tenant_id, dataset_id) / "images"
        extracted = extract_zip_files(data, images_dir, SUPPORTED_IMAGE_EXTENSIONS)
        images_indexed = self.build_index(runtime.df, images_dir, runtime.index_path)
        runtime.images_dir = images_dir
        runtime.clip = None
        meta = self.store.get_dataset_meta(tenant_id, dataset_id)
        meta.update({"images_dir": str(images_dir), "image_id_col": "id", "image_match_mode": "id_archive", "images_indexed": images_indexed, "updated_at": datetime.now(timezone.utc).isoformat()})
        self.store.set_dataset_meta(tenant_id, dataset_id, meta)
        return {"tenant_id": tenant_id, "dataset_id": dataset_id, "images_extracted": len(extracted), "images_indexed": images_indexed, "image_id_col": "id", "image_match_mode": "id_archive"}

    def build_existing_image_index(self, tenant_id: str, dataset_id: str) -> dict:
        runtime = self.get_runtime(tenant_id, dataset_id)
        images_indexed = self.build_index(runtime.df, runtime.images_dir, runtime.index_path)
        runtime.clip = None
        meta = self.store.get_dataset_meta(tenant_id, dataset_id)
        meta.update({"images_indexed": images_indexed, "image_match_mode": "id_archive", "updated_at": datetime.now(timezone.utc).isoformat()})
        self.store.set_dataset_meta(tenant_id, dataset_id, meta)
        return {"tenant_id": tenant_id, "dataset_id": dataset_id, "images_indexed": images_indexed, "image_id_col": "id", "image_match_mode": "id_archive"}

    def get_runtime(self, tenant_id: str, dataset_id: str) -> DatasetRuntime:
        key = (tenant_id, dataset_id)
        if key in self.runtimes:
            return self.runtimes[key]
        meta = self.store.get_dataset_meta(tenant_id, dataset_id)
        df = load_dataframe(meta["source_path"])
        cached = self.store.get_schema(tenant_id, meta["dataset_hash"])
        runtime = DatasetRuntime(tenant_id, dataset_id, Path(meta["source_path"]), meta["dataset_hash"], df, cached.config, cached.schema, Path(meta["images_dir"]) if meta.get("images_dir") else None, Path(meta["index_path"]))
        self.runtimes[key] = runtime
        return runtime

    def delete_dataset(self, tenant_id: str, dataset_id: str) -> None:
        self.runtimes.pop((tenant_id, dataset_id), None)
        self.store.delete_dataset_meta(tenant_id, dataset_id)
        shutil.rmtree(dataset_dir(tenant_id, dataset_id), ignore_errors=True)

    def search_text(self, tenant_id: str, dataset_id: str, dialog_id: str, query: str, limit: int = MAX_RESULTS, language: str = LANGUAGE) -> dict:
        runtime = self.get_runtime(tenant_id, dataset_id)
        message_id = self.store.next_message_id(tenant_id)
        history = self.store.get_dialog_history(tenant_id, dialog_id)
        result = parse_response(chat(build_messages(runtime.schema, query, language, history)))
        filtered = apply_filters(runtime.df, result, limit)
        response = {"tenant_id": tenant_id, "dataset_id": dataset_id, "dialog_id": dialog_id, "message_id": message_id, "mode": "text", "query": query, "filters": result.get("filters", []), "count": len(filtered), "items": rows(filtered, limit)}
        self.store.append_dialog_event(tenant_id, dialog_id, {"dataset_id": dataset_id, "mode": "text", "query": query, "filters": response["filters"], "count": response["count"]}, message_id)
        return response

    def search_image(self, tenant_id: str, dataset_id: str, dialog_id: str, image: bytes | io.BytesIO, text: str | None = None, limit: int = MAX_RESULTS, alpha: float = HYBRID_ALPHA) -> dict:
        runtime = self.get_runtime(tenant_id, dataset_id)
        message_id = self.store.next_message_id(tenant_id)
        results = runtime.get_clip().search_hybrid(image, text, limit, alpha) if text else runtime.get_clip().search_by_image(image, limit)
        items = []
        for row_idx, score in results:
            item = rows(runtime.df.loc[[row_idx]], 1)[0]
            item["_score"] = score
            items.append(item)
        response = {"tenant_id": tenant_id, "dataset_id": dataset_id, "dialog_id": dialog_id, "message_id": message_id, "mode": "hybrid" if text else "image", "query": text, "count": len(items), "items": items}
        self.store.append_dialog_event(tenant_id, dialog_id, {"dataset_id": dataset_id, "mode": response["mode"], "query": text, "count": response["count"]}, message_id)
        return response

    @staticmethod
    def filterable_fields(config: dict) -> list[dict]:
        return [{"name": n, "type": i["type"], "description": i.get("description", "")} for n, i in config["fields"].items() if i["filterable"]]
