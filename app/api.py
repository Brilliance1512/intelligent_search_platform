from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import CLIP_BUILD_ON_UPLOAD_DEFAULT, HYBRID_ALPHA, LANGUAGE, MAX_RESULTS
from app.dataset_service import DatasetService
from app.redis_store import RedisStore

store = RedisStore()
service = DatasetService(store)
app = FastAPI(title="Universal Multimodal Search Platform", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class TextSearchRequest(BaseModel):
    tenant_id: str
    dataset_id: str
    dialog_id: str = "default"
    query: str
    limit: int = Field(default=MAX_RESULTS, ge=1, le=100)
    language: str = LANGUAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "redis": store.ping()}


async def upload(tenant_id: str, dataset_id: str, file: UploadFile, images_archive: UploadFile | None, language: str, build_image_index: bool) -> dict:
    return service.register_dataset(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        filename=file.filename or "dataset.csv",
        data=await file.read(),
        language=language,
        build_image_index=build_image_index,
        images_archive_filename=images_archive.filename if images_archive else None,
        images_archive_data=await images_archive.read() if images_archive else None,
    )


@app.post("/v1/datasets/{dataset_id}")
async def upload_dataset_new_tenant(dataset_id: str, file: UploadFile = File(...), images_archive: UploadFile | None = File(default=None), language: Annotated[str, Form()] = LANGUAGE, build_image_index: Annotated[bool, Form()] = CLIP_BUILD_ON_UPLOAD_DEFAULT) -> dict:
    tenant_id = store.create_tenant_id(32)
    result = await upload(tenant_id, dataset_id, file, images_archive, language, build_image_index)
    result["tenant_id_generated"] = True
    return result


@app.post("/v1/tenants/{tenant_id}/datasets/{dataset_id}")
async def upload_dataset(tenant_id: str, dataset_id: str, file: UploadFile = File(...), images_archive: UploadFile | None = File(default=None), language: Annotated[str, Form()] = LANGUAGE, build_image_index: Annotated[bool, Form()] = CLIP_BUILD_ON_UPLOAD_DEFAULT) -> dict:
    result = await upload(tenant_id, dataset_id, file, images_archive, language, build_image_index)
    result["tenant_id_generated"] = False
    return result


@app.post("/v1/tenants/{tenant_id}/datasets/{dataset_id}/images")
async def upload_images_archive(tenant_id: str, dataset_id: str, file: UploadFile = File(...)) -> dict:
    return service.register_images_archive(tenant_id, dataset_id, file.filename or "images.zip", await file.read())


@app.post("/v1/tenants/{tenant_id}/datasets/{dataset_id}/index_images")
def build_existing_image_index(tenant_id: str, dataset_id: str) -> dict:
    return service.build_existing_image_index(tenant_id, dataset_id)


@app.get("/v1/tenants/{tenant_id}/datasets/{dataset_id}/schema")
def get_schema(tenant_id: str, dataset_id: str) -> dict:
    runtime = service.get_runtime(tenant_id, dataset_id)
    return {
        "tenant_id": tenant_id,
        "dataset_id": dataset_id,
        "dataset_hash": runtime.dataset_hash,
        "schema": runtime.schema,
        "config": runtime.config,
        "filterable_fields": service.filterable_fields(runtime.config),
        "image_col": None,
        "image_id_col": "id" if "id" in runtime.df.columns else None,
        "images_dir": str(runtime.images_dir) if runtime.images_dir else None,
    }


@app.delete("/v1/tenants/{tenant_id}/datasets/{dataset_id}")
def delete_dataset(tenant_id: str, dataset_id: str) -> dict:
    service.delete_dataset(tenant_id, dataset_id)
    return {"deleted": True, "tenant_id": tenant_id, "dataset_id": dataset_id}


@app.post("/v1/search/text")
def search_text_json(payload: TextSearchRequest) -> dict:
    return service.search_text(payload.tenant_id, payload.dataset_id, payload.dialog_id, payload.query, payload.limit, payload.language)


@app.post("/v1/search")
async def search_unified(tenant_id: Annotated[str, Form()], dataset_id: Annotated[str, Form()], dialog_id: Annotated[str, Form()] = "default", text: Annotated[str | None, Form()] = None, image: UploadFile | None = File(default=None), limit: Annotated[int, Form(ge=1, le=100)] = MAX_RESULTS, language: Annotated[str, Form()] = LANGUAGE, alpha: Annotated[float, Form(ge=0.0, le=1.0)] = HYBRID_ALPHA) -> dict:
    text = (text or "").strip()
    if image:
        return service.search_image(tenant_id, dataset_id, dialog_id, await image.read(), text or None, limit, alpha)
    if text:
        return service.search_text(tenant_id, dataset_id, dialog_id, text, limit, language)
    raise HTTPException(400, "Provide text, image, or both")


@app.get("/v1/tenants/{tenant_id}/dialogs/{dialog_id}")
def get_dialog_history(tenant_id: str, dialog_id: str, limit: int = 10) -> dict:
    return {"tenant_id": tenant_id, "dialog_id": dialog_id, "events": store.get_dialog_history(tenant_id, dialog_id, limit)}


@app.delete("/v1/tenants/{tenant_id}/dialogs/{dialog_id}")
def clear_dialog_history(tenant_id: str, dialog_id: str) -> dict:
    store.clear_dialog(tenant_id, dialog_id)
    return {"deleted": True, "tenant_id": tenant_id, "dialog_id": dialog_id}
