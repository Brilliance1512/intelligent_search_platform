import asyncio
import io
import shutil
from pathlib import Path

import pandas as pd
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message

from app.analyzer import analyze, load_dataframe
from app.config import CAT_THRESHOLD, CLIP_MODEL, CLIP_PRETRAINED, HYBRID_ALPHA, LANGUAGE, MAX_RESULTS, TELEGRAM_BOT_TOKEN
from app.enricher import enrich_descriptions
from app.filter_engine import apply_filters, format_items
from app.image_archive import SUPPORTED_DATASET_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS, extract_zip_files, find_single_dataset_file, resolve_image_paths_by_id
from app.llm_client import chat
from app.schema_compiler import build_messages, compile_schema, parse_response

router = Router()
sessions: dict[int, dict] = {}
_clip = None
UPLOAD_DIR = Path("/tmp/filter-bot-uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def user_dir(user_id: int) -> Path:
    path = UPLOAD_DIR / str(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_clip():
    global _clip
    if _clip is None:
        from app.image_search import ImageSearchEngine
        _clip = ImageSearchEngine(CLIP_MODEL, CLIP_PRETRAINED)
    return _clip


def clip_ready() -> bool:
    return _clip is not None and _clip.is_ready


async def download_doc(msg: Message, path: Path) -> None:
    file = await msg.bot.get_file(msg.document.file_id)
    await msg.bot.download_file(file.file_path, str(path))


async def load_dataset(user_id: int, status: Message, path: Path, images_dir: Path | None = None) -> None:
    df = load_dataframe(str(path))
    await status.edit_text(f"📊 {len(df):,} строк, {len(df.columns)} столбцов. Обогащаю описания...")
    config = enrich_descriptions(analyze(df, path.name, CAT_THRESHOLD), LANGUAGE)
    schema = compile_schema(config)
    sessions[user_id] = {"df": df, "config": config, "schema": schema, "images_dir": str(images_dir) if images_dir else None}

    fields = []
    for name, info in config["fields"].items():
        if info["filterable"]:
            desc = f" — {info.get('description', '')}" if info.get("description") else ""
            fields.append(f"  • {name} ({info['type']}){desc}")

    image_note = ""
    if images_dir:
        pairs = resolve_image_paths_by_id(df, images_dir, "id")
        image_note = f"\n\n🖼 Сопоставлено картинок: {len(pairs)}. Для индексации отправь /index_images"

    await status.edit_text(f"✅ Датасет загружен: {len(df):,} строк\n\nФильтруемые поля:\n" + "\n".join(fields) + image_note)


def build_index(session: dict) -> int:
    pairs = resolve_image_paths_by_id(session["df"], Path(session["images_dir"]), "id")
    ids, paths = zip(*pairs)
    clip = get_clip()
    clip.clear_index()
    return clip.build_index(list(paths), list(ids))


@router.message(Command("start"))
async def start(msg: Message):
    await msg.answer(
        "Отправь CSV/JSON/XLSX или ZIP с таблицей и картинками id.*.\n"
        "Команды: /schema, /index_images, /reset.\n"
        "После загрузки пиши текстовый запрос или отправь фото."
    )


@router.message(Command("schema"))
async def schema(msg: Message):
    session = sessions[msg.from_user.id]
    text = f"```\n{session['schema']}\n```"
    if clip_ready():
        text += f"\n\n🖼 Визуальный поиск: {len(_clip.index_ids)} картинок"
    await msg.answer(text, parse_mode="Markdown")


@router.message(Command("reset"))
async def reset(msg: Message):
    global _clip
    sessions.pop(msg.from_user.id, None)
    if _clip:
        _clip.clear_index()
        _clip = None
    shutil.rmtree(user_dir(msg.from_user.id), ignore_errors=True)
    await msg.answer("Сброшено.")


@router.message(Command("index_images"))
async def index_images(msg: Message):
    status = await msg.answer("🖼 Индексирую картинки...")
    count = build_index(sessions[msg.from_user.id])
    await status.edit_text(f"✅ Индекс построен: {count} картинок.")


@router.message(F.document)
async def document(msg: Message):
    fname = msg.document.file_name or "file"
    ext = Path(fname).suffix.lower()
    status = await msg.answer("⏳ Скачиваю файл...")
    path = user_dir(msg.from_user.id) / fname
    await download_doc(msg, path)

    if ext == ".zip":
        extracted_dir = user_dir(msg.from_user.id) / "zip"
        files = extract_zip_files(path.read_bytes(), extracted_dir, SUPPORTED_DATASET_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS)
        dataset_files = [p for p in files if p.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS]
        if dataset_files:
            await load_dataset(msg.from_user.id, status, find_single_dataset_file(files), extracted_dir)
            return
        session = sessions[msg.from_user.id]
        session["images_dir"] = str(extracted_dir)
        pairs = resolve_image_paths_by_id(session["df"], extracted_dir, "id")
        await status.edit_text(f"✅ ZIP принят. Сопоставлено картинок: {len(pairs)}. Для индексации отправь /index_images")
        return

    await load_dataset(msg.from_user.id, status, path)


@router.message(F.photo)
async def photo(msg: Message):
    status = await msg.answer("🔍 Ищу по фото...")
    file = await msg.bot.get_file(msg.photo[-1].file_id)
    buf = io.BytesIO()
    await msg.bot.download_file(file.file_path, buf)
    buf.seek(0)

    clip = get_clip()
    caption = (msg.caption or "").strip()
    results = clip.search_hybrid(buf, caption, MAX_RESULTS, HYBRID_ALPHA) if caption else clip.search_by_image(buf, MAX_RESULTS)
    df = sessions[msg.from_user.id]["df"]
    lines = [f"Найдено: {len(results)}\n"]
    for rank, (row_idx, score) in enumerate(results, 1):
        row = df.loc[row_idx]
        shown = []
        for col in df.columns[:8]:
            val = row[col]
            if pd.notna(val) and len(str(val)) <= 80 and not str(val).startswith("http"):
                shown.append(f"  {col}: {val}")
            if len(shown) == 5:
                break
        lines.append(f"{rank}. score: {score:.3f}\n" + "\n".join(shown))
    await status.edit_text("\n\n".join(lines)[:4000])


@router.message(F.text)
async def text(msg: Message):
    query = msg.text.strip()
    status = await msg.answer("🔍 Обрабатываю запрос...")
    session = sessions[msg.from_user.id]
    result = parse_response(chat(build_messages(session["schema"], query, LANGUAGE)))
    filtered = apply_filters(session["df"], result, MAX_RESULTS)
    filters = result.get("filters", [])
    header = "Фильтры: " + ", ".join(f"{f['field']} {f['op']} {f['value']}" for f in filters) + "\n\n" if filters else "Фильтры не определены.\n\n"
    await status.edit_text((header + format_items(filtered))[:4000])


async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
