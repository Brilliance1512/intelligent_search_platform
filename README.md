# Универсальная платформа для мультимодального поиска и интеллектуальной рекомендации товаров

Была разработана платформа для поиска по каталогам товаров и предметов, объединяющая два сценария: запросы на естественном языке и поиск по изображению.
Главная особенность решения заключается в том, что оно не предполагает заранее заданную структуру каталога. Система получает таблицу, анализирует её поля и на основе этого
строит схему, которую затем использует при обработке запросов

# Multi-tenant API platform

Реализация добавляет FastAPI-сервис поверх существующих модулей анализа датасета, DSL-компиляции, NL-фильтрации и CLIP-поиска.

## Что есть

- `app/api.py` - HTTP API с multi-tenant маршрутизацией.
- `app/dataset_service.py` - регистрация датасетов, загрузка runtime по tenant/dataset, текстовый и визуальный поиск.
- `app/redis_store.py` - Redis-кеш DSL-схем, история диалогов и incremental `message_id`.
- `app/image_archive.py` - загрузка ZIP с картинками и сопоставление `id` → `id.*`.
- `app/image_search.py` - CLIP-модель общая на процесс, индексы раздельные для каждого датасета.
- `app/bot.py` - автономный Telegram demo-стенд.

## Запуск

```bash
cp .env.example .env
# заполнить GIGACHAT_CREDENTIALS, при необходимости TELEGRAM_BOT_TOKEN
docker compose up --build api redis
```

API будет доступно на `http://localhost:8000`, Swagger UI - на `http://localhost:8000/docs`.

## Формат датасета с картинками

В таблице должна быть колонка типа `id`. Пути к картинкам в таблице не нужны.

Картинки передаются ZIP-архивом. Имя файла должно совпадать с `id` строки:

```text
catalog.csv
images.zip
  1.jpg      -> строка с id = 1
  2.png      -> строка с id = 2
  15.webp    -> строка с id = 15
```

Поддерживаются изображения: `jpg`, `jpeg`, `png`, `webp`, `gif`, `bmp`.

Также можно отправить один ZIP-пакет, внутри которого лежит ровно одна таблица (`csv/json/jsonl/xlsx/xls`) и картинки `id.*`.

## Регистрация датасета с автоматическим tenant_id

Новый основной endpoint сам создаёт `tenant_id`: строку из 32 случайных английских букв и цифр.

```bash
curl -X POST "http://localhost:8000/v1/datasets/sneakers" \
  -F "file=@catalog.csv" \
  -F "images_archive=@images.zip" \
  -F "language=ru" \
  -F "build_image_index=false"
```

В ответе вернётся, например:

```json
{
  "tenant_id": "7uaQf7Y2BQbkV7cVqpXSy9QubymG2b3P",
  "tenant_id_generated": true,
  "dataset_id": "sneakers",
  "images_indexed": 120
}
```

Сохраните `tenant_id`: он нужен для последующих search-запросов.

### ZIP-пакет с таблицей и картинками одним файлом

```bash
curl -X POST "http://localhost:8000/v1/datasets/sneakers" \
  -F "file=@sneakers_package.zip" \
  -F "language=ru" \
  -F "build_image_index=false"
```

Архив должен содержать ровно одну таблицу и картинки `id.*`.

## Загрузка/перестроение архива картинок отдельно

Если таблица уже загружена, можно отправить ZIP с картинками отдельно:

```bash
curl -X POST "http://localhost:8000/v1/tenants/7uaQf7Y2BQbkV7cVqpXSy9QubymG2b3P/datasets/sneakers/images" \
  -F "file=@images.zip"
```

API извлечёт картинки, сопоставит их по колонке `id` и перестроит CLIP-индекс. При первом запуске может скачиваться CLIP-модель.

## Построение индекса из уже загруженных картинок

Если датасет был загружен ZIP-пакетом с таблицей и картинками, но `build_image_index=false`, индекс можно построить отдельной командой без повторной загрузки архива:

```bash
curl -X POST "http://localhost:8000/v1/tenants/<tenant_id>/datasets/sneakers/index_images"
```

## Текстовый поиск

`message_id` не передаётся клиентом. Он создаётся автоматически через Redis `INCR` и уникален внутри каждого `tenant_id`.

```bash
curl -X POST "http://localhost:8000/v1/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "7uaQf7Y2BQbkV7cVqpXSy9QubymG2b3P",
    "dataset_id": "sneakers",
    "dialog_id": "default",
    "query": "черные Nike дешевле 10000",
    "limit": 10,
    "language": "ru"
  }'
```

Ответ содержит `message_id`:

```json
{
  "tenant_id": "7uaQf7Y2BQbkV7cVqpXSy9QubymG2b3P",
  "dataset_id": "sneakers",
  "dialog_id": "default",
  "message_id": 1,
  "mode": "text",
  "items": []
}
```

## Единый endpoint: текст, изображение или hybrid

```bash
curl -X POST "http://localhost:8000/v1/search" \
  -F "tenant_id=7uaQf7Y2BQbkV7cVqpXSy9QubymG2b3P" \
  -F "dataset_id=sneakers" \
  -F "dialog_id=default" \
  -F "text=но в черном цвете" \
  -F "image=@query.jpg" \
  -F "limit=10"
```

Если передать только `text`, endpoint выполнит NL-фильтрацию. Если только `image` - image-to-image поиск. Если `text + image` - hybrid поиск. В каждом ответе будет новый `message_id`.

## История диалога

```bash
curl "http://localhost:8000/v1/tenants/7uaQf7Y2BQbkV7cVqpXSy9QubymG2b3P/dialogs/default"
```

История хранится в Redis с TTL и используется как краткий контекст для уточняющих запросов: "покажи дешевле", "такие же, но красные".

## Запуск Telegram-демо

Telegram-бот оставлен как автономный демо-стенд и не зависит от API.

1. Укажите реальный токен в `.env`:

```env
TELEGRAM_BOT_TOKEN=123456:ABCDEF_your_real_bot_token
```

2. Запустите только сервис бота:

```powershell
docker compose --profile bot up --build bot
```