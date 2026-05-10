import json
from typing import Any

ASSISTANT_PREFILL = '{"filters": ['

PROMPTS = {
    "ru": """Ты — модуль парсинга фильтров. Преобразуй запрос пользователя в JSON.

Схема датасета:
{schema}

{context}Операторы: eq, neq, gt, gte, lt, lte, in, between, contains.
Для cat-полей используй только перечисленные значения. Для num-полей учитывай диапазон. Для bool-полей используй true/false.
Если запрос уточняет прошлый контекст, сохрани применимые старые фильтры.
Ответь только JSON: {{"filters":[{{"field":"...","op":"...","value":...}}]}}
Если фильтров нет — {{"filters":[]}}""",
    "en": """You parse user filters into JSON.

Dataset schema:
{schema}

{context}Operators: eq, neq, gt, gte, lt, lte, in, between, contains.
Use only listed categorical values. Respect numeric ranges. Use true/false for booleans.
If the query refines previous context, keep applicable old filters.
Reply with JSON only: {{"filters":[{{"field":"...","op":"...","value":...}}]}}
If there are no filters — {{"filters":[]}}""",
}


def compile_schema(config: dict, max_cat_values: int = 30) -> str:
    ds = config["dataset"]
    lines = [f"Dataset: {ds['name']} ({ds['total_rows']} items)", "", "Filterable fields:"]
    for name, info in config["fields"].items():
        if not info["filterable"]:
            continue
        desc = f' — "{info.get("description", "")}"' if info.get("description") else ""
        stats = info.get("stats", {})
        if info["type"] == "categorical":
            vals = "|".join(str(v["value"]) for v in stats.get("top_values", [])[:max_cat_values])
            extra = stats.get("unique_count", 0) - min(stats.get("unique_count", 0), max_cat_values)
            lines.append(f"  {name}: cat[{vals}{f'|+{extra}' if extra > 0 else ''}]{desc}")
        if info["type"] == "numerical":
            lines.append(f"  {name}: num[{stats['min']}..{stats['max']}]{desc}")
        if info["type"] == "boolean":
            lines.append(f"  {name}: bool{desc}")
    return "\n".join(lines)


def context(history: list[dict[str, Any]] | None, language: str) -> str:
    items = [{"query": e.get("query"), "filters": e.get("filters", [])} for e in (history or []) if e.get("mode") == "text"][-3:]
    if not items:
        return ""
    prefix = "Контекст последних запросов: " if language == "ru" else "Recent context: "
    return prefix + json.dumps(items, ensure_ascii=False) + "\n"


def build_messages(schema: str, user_query: str, language: str = "ru", dialog_history: list[dict[str, Any]] | None = None) -> list[dict]:
    lang = "ru" if language == "ru" else "en"
    return [
        {"role": "system", "content": PROMPTS[lang].format(schema=schema, context=context(dialog_history, lang))},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": ASSISTANT_PREFILL},
    ]


def parse_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(ASSISTANT_PREFILL + text)
    except json.JSONDecodeError:
        return json.loads(text)
