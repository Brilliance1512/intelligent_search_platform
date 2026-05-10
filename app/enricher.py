import json

from app.llm_client import chat


def enrich_descriptions(config: dict, language: str = "ru") -> dict:
    fields = config["fields"]
    ds_name = config["dataset"]["name"]
    summaries = []

    for name, info in fields.items():
        s = f"- {name} (type: {info['type']})"
        stats = info.get("stats", {})
        if info["type"] == "categorical":
            vals = [str(v["value"]) for v in stats.get("top_values", [])[:8]]
            s += f" values: [{', '.join(vals)}]"
        if info["type"] == "numerical":
            s += f" range: {stats.get('min')}..{stats.get('max')}"
        summaries.append(s)

    lang = "Ответь на русском." if language == "ru" else "Answer in English."
    prompt = (
        f'Датасет "{ds_name}". Для каждого поля напиши описание 5-15 слов.\n'
        f"{lang}\n\nПоля:\n" + "\n".join(summaries) +
        '\n\nТолько JSON: {"field_name": "описание", ...}'
    )
    raw = chat([
        {"role": "system", "content": "Ты описываешь поля датасета. Отвечай только JSON."},
        {"role": "user", "content": prompt},
    ]).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    for name, desc in json.loads(raw).items():
        fields[name]["description"] = desc
    return config
