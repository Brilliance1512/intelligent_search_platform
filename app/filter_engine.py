import pandas as pd

OPS = {
    "eq": lambda s, v: s == v,
    "neq": lambda s, v: s != v,
    "gt": lambda s, v: s > v,
    "gte": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "lte": lambda s, v: s <= v,
    "in": lambda s, v: s.isin(v),
    "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
    "between": lambda s, v: s.between(v[0], v[1]),
}


def apply_filters(df: pd.DataFrame, filter_result: dict, limit: int = 10) -> pd.DataFrame:
    result = df
    for f in filter_result.get("filters", []):
        result = result[OPS[f["op"]](result[f["field"]], f["value"])]
    return result.head(limit)


def format_items(df: pd.DataFrame, max_cols: int = 6) -> str:
    if df.empty:
        return "Ничего не найдено по вашему запросу."

    cols = []
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if not sample.empty and sample.str.len().mean() <= 100 and sample.str.match(r"^https?://").mean() <= 0.5:
            cols.append(col)
        if len(cols) == max_cols:
            break
    cols = cols or list(df.columns[:max_cols])

    lines = [f"Найдено: {len(df)}\n"]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(f"{i}. " + "\n".join(f"  {c}: {row[c]}" for c in cols if pd.notna(row[c])))
    return "\n\n".join(lines)
