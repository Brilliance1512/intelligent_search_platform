from collections import OrderedDict

import numpy as np
import pandas as pd


def classify_field(s: pd.Series, cat_threshold: int = 50) -> str:
    x = s.dropna()
    if x.empty:
        return "skip"
    values = set(x.unique())
    if values <= {True, False, 0, 1, "true", "false", "True", "False", "yes", "no"}:
        return "boolean"
    if pd.api.types.is_numeric_dtype(x):
        n = x.nunique()
        if n <= 10 and n / len(x) < 0.01:
            return "categorical"
        if n == len(x) and (n > 100 or (pd.api.types.is_integer_dtype(x) and (x.diff().dropna() == 1).all())):
            return "id"
        return "numerical"
    t = x.astype(str)
    avg_len = t.str.len().mean()
    n = x.nunique()
    if t.str.match(r"^https?://").mean() > 0.8:
        return "url"
    if t.str.contains(r"\.(jpg|jpeg|png|gif|webp|svg|bmp|tiff)$", case=False, regex=True).mean() > 0.5:
        return "image_path"
    if n == len(x) and avg_len < 40:
        return "id"
    if n <= cat_threshold or (n <= 200 and avg_len < 30):
        return "categorical"
    return "text"


def native(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 2)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def analyze(df: pd.DataFrame, name: str = "dataset", cat_threshold: int = 50) -> dict:
    fields = OrderedDict()
    for col in df.columns:
        s = df[col]
        ftype = classify_field(s, cat_threshold)
        info = {"type": ftype, "filterable": ftype in {"categorical", "numerical", "boolean"}, "description": "", "stats": {}}
        if ftype == "categorical":
            vc = s.dropna().value_counts()
            info["stats"] = {"unique_count": int(s.nunique()), "top_values": [{"value": native(v), "count": int(c)} for v, c in vc.head(100).items()]}
        if ftype == "numerical":
            x = s.dropna()
            info["stats"] = {"min": native(x.min()), "max": native(x.max())}
        fields[col] = info
    return {"dataset": {"name": name, "total_rows": len(df)}, "fields": fields}


def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, on_bad_lines="skip")
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported format: {path}")
