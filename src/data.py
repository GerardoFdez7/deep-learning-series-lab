from __future__ import annotations

from pathlib import Path
import unicodedata

import pandas as pd

from .config import DATA_CANDIDATES

COLUMN_ALIASES = {
    "año": "anio",
    "ano": "anio",
    "year": "anio",
    "mes cod": "mes_cod",
    "mes_cod": "mes_cod",
    "mes": "mes",
    "via": "via",
    "vía": "via",
    "pais": "pais",
    "país": "pais",
    "region": "region",
    "región": "region",
    "tipo de viajero": "tipo_viajero",
    "tipo_viajero": "tipo_viajero",
    "tipo viajero": "tipo_viajero",
    "viajeros": "viajeros",
    "viajero": "viajeros",
}

TIPOS_CONSISTENTES = {"turista", "excursionista"}


def _normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def find_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    checked = "\n".join(f"- {p}" for p in DATA_CANDIDATES)
    raise FileNotFoundError(f"Dataset not found. Checked paths:\n{checked}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {col: COLUMN_ALIASES.get(col, col) for col in df.columns}
    return df.rename(columns=rename_map)


def _build_fecha(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "anio" not in df.columns:
        raise KeyError("Column 'anio' not found after normalization.")

    if "mes_cod" not in df.columns and "mes" in df.columns:
        mes_map = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        df["mes_cod"] = df["mes"].astype(str).str.strip().str.lower().map(mes_map)

    if "mes_cod" not in df.columns:
        raise KeyError("Column 'mes_cod' not found after normalization.")

    df["fecha"] = pd.to_datetime(
        {
            "year": pd.to_numeric(df["anio"], errors="coerce"),
            "month": pd.to_numeric(df["mes_cod"], errors="coerce"),
            "day": 1,
        },
        errors="coerce",
    )

    return df


def load_clean_data() -> pd.DataFrame:
    path = find_data_path()
    print(f"[data] Loading: {path}")
    df = pd.read_excel(path, engine="openpyxl")

    df = _normalize_columns(df)
    df = _build_fecha(df)

    df = df.dropna(subset=["fecha", "viajeros"]).copy()
    df["viajeros"] = pd.to_numeric(df["viajeros"], errors="coerce")
    df = df.dropna(subset=["viajeros"])
    df = df[df["viajeros"] >= 0]

    if "tipo_viajero" in df.columns:
        df["tipo_viajero"] = df["tipo_viajero"].astype(str).str.strip()
    if "via" in df.columns:
        df["via"] = df["via"].astype(str).str.strip()

    return df.sort_values("fecha")


def _aggregate_monthly(df: pd.DataFrame, mask: pd.Series | None = None) -> pd.Series:
    subset = df[mask] if mask is not None else df
    return subset.groupby("fecha")["viajeros"].sum().asfreq("MS").fillna(0.0)


def build_selected_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    if "tipo_viajero" not in df.columns:
        raise KeyError("Column 'tipo_viajero' is required for total_viajeros.")

    total_mask = df["tipo_viajero"].map(_normalize_text).isin(TIPOS_CONSISTENTES)
    total_series = _aggregate_monthly(df, total_mask)

    if "via" not in df.columns:
        raise KeyError("Column 'via' is required for via_aerea.")

    via_normalized = df["via"].map(_normalize_text)
    via_mask = via_normalized.str.contains("aer", na=False)
    via_aerea_series = _aggregate_monthly(df, via_mask)

    if via_aerea_series.sum() == 0:
        available_vias = sorted(via_normalized.dropna().unique().tolist())
        raise ValueError(
            "Series 'via_aerea' is empty after filtering column 'via'. "
            "Verify category names in your dataset. "
            f"Detected normalized categories: {available_vias}"
        )

    return {
        "total_viajeros": total_series,
        "via_aerea": via_aerea_series,
    }
