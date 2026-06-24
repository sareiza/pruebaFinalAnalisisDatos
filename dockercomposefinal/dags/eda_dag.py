"""
DAG de Análisis Exploratorio de Datos (EDA).

Analiza las dos fuentes crudas antes de cualquier transformación:
  - Fuente 1: data.csv              (Kaggle: ecommerce-data)
  - Fuente 2: online_retail_II.xlsx (Kaggle: online-retail-transaction-dataset,
              hojas "Year 2009-2010" y "Year 2010-2011")

No requiere descomprimir nada a mano: la primera tarea extrae los .zip que ya
están en DATA_DIR. Cada tarea guarda sus resultados en disco (JSON/CSV) dentro
de DATA_DIR/eda_reports y DATA_DIR/staging para que el DAG de transformación
pueda leer estas decisiones (nulos por columna, registros rechazados con
motivo, descripciones canónicas por código, reporte de solapamiento).
"""
from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task
from airflow.models import Variable

DATA_DIR = Path(Variable.get("DATA_DIR", default_var="/opt/airflow/data"))
REPORTS_DIR = DATA_DIR / "eda_reports"
STAGING_DIR = DATA_DIR / "staging"

SOURCE1_ZIP = DATA_DIR / "archive.zip"
SOURCE1_ZIP_MEMBER = "data.csv"
SOURCE1_CSV = DATA_DIR / "data.csv"

SOURCE2_ZIP = DATA_DIR / "archive (1).zip"
SOURCE2_ZIP_MEMBER = "online_retail_II.xlsx"
SOURCE2_XLSX = DATA_DIR / "online_retail_II.xlsx"

# Mapeo semántico -> nombre real de columna en cada fuente.
SOURCE1_COLUMNS = {
    "invoice": "InvoiceNo",
    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoice_date": "InvoiceDate",
    "price": "UnitPrice",
    "customer_id": "CustomerID",
    "country": "Country",
}

SOURCE2_COLUMNS = {
    "invoice": "Invoice",
    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoice_date": "InvoiceDate",
    "price": "Price",
    "customer_id": "Customer ID",
    "country": "Country",
}

NUMERIC_CODE_RE = re.compile(r"^\d+$")
STARTS_WITH_LETTER_RE = re.compile(r"^[A-Za-z]")


# ---------------------------------------------------------------------------
# Lectura de fuentes
# ---------------------------------------------------------------------------

def _read_source1() -> pd.DataFrame:
    df = pd.read_csv(
        SOURCE1_CSV,
        dtype={"StockCode": "string", "InvoiceNo": "string"},
        encoding="latin1",
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    return df


def _read_source2() -> pd.DataFrame:
    sheets = pd.read_excel(
        SOURCE2_XLSX,
        sheet_name=None,
        dtype={"StockCode": "string", "Invoice": "string"},
    )
    frames = []
    for sheet_name, sheet_df in sheets.items():
        sheet_df = sheet_df.copy()
        sheet_df["SourceSheet"] = sheet_name
        frames.append(sheet_df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Chequeos reutilizables
# ---------------------------------------------------------------------------

def _schema_report(df: pd.DataFrame) -> dict:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def _null_report(df: pd.DataFrame) -> dict:
    return {col: int(n) for col, n in df.isna().sum().items()}


def _quantity_report(df: pd.DataFrame, qty_col: str) -> dict:
    return {
        "non_positive_count": int((df[qty_col] <= 0).sum()),
        "min": float(df[qty_col].min()),
        "max": float(df[qty_col].max()),
        "mean": float(df[qty_col].mean()),
    }


def _price_report(df: pd.DataFrame, price_col: str) -> dict:
    return {
        "non_positive_count": int((df[price_col] <= 0).sum()),
        "min": float(df[price_col].min()),
        "max": float(df[price_col].max()),
        "mean": float(df[price_col].mean()),
    }


def _stock_code_report(df: pd.DataFrame, code_col: str) -> dict:
    codes = df[code_col].dropna().astype(str)
    non_numeric = codes[~codes.str.match(NUMERIC_CODE_RE)]
    starts_with_letter = codes[codes.str.match(STARTS_WITH_LETTER_RE)]
    return {
        "unique_codes": int(codes.nunique()),
        "non_numeric_codes_count": int(non_numeric.nunique()),
        "starts_with_letter_count": int(starts_with_letter.nunique()),
        "non_numeric_examples": sorted(non_numeric.unique().tolist())[:20],
    }


def _date_report(df: pd.DataFrame, date_col: str) -> dict:
    return {
        "min_date": df[date_col].min().isoformat(),
        "max_date": df[date_col].max().isoformat(),
        "days_covered": int((df[date_col].max() - df[date_col].min()).days),
    }


def _country_report(df: pd.DataFrame, country_col: str) -> dict:
    counts = df[country_col].value_counts(dropna=False)
    return {
        "distinct_countries": int(df[country_col].nunique(dropna=True)),
        "distribution": {str(k): int(v) for k, v in counts.items()},
    }


def _description_variation_report(
    df: pd.DataFrame, code_col: str, desc_col: str
) -> tuple[dict, pd.DataFrame]:
    """Variaciones de escritura (mayúsculas/minúsculas/mixto) por código.
    El "canónico" elegido es la descripción literal más frecuente para ese código.
    """
    sub = df[[code_col, desc_col]].dropna().copy()
    sub[desc_col] = sub[desc_col].astype(str).str.strip()

    variations = sub.groupby(code_col)[desc_col].agg(lambda s: sorted(set(s)))
    canonical = sub.groupby(code_col)[desc_col].agg(lambda s: s.value_counts().idxmax())

    canon_df = pd.DataFrame(
        {
            code_col: variations.index,
            "canonical_description": canonical.values,
            "n_variations": variations.apply(len).values,
            "variations": variations.apply(lambda v: " | ".join(v)).values,
        }
    )

    report = {
        "codes_with_multiple_descriptions": int((canon_df["n_variations"] > 1).sum()),
        "max_variations_for_a_code": int(canon_df["n_variations"].max()) if len(canon_df) else 0,
    }
    return report, canon_df


def _build_rejected(df: pd.DataFrame, qty_col: str, price_col: str) -> pd.DataFrame:
    mask = (df[qty_col] <= 0) | (df[price_col] <= 0)
    rejected = df[mask].copy()

    def _reason(row) -> str:
        reasons = []
        if row[qty_col] <= 0:
            reasons.append("quantity_non_positive")
        if row[price_col] <= 0:
            reasons.append("price_non_positive")
        return ",".join(reasons)

    rejected["reject_reason"] = rejected.apply(_reason, axis=1)
    return rejected


default_args = {"owner": "sabin", "retries": 1}


@dag(
    dag_id="EDA",
    description="Analisis exploratorio (EDA) de data.csv y online_retail_II antes de transformar",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["eda", "ecommerce"],
)
def eda_dag():

    @task
    def extract_sources() -> dict:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

        if not SOURCE1_CSV.exists():
            with zipfile.ZipFile(SOURCE1_ZIP) as zf:
                zf.extract(SOURCE1_ZIP_MEMBER, DATA_DIR)

        if not SOURCE2_XLSX.exists():
            with zipfile.ZipFile(SOURCE2_ZIP) as zf:
                zf.extract(SOURCE2_ZIP_MEMBER, DATA_DIR)

        return {"source1_csv": str(SOURCE1_CSV), "source2_xlsx": str(SOURCE2_XLSX)}

    @task
    def profile_source1(paths: dict) -> dict:
        df = _read_source1()
        c = SOURCE1_COLUMNS

        null_counts = _null_report(df)
        profile = {
            "source": "data.csv",
            "row_count": int(len(df)),
            "schema": _schema_report(df),
            "nulls": null_counts,
            "missing_customer_id": null_counts.get(c["customer_id"], 0),
            "quantity": _quantity_report(df, c["quantity"]),
            "price": _price_report(df, c["price"]),
            "stock_code": _stock_code_report(df, c["stock_code"]),
            "dates": {
                **_date_report(df, c["invoice_date"]),
                "raw_format": "texto '%m/%d/%Y %H:%M', ej: '12/1/2010 8:26'",
            },
            "country": _country_report(df, c["country"]),
        }

        desc_report, canon_df = _description_variation_report(df, c["stock_code"], c["description"])
        profile["description_variations"] = desc_report

        rejected = _build_rejected(df, c["quantity"], c["price"])
        profile["rejected_count"] = int(len(rejected))

        rejected_path = REPORTS_DIR / "source1_rejected.csv"
        rejected.to_csv(rejected_path, index=False)

        canon_path = STAGING_DIR / "source1_canonical_descriptions.csv"
        canon_df.to_csv(canon_path, index=False)

        (REPORTS_DIR / "source1_profile.json").write_text(
            json.dumps(profile, indent=2, ensure_ascii=False, default=str)
        )

        keys = df[[c["invoice"], c["stock_code"]]].copy()
        keys.columns = ["invoice", "stock_code"]
        keys["date_only"] = df[c["invoice_date"]].dt.date.astype(str)
        keys_path = STAGING_DIR / "source1_keys.csv"
        keys.to_csv(keys_path, index=False)

        return {
            "profile": profile,
            "rejected_path": str(rejected_path),
            "canonical_path": str(canon_path),
            "keys_path": str(keys_path),
            "min_date": df[c["invoice_date"]].min().isoformat(),
            "max_date": df[c["invoice_date"]].max().isoformat(),
            "unique_codes": canon_df["StockCode"].tolist(),
        }

    @task
    def profile_source2(paths: dict, s1: dict) -> dict:
        df = _read_source2()
        c = SOURCE2_COLUMNS

        null_counts = _null_report(df)
        profile = {
            "source": "online_retail_II.xlsx (Year 2009-2010 + Year 2010-2011)",
            "row_count": int(len(df)),
            "schema": _schema_report(df),
            "nulls": null_counts,
            "missing_customer_id": null_counts.get(c["customer_id"], 0),
            "quantity": _quantity_report(df, c["quantity"]),
            "price": _price_report(df, c["price"]),
            "stock_code": _stock_code_report(df, c["stock_code"]),
            "dates": {
                **_date_report(df, c["invoice_date"]),
                "raw_format": "datetime64 nativo de Excel (ya parseado)",
            },
            "country": _country_report(df, c["country"]),
            "compatibility_with_source1": {
                "column_name_diffs": {
                    key: {
                        "source1": SOURCE1_COLUMNS[key],
                        "source2": SOURCE2_COLUMNS[key],
                        "same_name": SOURCE1_COLUMNS[key] == SOURCE2_COLUMNS[key],
                    }
                    for key in SOURCE1_COLUMNS
                },
                "date_format_matches": False,
                "date_format_note": "fuente1 es texto m/d/Y H:M, fuente2 es datetime nativo de Excel",
            },
        }

        desc_report, canon_df = _description_variation_report(df, c["stock_code"], c["description"])
        profile["description_variations"] = desc_report

        rejected = _build_rejected(df, c["quantity"], c["price"])
        profile["rejected_count"] = int(len(rejected))

        rejected_path = REPORTS_DIR / "source2_rejected.csv"
        rejected.to_csv(rejected_path, index=False)

        canon_path = STAGING_DIR / "source2_canonical_descriptions.csv"
        canon_df.to_csv(canon_path, index=False)

        (REPORTS_DIR / "source2_profile.json").write_text(
            json.dumps(profile, indent=2, ensure_ascii=False, default=str)
        )

        keys = df[[c["invoice"], c["stock_code"]]].copy()
        keys.columns = ["invoice", "stock_code"]
        keys["date_only"] = df[c["invoice_date"]].dt.date.astype(str)
        keys_path = STAGING_DIR / "source2_keys.csv"
        keys.to_csv(keys_path, index=False)

        return {
            "profile": profile,
            "rejected_path": str(rejected_path),
            "canonical_path": str(canon_path),
            "keys_path": str(keys_path),
            "min_date": df[c["invoice_date"]].min().isoformat(),
            "max_date": df[c["invoice_date"]].max().isoformat(),
            "unique_codes": canon_df["StockCode"].tolist(),
        }

    @task
    def cross_source_checks(s1: dict, s2: dict) -> dict:
        s1_min, s1_max = pd.Timestamp(s1["min_date"]), pd.Timestamp(s1["max_date"])
        s2_min, s2_max = pd.Timestamp(s2["min_date"]), pd.Timestamp(s2["max_date"])

        overlap_start = max(s1_min, s2_min)
        overlap_end = min(s1_max, s2_max)
        has_overlap = overlap_start <= overlap_end

        keys1 = pd.read_csv(s1["keys_path"], dtype=str)
        keys2 = pd.read_csv(s2["keys_path"], dtype=str)
        keys1["key"] = keys1["invoice"] + "|" + keys1["stock_code"] + "|" + keys1["date_only"]
        keys2["key"] = keys2["invoice"] + "|" + keys2["stock_code"] + "|" + keys2["date_only"]

        set1, set2 = set(keys1["key"]), set(keys2["key"])
        duplicate_keys = set1 & set2

        records_in_overlap_window = None
        if has_overlap:
            d1 = pd.to_datetime(keys1["date_only"])
            records_in_overlap_window = int(((d1 >= overlap_start) & (d1 <= overlap_end)).sum())

        codes1, codes2 = set(s1["unique_codes"]), set(s2["unique_codes"])

        canon1 = pd.read_csv(s1["canonical_path"])
        canon2 = pd.read_csv(s2["canonical_path"])
        merged = canon1.merge(canon2, on="StockCode", how="inner", suffixes=("_s1", "_s2"))
        merged["consistent"] = (
            merged["canonical_description_s1"].str.upper().str.strip()
            == merged["canonical_description_s2"].str.upper().str.strip()
        )
        merged[~merged["consistent"]].to_csv(
            REPORTS_DIR / "description_inconsistencies.csv", index=False
        )

        cross = {
            "date_overlap": {
                "source1_range": [s1["min_date"], s1["max_date"]],
                "source2_range": [s2["min_date"], s2["max_date"]],
                "has_overlap": bool(has_overlap),
                "overlap_start": overlap_start.isoformat() if has_overlap else None,
                "overlap_end": overlap_end.isoformat() if has_overlap else None,
                "source1_records_in_overlap_window": records_in_overlap_window,
            },
            "duplicate_keys_between_sources": len(duplicate_keys),
            "combined_unique_codes": len(codes1 | codes2),
            "codes_only_in_source1": len(codes1 - codes2),
            "codes_only_in_source2": len(codes2 - codes1),
            "codes_in_both_sources": len(codes1 & codes2),
            "description_consistency": {
                "codes_compared": int(len(merged)),
                "consistent_count": int(merged["consistent"].sum()),
                "inconsistent_count": int((~merged["consistent"]).sum()),
            },
        }

        pd.DataFrame({"duplicate_key": sorted(duplicate_keys)}).to_csv(
            REPORTS_DIR / "duplicates_between_sources.csv", index=False
        )
        (REPORTS_DIR / "cross_source_overlap.json").write_text(
            json.dumps(cross, indent=2, ensure_ascii=False, default=str)
        )

        return cross

    @task
    def build_summary(s1: dict, s2: dict, cross: dict) -> str:
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "source1": {
                "profile_path": str(REPORTS_DIR / "source1_profile.json"),
                "rejected_path": s1["rejected_path"],
                "canonical_descriptions_path": s1["canonical_path"],
                "nulls": s1["profile"]["nulls"],
                "rejected_count": s1["profile"]["rejected_count"],
            },
            "source2": {
                "profile_path": str(REPORTS_DIR / "source2_profile.json"),
                "rejected_path": s2["rejected_path"],
                "canonical_descriptions_path": s2["canonical_path"],
                "nulls": s2["profile"]["nulls"],
                "rejected_count": s2["profile"]["rejected_count"],
                "compatibility_with_source1": s2["profile"]["compatibility_with_source1"],
            },
            "cross_source": cross,
        }
        path = REPORTS_DIR / "eda_summary.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return str(path)

    paths = extract_sources()
    s1 = profile_source1(paths)
    s2 = profile_source2(paths, s1)
    cross = cross_source_checks(s1, s2)
    build_summary(s1, s2, cross)


eda_dag()
