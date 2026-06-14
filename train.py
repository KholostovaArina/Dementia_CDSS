from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "visits+mapping.xlsx"
MODEL_PATH = ROOT / "future_conversion_model.joblib"

DOMAINS = [
    "Память",
    "Внимание",
    "Управляющие функции",
    "Праксис",
    "Речь",
    "Зрительное восприятие",
]

DOMAIN_ABBREV = {
    "В": "Внимание",
    "Па": "Память",
    "УФ": "Управляющие функции",
    "Пр": "Праксис",
    "Р": "Речь",
    "З": "Зрительное восприятие",
}

HORIZON_YEARS = 3


EXTRA_MODEL_FEATURES: Tuple[str, ...] = (
    "тмт1",
    "DigitSpanвперед",
    "DigitSpanназад",
    "DigitSpanобщее",
    "цифров.замещ",
)

MAPPING_COLS = {
    "test": ("test", "тест"),
    "domain": ("domain", "домен"),
    "direction": ("direction", "направление"),
    "max": ("max", "макс"),
}


def _parse_domains(raw: object) -> List[str]:
    if pd.isna(raw) or str(raw).strip() in ("", "—", "-"):
        return []
    out: List[str] = []
    for part in str(raw).split("+"):
        p = part.strip()
        dom = DOMAIN_ABBREV.get(p)
        if dom and dom not in out:
            out.append(dom)
    return out


def _mapping_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    out: Dict[str, str] = {}
    for key, aliases in MAPPING_COLS.items():
        for alias in aliases:
            if alias in lower:
                out[key] = lower[alias]
                break
    missing = [k for k in ("test", "domain", "direction", "max") if k not in out]
    if missing:
        raise ValueError(f"mapping sheet missing columns: {missing}")
    return out


def _test_entry(domains: List[str], mx: float, higher: bool) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "max": mx,
        "higher_is_better": higher,
    }
    if len(domains) == 1:
        entry["domain"] = domains[0]
    elif domains:
        entry["domains"] = domains
    return entry


def load_mapping(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    cols = _mapping_columns(df)
    cfg: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        test = str(row[cols["test"]]).strip()
        if not test or test.lower() in ("test", "тест"):
            continue

        domains = _parse_domains(row[cols["domain"]])
        direction = (
            str(row[cols["direction"]]).strip().lower()
            if pd.notna(row[cols["direction"]])
            else ""
        )
        max_v = row[cols["max"]]

        if direction not in ("прямое", "обратное") or pd.isna(max_v):
            continue

        cfg[test] = _test_entry(domains, float(max_v), direction == "прямое")

    return cfg


def _test_domains(entry: Dict[str, Any]) -> List[str]:
    if entry.get("domains"):
        return list(entry["domains"])
    d = entry.get("domain")
    return [d] if d else []


def _norm_val(value: float, mx: float, higher: bool) -> Optional[float]:
    if mx <= 0:
        return None
    v = float(max(0.0, min(mx, value)))
    if higher:
        x = v / mx
    else:
        x = (mx - v) / mx
    return float(np.clip(x, 0.0, 1.0))


def _col(row: pd.Series, name: str) -> Optional[str]:
    if name in row.index:
        return name
    low = name.lower()
    for c in row.index:
        if str(c).lower() == low:
            return str(c)
    return None


def _read_test_value(row: pd.Series, test: str) -> Optional[float]:
    col = _col(row, test)
    if col is None:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def domain_pcts(row: pd.Series, cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    by_dom: Dict[str, List[float]] = {d: [] for d in DOMAINS}

    for test, entry in cfg.items():
        raw = _read_test_value(row, test)
        if raw is None or not _test_domains(entry):
            continue
        u = _norm_val(raw, entry["max"], entry["higher_is_better"])
        if u is None:
            continue
        pct = u * 100.0
        for d in _test_domains(entry):
            if d in by_dom:
                by_dom[d].append(pct)

    return {d: round(sum(v) / len(v), 1) if v else None for d, v in by_dom.items()}


def feature_row(row: pd.Series, cfg: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    dom = domain_pcts(row, cfg)
    out: Dict[str, float] = {}
    for d in DOMAINS:
        v = dom.get(d)
        out[d] = float(v) if v is not None else float("nan")
    for test in EXTRA_MODEL_FEATURES:
        entry = cfg.get(test)
        if entry is None:
            out[test] = float("nan")
            continue
        raw = _read_test_value(row, test)
        if raw is None:
            out[test] = float("nan")
            continue
        u = _norm_val(raw, entry["max"], entry["higher_is_better"])
        out[test] = float(u) if u is not None else float("nan")
    return out


def _norm_fio(value: object) -> str:
    if pd.isna(value):
        return ""
    parts = str(value).strip().lower().replace("ё", "е").split()
    if not parts:
        return ""
    return parts[0] + "".join(p[0] for p in parts[1:])


def _target_ba(row: pd.Series) -> int:
    t = row.get("тяжестьнаруш")
    if pd.isna(t) or str(t).strip() == "":
        return 0
    s = str(t).strip().lower().replace("ё", "е")
    if s == "отсутствуют" or "субъектив" in s:
        return 0
    return 1


def _visit_year(row: pd.Series) -> Optional[int]:
    if "дата1" in row.index and pd.notna(row.get("дата1")):
        try:
            y = int(float(row["дата1"]))
            if 1985 <= y <= 2035:
                return y
        except (TypeError, ValueError):
            pass
    if "ДатаОсм" in row.index:
        dt = pd.to_datetime(row["ДатаОсм"], errors="coerce")
        if pd.notna(dt) and dt.year > 1985:
            return int(dt.year)
    return None


def _visit_num(row: pd.Series) -> int:
    if "nвизит" in row.index and pd.notna(row.get("nвизит")):
        try:
            return int(float(row["nвизит"]))
        except (TypeError, ValueError):
            pass
    return 1


def build_cohort(visits: pd.DataFrame) -> pd.DataFrame:
    df = visits.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["target_ba"] = df.apply(_target_ba, axis=1)
    if "фио" in df.columns:
        df["fio_norm"] = df["фио"].map(_norm_fio)
    df["visit_year"] = df.apply(_visit_year, axis=1)
    df["visit_num"] = df.apply(_visit_num, axis=1)
    df["visit_sort"] = df.apply(
        lambda r: (
            int(r["visit_year"]) if pd.notna(r["visit_year"]) else 9999,
            int(r["visit_num"]),
        ),
        axis=1,
    )

    rows = []
    for pid, grp in df.groupby("fio_norm", dropna=False):
        if not pid:
            continue
        g = grp.sort_values("visit_sort", kind="mergesort")
        base_idx = None
        for idx, row in g.iterrows():
            if row["target_ba"] == 0:
                base_idx = idx
                break
        if base_idx is None:
            continue

        base = g.loc[base_idx]
        y0 = base["visit_year"]
        if pd.isna(y0):
            continue
        y0, v0 = int(y0), int(base["visit_num"])
        horizon = y0 + HORIZON_YEARS

        later = g.loc[g.index != base_idx]

        def in_window(sort_key: Tuple[int, int]) -> bool:
            y, v = sort_key
            if y > horizon or y < y0:
                return False
            if y == y0:
                return v > v0
            return True

        in_win = later[later["visit_sort"].map(in_window)]
        if in_win.empty:
            continue

        converted = bool((in_win["target_ba"] == 1).any())
        out = base.to_dict()
        out["y_future_3y"] = int(converted)
        out["patient_key"] = pid
        rows.append(out)

    return pd.DataFrame(rows)


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    C=0.5,
                    random_state=42,
                ),
            ),
        ]
    )


def cv_metrics(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Optional[Dict[str, Any]]:
    n_splits = min(4, int(groups.nunique()), len(y))
    if n_splits < 2 or y.nunique() < 2:
        return None
    pipe = _make_pipeline()
    proba = cross_val_predict(
        pipe, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits), method="predict_proba"
    )[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "n_samples": int(len(y)),
        "n_conversions": int(y.sum()),
        "cv_folds": n_splits,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def _patient_count(visits: pd.DataFrame) -> int:
    if "fio_norm" in visits.columns:
        return int(visits["fio_norm"].nunique())
    if "фио" in visits.columns:
        return int(visits["фио"].astype(str).str.strip().nunique())
    return 0


def _print_metrics(report: Dict[str, Any]) -> None:
    print("--- metrics (group k-fold) ---")
    for key in (
        "visits_in_file",
        "patients_in_file",
        "cohort_n",
        "conversions",
        "cv_folds",
        "roc_auc",
        "recall",
        "accuracy",
    ):
        if key in report:
            print(f"{key}: {report[key]}")
    if report.get("features"):
        print("features:", ", ".join(report["features"]))


def main() -> None:
    visits = pd.read_excel(XLSX, sheet_name="visits")
    mapping = pd.read_excel(XLSX, sheet_name="mapping")
    cfg = load_mapping(mapping)
    missing = [t for t in EXTRA_MODEL_FEATURES if t not in cfg]
    if missing:
        print("warning: extra features not in mapping (no max/direction):", ", ".join(missing))

    cohort = build_cohort(visits)
    if cohort.empty:
        raise SystemExit("empty cohort")

    feats = [feature_row(cohort.loc[i], cfg) for i in cohort.index]
    X = pd.DataFrame(feats, index=cohort.index)
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep]
    y = cohort["y_future_3y"].astype(int)
    groups = cohort["patient_key"].astype(str)

    metrics = cv_metrics(X, y, groups)
    report: Dict[str, Any] = {
        "visits_in_file": int(len(visits)),
        "patients_in_file": _patient_count(visits),
        "cohort_n": int(len(y)),
        "conversions": int(y.sum()),
        "horizon_years": HORIZON_YEARS,
        "features": list(X.columns),
    }
    if metrics:
        report.update(metrics)
    _print_metrics(report)

    pipe = _make_pipeline()
    pipe.fit(X, y)
    blob = {
        "pipeline": pipe,
        "feature_names": list(X.columns),
        "test_config": cfg,
        "extra_model_features": list(EXTRA_MODEL_FEATURES),
        "horizon_years": HORIZON_YEARS,
        "metrics": report,
    }
    joblib.dump(blob, MODEL_PATH)
    print(f"saved {MODEL_PATH.name} (metrics inside)")


if __name__ == "__main__":
    main()
