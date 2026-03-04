#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DescStats (dataset-driven)
- Uses ONLY the provided encoded dataset (CSV) and codebook (TSV)
- Auto-detects id/year columns, code columns, and variable names from the codebook
- English-only outputs (labels from codebook.en_label)
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Logging & FS
# -----------------------------
def log(msg: str):
    print(f"[descstats] {msg}", flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------
# Smart IO (encodings)
# -----------------------------
def read_table_smart(path: str, sep: str = ",", preferred: Optional[str] = None) -> pd.DataFrame:
    enc_candidates = []
    if preferred and isinstance(preferred, str) and preferred.strip():
        enc_candidates.append(preferred.strip().lower())
    enc_candidates += ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]

    last_err = None
    for enc in enc_candidates:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc)
            log(f"Loaded '{os.path.basename(path)}' with encoding={enc}")
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # Surface other parsing errors immediately
            raise
    msg = (f"Unable to decode file '{path}' with tried encodings: {enc_candidates}. "
           f"Last error: {type(last_err).__name__}: {last_err}") if last_err else \
          (f"Unable to decode file '{path}' with tried encodings: {enc_candidates}.")
    raise RuntimeError(msg)


# -----------------------------
# Helpers
# -----------------------------
def norm_token(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum() or ch in ["_", "-"])


def autodetect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # 1) exact
    for c in candidates:
        if c in df.columns:
            return c
    # 2) loose matching
    norm_map = {norm_token(c): c for c in df.columns}
    for c in candidates:
        key = norm_token(c)
        if key in norm_map:
            return norm_map[key]
    return None


def proportions(series: pd.Series, drop_unknown: bool = True) -> pd.Series:
    if drop_unknown:
        series = series[series != "Unknown"]
    cnt = series.value_counts(dropna=False)
    tot = cnt.sum()
    if tot == 0:
        return pd.Series(dtype=float)
    return (cnt / tot).sort_values(ascending=False)


def diversity_index_1_minus_hhi(probs: pd.Series) -> float:
    if probs is None or probs.empty:
        return np.nan
    p = probs / probs.sum()
    return float(1.0 - np.sum(np.square(p)))


def make_yearbin(year: pd.Series, bins: List[List[int]]) -> pd.Series:
    labels = [f"{int(lo)}–{int(hi)}" for lo, hi in bins]
    res = pd.Series(index=year.index, dtype="string")
    for (lo, hi), lab in zip(bins, labels):
        mask = (year >= lo) & (year <= hi)
        res.loc[mask] = lab
    return res


def safe_save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=True if df.index.name else False, encoding="utf-8")


def safe_save_png(fig, path: str, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Variable/column detection (dataset-driven)
# -----------------------------
def detect_id_year_columns(df: pd.DataFrame,
                           cfg: dict) -> Tuple[str, str]:
    # From config if provided
    id_col = cfg.get("paper_id_column", "")
    year_col = cfg.get("year_column", "")

    # Fallback candidates
    if not id_col:
        id_col = autodetect_column(df, ["paper_id", "id", "doc_id", "pid", "paperid"])
    if not year_col:
        year_col = autodetect_column(df, ["year", "pub_year", "yr"])

    if not id_col or id_col not in df.columns:
        raise KeyError("Failed to detect paper_id column. Set 'paper_id_column' in config_descstats.yaml.")
    if not year_col or year_col not in df.columns:
        raise KeyError("Failed to detect year column. Set 'year_column' in config_descstats.yaml.")

    return id_col, year_col


def map_code_columns(df: pd.DataFrame, cb: pd.DataFrame) -> Dict[str, str]:
    """
    Return mapping: variable_name_in_codebook -> dataset_code_column
    Strategy:
      - Find all dataset columns likely to be code columns: ending with '_codes' OR integer-like category columns
      - For each codebook 'variable', try to match a dataset column by (1) exact '<variable>_codes',
        (2) loose norm match, (3) if multiple matches, prefer the one with more non-null values.
    """
    # candidate dataset code columns
    code_like_cols = [c for c in df.columns if c.lower().endswith("_codes")]
    # also consider integer-like small-cardinality columns (but skip year/id)
    for c in df.columns:
        if c in code_like_cols:
            continue
        s = df[c]
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s):
            # heuristic: not too many unique values, not all unique
            try:
                nunq = s.dropna().astype(str).nunique()
            except Exception:
                continue
            if 2 <= nunq <= 1000:  # wide but reasonable
                code_like_cols.append(c)

    code_like_cols = list(dict.fromkeys(code_like_cols))  # dedup
    var_values = cb["variable"].astype(str).dropna().unique().tolist()

    var2col = {}
    for v in var_values:
        preferred = [
            f"{v}_codes", f"{v}_code", v,
            v.lower(), v.upper()
        ]
        # exact first
        chosen = autodetect_column(df, preferred)
        if not chosen:
            # loose by normalized token overlap
            v_norm = norm_token(v)
            best = None
            best_nonnull = -1
            for c in code_like_cols:
                if v_norm in norm_token(c):
                    nonnull = df[c].notna().sum()
                    if nonnull > best_nonnull:
                        best_nonnull = nonnull
                        best = c
            chosen = best
        if chosen:
            var2col[v] = chosen
    return var2col


def choose_primary_variable(var2col: Dict[str, str],
                            cb: pd.DataFrame,
                            df: pd.DataFrame) -> Optional[str]:
    """
    Choose the 'primary' variable (used for method spectrum & diversity).
    Priority:
      1) If config specifies primary_variable and exists -> use it (handled in caller)
      2) Variable name contains any of: MethodFamily, Methodology, MethodClass, MethodTag
      3) Among remaining, choose the one with highest entropy (but <= 20 distinct labels)
    """
    if not var2col:
        return None

    # 2) name priority
    name_priority = ["methodfamily", "methodology", "methodclass", "methodtag",
                     "方法", "门类", "家族"]
    for v, col in var2col.items():
        vn = norm_token(v)
        if any(tok in vn for tok in name_priority):
            return v

    # 3) entropy criterion on en_labels (we don't have labels yet; approximate with code cardinality)
    best_v = None
    best_score = -1.0
    for v, col in var2col.items():
        # prefer variables with 3-20 categories
        nunq = df[col].astype(str).nunique(dropna=True)
        if 3 <= nunq <= 20:
            # use nunq as a proxy for diversity/entropy
            score = float(nunq)
            if score > best_score:
                best_score = score
                best_v = v
    if best_v:
        return best_v

    # Fallback: the variable with maximum non-null count
    best_v = None
    best_nonnull = -1
    for v, col in var2col.items():
        nn = df[col].notna().sum()
        if nn > best_nonnull:
            best_nonnull = nn
            best_v = v
    return best_v


def to_en_label_from_code(s_codes: pd.Series, cb: pd.DataFrame, variable_name: str) -> pd.Series:
    sub = cb.loc[cb["variable"].astype(str) == str(variable_name), ["code", "en_label"]].copy()
    sub["code_str"] = sub["code"].astype(str).str.strip()
    s_str = s_codes.astype(str).str.strip()
    out = s_str.map(dict(zip(sub["code_str"], sub["en_label"])))
    out = out.astype("string")
    out[out.isna()] = "Unknown"
    return out


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(config_path: str):
    # Load config
    cfg = load_yaml(config_path)
    data_path = cfg.get("data_path", "dataset_encoded.csv")
    codebook_path = cfg.get("codebook_path", "codebook_mapping.tsv")
    base_output_dir = cfg.get("output_dir", "results_descstats")
    dpi = int(cfg.get("dpi", 300))
    style = cfg.get("style", "whitegrid")
    figsize = cfg.get("figsize", {"width": 9, "height": 5})
    show_plots = bool(cfg.get("show_plots", False))
    save_png = bool(cfg.get("save_png", True))
    save_csv = bool(cfg.get("save_csv", True))
    rolling_window = int(cfg.get("rolling_window", 3))

    # encodings (optional)
    data_encoding_pref = cfg.get("data_encoding", "")
    codebook_encoding_pref = cfg.get("codebook_encoding", "")

    # Optional: explicit variable names (must exist in codebook.variable)
    primary_variable_cfg = cfg.get("primary_variable", "")           # e.g., "MethodFamily"
    transparency_variable_cfg = cfg.get("transparency_variable", "") # e.g., "MethodTransparency"
    topic_variable_cfg = cfg.get("topic_variable", "")               # e.g., "Topic"

    # Year bins
    year_bins = cfg.get("year_bins", [[1999, 2014], [2015, 2019], [2020, 2025]])

    # Read files
    log("Loading files ...")
    if not os.path.isfile(data_path):
        log(f"ERROR: dataset not found at {data_path}")
        sys.exit(1)
    if not os.path.isfile(codebook_path):
        log(f"ERROR: codebook not found at {codebook_path}")
        sys.exit(1)

    df = read_table_smart(data_path, sep=",", preferred=data_encoding_pref)
    cb = read_table_smart(codebook_path, sep="\t", preferred=codebook_encoding_pref)

    # Basic columns in codebook
    for col in ["variable", "code", "en_label"]:
        if col not in cb.columns:
            log(f"ERROR: codebook must contain column '{col}'.")
            sys.exit(1)

    # Detect id/year
    try:
        paper_id_col, year_col = detect_id_year_columns(df, cfg)
    except KeyError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    # Coerce year, clean id
    log("Coercing year and indexing ...")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[year_col]).copy()
    df[paper_id_col] = df[paper_id_col].astype("string")

    # Map variables -> dataset code columns
    var2col = map_code_columns(df, cb)
    log(f"Detected variable->column mapping (from codebook vs dataset): {var2col}")

    if not var2col:
        log("ERROR: No code columns could be mapped from codebook variables to dataset columns.")
        sys.exit(1)

    # Decide variables (primary/transparency/topic)
    primary_var = primary_variable_cfg if primary_variable_cfg in var2col else None
    if primary_variable_cfg and not primary_var:
        log(f"WARNING: primary_variable='{primary_variable_cfg}' not found in dataset/codebook mapping; will auto-choose.")
    if not primary_var:
        primary_var = choose_primary_variable(var2col, cb, df)
    if not primary_var:
        log("ERROR: Failed to determine a primary variable. Consider setting 'primary_variable' in YAML to a name in codebook.variable.")
        sys.exit(1)

    transparency_var = transparency_variable_cfg if transparency_variable_cfg in var2col else None
    if transparency_variable_cfg and not transparency_var:
        log(f"INFO: transparency_variable='{transparency_variable_cfg}' not found; transparency-related outputs will be skipped.")
    if not transparency_var:
        # try common names
        for cand in ["MethodTransparency", "Transparency"]:
            if cand in var2col:
                transparency_var = cand
                break

    topic_var = topic_variable_cfg if topic_variable_cfg in var2col else None
    if topic_variable_cfg and not topic_var:
        log(f"INFO: topic_variable='{topic_variable_cfg}' not found; topic-related outputs will be skipped.")
    if not topic_var:
        for cand in ["Topic", "Subject", "Theme"]:
            if cand in var2col:
                topic_var = cand
                break

    # Build label columns
    log("Mapping codes to English labels based on codebook ...")
    df_primary_en = to_en_label_from_code(df[var2col[primary_var]], cb, primary_var)
    df["Primary_en"] = df_primary_en

    if transparency_var:
        df["Transparency_en"] = to_en_label_from_code(df[var2col[transparency_var]], cb, transparency_var)
    else:
        df["Transparency_en"] = "Unknown"

    has_topic = False
    if topic_var:
        df["Topic_en"] = to_en_label_from_code(df[var2col[topic_var]], cb, topic_var)
        has_topic = True
    else:
        df["Topic_en"] = "Unknown"

    # YearBin
    log("Creating YearBin ...")
    df["YearBin"] = make_yearbin(df[year_col].astype(float), year_bins)

    # Prepare output dir
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = f"{base_output_dir}_{date_tag}"
    ensure_dir(out_dir)
    sns.set_style(style)

    # META
    meta = {
        "run_datetime": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(df.shape[0]),
        "year_min": int(df[year_col].min()) if not df.empty else None,
        "year_max": int(df[year_col].max()) if not df.empty else None,
        "year_bins": year_bins,
        "id_column": paper_id_col,
        "year_column": year_col,
        "var2col": var2col,
        "primary_variable": primary_var,
        "transparency_variable": transparency_var,
        "topic_variable": topic_var,
        "output_dir": out_dir
    }
    with open(os.path.join(out_dir, "run_meta_descstats.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Diagnostics
    log(f"Primary variable chosen: '{primary_var}' -> column '{var2col[primary_var]}'")
    log(f"Transparency variable: '{transparency_var}' -> column '{var2col.get(transparency_var, None)}'")
    log(f"Topic variable: '{topic_var}' -> column '{var2col.get(topic_var, None)}'")
    log(f"Head (Primary/Transparency/Topic):\n{df[['Primary_en','Transparency_en','Topic_en']].head().to_string(index=False)}")
    unknown_rate = (df["Primary_en"] == "Unknown").mean()
    log(f"Unknown rate of Primary_en: {unknown_rate:.2%}")

    # ============ Tables ============
    log("Generating tables ...")

    # Overall proportions (include/exclude Unknown)
    overall_prop = proportions(df["Primary_en"], drop_unknown=False).rename("proportion_incl_unknown")
    overall_prop_ex = proportions(df["Primary_en"], drop_unknown=True).rename("proportion_excl_unknown")
    tbl_overall = pd.concat([overall_prop, overall_prop_ex], axis=1)
    if save_csv:
        safe_save_csv(tbl_overall, os.path.join(out_dir, "primary_overall.csv"))

    # By year (exclude Unknown)
    by_year_counts = (
        df.loc[df["Primary_en"] != "Unknown"]
          .groupby([year_col, "Primary_en"])
          .size()
          .rename("n")
          .reset_index()
    )
    by_year = pd.DataFrame(columns=[year_col, "Primary_en", "n", "N_year", "prop"])
    if not by_year_counts.empty:
        totals_y = by_year_counts.groupby(year_col)["n"].sum().rename("N_year").reset_index()
        by_year = by_year_counts.merge(totals_y, on=year_col, how="left")
        by_year["prop"] = by_year["n"] / by_year["N_year"]

    by_year_wide = pd.DataFrame()
    if not by_year.empty:
        by_year_wide = by_year.pivot(index=year_col, columns="Primary_en", values="prop").fillna(0.0)
        if save_csv:
            safe_save_csv(by_year_wide, os.path.join(out_dir, "primary_by_year.csv"))
    # rolling
    by_year_wide_rolling = by_year_wide.copy()
    if not by_year_wide.empty:
        rw = int(max(1, rolling_window))
        by_year_wide_rolling = by_year_wide.rolling(window=rw, min_periods=1).mean()

    # By YearBin stacked shares (exclude Unknown)
    by_bin = pd.DataFrame(columns=["YearBin", "Primary_en", "n", "N_bin", "prop"])
    by_bin_counts = (
        df.loc[df["Primary_en"] != "Unknown"]
          .groupby(["YearBin", "Primary_en"])
          .size()
          .rename("n")
          .reset_index()
    )
    if not by_bin_counts.empty:
        totals_bin = by_bin_counts.groupby("YearBin")["n"].sum().rename("N_bin").reset_index()
        by_bin = by_bin_counts.merge(totals_bin, on="YearBin", how="left")
        by_bin["prop"] = by_bin["n"] / by_bin["N_bin"]
        if save_csv:
            by_bin_pivot = by_bin.pivot(index="YearBin", columns="Primary_en", values="prop").fillna(0.0)
            safe_save_csv(by_bin_pivot, os.path.join(out_dir, "primary_by_yearbin.csv"))

    # Diversity index (1 - HHI) by year (exclude Unknown)
    diversity_rows = []
    for y, g in df.loc[df["Primary_en"] != "Unknown"].groupby(year_col):
        p = g["Primary_en"].value_counts(normalize=True)
        diversity_rows.append({"year": int(y), "diversity_1_minus_hhi": diversity_index_1_minus_hhi(p)})

    if len(diversity_rows) == 0:
        log("WARNING: No valid rows for diversity (all Unknown or empty). Skipping diversity table/figure.")
        tbl_diversity = pd.DataFrame(columns=["year", "diversity_1_minus_hhi"])
    else:
        tbl_diversity = pd.DataFrame(diversity_rows).sort_values("year")
        if save_csv:
            safe_save_csv(tbl_diversity, os.path.join(out_dir, "diversity_by_year.csv"))

    # Transparency overview (overall & by YearBin)
    if transparency_var:
        trans_counts = df["Transparency_en"].value_counts(dropna=False).rename("n").to_frame()
        trans_counts["prop"] = trans_counts["n"] / trans_counts["n"].sum()
        if save_csv:
            safe_save_csv(trans_counts, os.path.join(out_dir, "transparency_overview.csv"))

        trans_by_bin = (
            df.groupby(["YearBin", "Transparency_en"]).size().rename("n").reset_index()
        )
        totals_tb = trans_by_bin.groupby("YearBin")["n"].sum().rename("N").reset_index()
        trans_by_bin = trans_by_bin.merge(totals_tb, on="YearBin", how="left")
        trans_by_bin["prop"] = trans_by_bin["n"] / trans_by_bin["N"]
        if save_csv:
            trans_by_bin_pv = trans_by_bin.pivot(index="YearBin", columns="Transparency_en", values="prop").fillna(0.0)
            safe_save_csv(trans_by_bin_pv, os.path.join(out_dir, "transparency_by_yearbin.csv"))
    else:
        log("INFO: Transparency variable not available; skipping transparency tables.")

    # Topic × Transparency (optional)
    if transparency_var and has_topic:
        ct = pd.crosstab(df["Topic_en"], df["Transparency_en"], dropna=False)
        ct_row = ct.div(ct.sum(axis=1), axis=0).fillna(0.0)
        ct_col = ct.div(ct.sum(axis=0), axis=1).fillna(0.0)
        if save_csv:
            safe_save_csv(ct, os.path.join(out_dir, "topic_by_transparency_counts.csv"))
            safe_save_csv(ct_row, os.path.join(out_dir, "topic_by_transparency_row_norm.csv"))
            safe_save_csv(ct_col, os.path.join(out_dir, "topic_by_transparency_col_norm.csv"))
    else:
        log("INFO: Topic or Transparency not available; skipping Topic×Transparency tables.")

    # ============ Figures ============
    log("Drawing figures ...")

    # Year lines (primary)
    if not by_year_wide.empty:
        fig, ax = plt.subplots(figsize=(figsize.get("width", 9), figsize.get("height", 5)))
        (by_year_wide * 100).plot(ax=ax, linewidth=1.6)
        ax.set_title(f"{primary_var}: Share by Year (%, excl. Unknown)", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Share (%)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=primary_var)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if save_png:
            safe_save_png(fig, os.path.join(out_dir, "primary_year_lines.png"), dpi=dpi)
        if show_plots:
            plt.show()

        # rolling
        fig, ax = plt.subplots(figsize=(figsize.get("width", 9), figsize.get("height", 5)))
        (by_year_wide_rolling * 100).plot(ax=ax, linewidth=1.6)
        ax.set_title(f"{primary_var}: Share by Year (Rolling {rolling_window}, %, excl. Unknown)", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Share (%)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=primary_var)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if save_png:
            safe_save_png(fig, os.path.join(out_dir, "primary_year_lines_rolling.png"), dpi=dpi)
        if show_plots:
            plt.show()
    else:
        log("Skip primary yearly lines (no valid yearly proportions).")

    # YearBin stacked
    if 'by_bin' in locals() and not by_bin.empty:
        pv = by_bin.pivot(index="YearBin", columns="Primary_en", values="prop").fillna(0.0)
        # keep yearbin order
        order_bins = [f"{int(lo)}–{int(hi)}" for lo, hi in year_bins]
        pv = pv.reindex(order_bins)
        fig, ax = plt.subplots(figsize=(figsize.get("width", 9), figsize.get("height", 5)))
        bottom = np.zeros(len(pv))
        x = np.arange(len(pv.index))
        for col in pv.columns:
            vals = pv[col].values * 100.0
            ax.bar(x, vals, bottom=bottom, label=col)
            bottom = bottom + vals
        ax.set_xticks(x, pv.index)
        ax.set_title(f"{primary_var}: Share by Year Bin (Stacked, %, excl. Unknown)", fontsize=12)
        ax.set_ylabel("Share (%)")
        ax.set_xlabel("Year Bin")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=primary_var)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if save_png:
            safe_save_png(fig, os.path.join(out_dir, "primary_yearbin_stacked.png"), dpi=dpi)
        if show_plots:
            plt.show()
    else:
        log("Skip primary year-bin stacked bars (no valid by-bin proportions).")

    # Diversity line
    if 'tbl_diversity' in locals() and not tbl_diversity.empty:
        fig, ax = plt.subplots(figsize=(figsize.get("width", 9), figsize.get("height", 5)))
        ax.plot(tbl_diversity["year"], tbl_diversity["diversity_1_minus_hhi"], linewidth=1.8, marker="o")
        ax.set_title(f"{primary_var}: Diversity over Time (1 − HHI)", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("1 − HHI")
        ax.grid(True, axis="both", linestyle="--", alpha=0.4)
        if save_png:
            safe_save_png(fig, os.path.join(out_dir, "diversity_line.png"), dpi=dpi)
        if show_plots:
            plt.show()
    else:
        log("Skip diversity figure (empty table).")

    # Topic×Transparency heatmap
    if transparency_var and has_topic:
        # prefer row-normalized if existed
        row_norm_path = os.path.join(out_dir, "topic_by_transparency_row_norm.csv")
        if os.path.exists(row_norm_path):
            ct_row = pd.read_csv(row_norm_path, index_col=0)
        else:
            ct_row = pd.crosstab(df["Topic_en"], df["Transparency_en"], normalize="index")
        fig, ax = plt.subplots(figsize=(figsize.get("width", 9), figsize.get("height", 5)))
        sns.heatmap(ct_row, annot=True, fmt=".2f", ax=ax, cbar=True)
        ax.set_title("Topic × Transparency (Row-normalized)", fontsize=12)
        ax.set_xlabel("Transparency")
        ax.set_ylabel("Topic")
        if save_png:
            safe_save_png(fig, os.path.join(out_dir, "topic_transparency_heatmap.png"), dpi=dpi)
        if show_plots:
            plt.show()

    # Done
    log("All done.")
    log(f"Output directory: {out_dir}")
    for name in [
        "primary_overall.csv",
        "primary_by_year.csv",
        "primary_by_yearbin.csv",
        "diversity_by_year.csv",
        "transparency_overview.csv",
        "transparency_by_yearbin.csv",
        "primary_year_lines.png",
        "primary_year_lines_rolling.png",
        "primary_yearbin_stacked.png",
        "diversity_line.png",
        "topic_transparency_heatmap.png"
    ]:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            print(" -", p)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Descriptive statistics (English output) — dataset-driven",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--config", type=str, default="config_descstats.yaml",
                    help="Path to YAML config file (default: config_descstats.yaml)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config)
