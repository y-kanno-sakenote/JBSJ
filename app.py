# app.py
# -*- coding: utf-8 -*-
import io
import re
from typing import List, Set

import pandas as pd
import streamlit as st

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="論文検索（JBSJ）", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv(path_or_url: str) -> pd.DataFrame:
    # Excel/Numbers対策で utf-8-sig を優先
    try:
        df = pd.read_csv(path_or_url, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path_or_url, encoding="utf-8")
    return df

def normalize_semicolon_list(s: str) -> List[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    # 全角/半角セミコロン、カンマ等を一応吸収
    parts = re.split(r"[;；,、]\s*", str(s))
    # 空要素除去 & 前後空白除去
    parts = [p.strip() for p in parts if p and p.strip()]
    # 重複除去（順序維持）
    seen: Set[str] = set()
    uniq: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def explode_unique_options(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    bag: Set[str] = set()
    for v in df[col].fillna(""):
        for item in normalize_semicolon_list(v):
            bag.add(item)
    return sorted(bag)

def contains_any_labels(cell: str, wanted: List[str]) -> bool:
    if not wanted:
        return True
    labels = set(normalize_semicolon_list(cell))
    return any(w in labels for w in wanted)

def text_contains(s: str, q: str) -> bool:
    if not q:
        return True
    return q.lower() in str(s).lower()

# =========================
# データ読み込み
# =========================
st.sidebar.header("データ読み込み")
# デフォルトは同梱CSV名（必要に応じて変更）
default_path = "keywords_summary4.csv"
csv_path = st.sidebar.text_input("CSV パス/URL", value=default_path, help="例: keywords_summary4.csv など")
df = load_csv(csv_path)

# No. が None/NaN の行を非表示。数字化可能な行だけ残す。
if "No." in df.columns:
    # 数値化（失敗は NaN）
    df["No._num"] = pd.to_numeric(df["No."], errors="coerce")
    df = df[df["No._num"].notna()].copy()
    df["No."] = df["No._num"].astype(int)
    df = df.drop(columns=["No._num"]).sort_values("No.").reset_index(drop=True)

# 年/巻/号は可能なら整数へ（表示を綺麗に）
for c in ["発行年", "巻数", "号数", "開始ページ", "終了ページ"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

st.title("Journal of the Brewing Society of Japan — 論文検索")

# =========================
# サイドバー：検索・絞り込み（top3列を採用）
# =========================
st.sidebar.header("絞り込み（top3列を使用）")

# 対象物 / 研究タイプ は *_top3 列を使用
target_col = "対象物_top3" if "対象物_top3" in df.columns else "対象物"
type_col   = "研究タイプ_top3" if "研究タイプ_top3" in df.columns else "研究タイプ"

target_opts = explode_unique_options(df, target_col)
type_opts   = explode_unique_options(df, type_col)

sel_targets = st.sidebar.multiselect("対象物で絞り込み（top3）", target_opts, default=[])
sel_types   = st.sidebar.multiselect("研究タイプで絞り込み（top3）", type_opts, default=[])

# 年度レンジ
if "発行年" in df.columns and df["発行年"].notna().any():
    y_min = int(df["発行年"].min())
    y_max = int(df["発行年"].max())
    sel_year = st.sidebar.slider("発行年の範囲", min_value=y_min, max_value=y_max, value=(y_min, y_max))
else:
    sel_year = None

# フリーテキスト検索（タイトル/著者/キーワード）
q = st.sidebar.text_input("キーワード検索（タイトル/著者/キーワードに対して）", value="")

# =========================
# 実フィルタ適用
# =========================
df_filtered = df.copy()

# 対象物フィルタ（top3列）
if sel_targets and target_col in df_filtered.columns:
    df_filtered = df_filtered[df_filtered[target_col].apply(lambda x: contains_any_labels(str(x), sel_targets))]

# 研究タイプフィルタ（top3列）
if sel_types and type_col in df_filtered.columns:
    df_filtered = df_filtered[df_filtered[type_col].apply(lambda x: contains_any_labels(str(x), sel_types))]

# 年レンジ
if sel_year and "発行年" in df_filtered.columns:
    y0, y1 = sel_year
    df_filtered = df_filtered[df_filtered["発行年"].between(y0, y1, inclusive="both")]

# テキスト検索（タイトル/著者/llm_keywords/primary/secondary）
text_cols = [c for c in ["論文タイトル", "著者", "llm_keywords", "primary_keywords", "secondary_keywords"] if c in df_filtered.columns]
if q and text_cols:
    mask = False
    for c in text_cols:
        mask = (mask | df_filtered[c].astype(str).str.contains(q, case=False, na=False)) if isinstance(mask, pd.Series) else df_filtered[c].astype(str).str.contains(q, case=False, na=False)
    df_filtered = df_filtered[mask]

st.write(f"検索ヒット: **{len(df_filtered):,} / {len(df):,}** 件")

# =========================
# 表示用の列構成（top3列を優先して見せる）
# =========================
display_cols = []
preferred_order = [
    "No.","発行年","巻数","号数","開始ページ","終了ページ",
    "論文タイトル","著者","file_name","HPリンク先","PDFリンク先",
    "llm_keywords","primary_keywords","secondary_keywords",
    "対象物_top3","研究タイプ_top3",
    "対象物","研究タイプ",
    "対象物_all","研究タイプ_all",
    "対象物_根拠","研究タイプ_根拠",
]
for c in preferred_order:
    if c in df_filtered.columns:
        display_cols.append(c)

st.dataframe(
    df_filtered[display_cols],
    use_container_width=True,
    height=560,
)

# =========================
# ダウンロード（絞り込み結果）
# =========================
st.subheader("絞り込み結果のダウンロード")
buf = io.StringIO()
# Excel/Numbersでも崩れにくい BOM 付きを選択
df_filtered.to_csv(buf, index=False, encoding="utf-8")
b = buf.getvalue().encode("utf-8-sig")
st.download_button(
    "CSV をダウンロード (UTF-8 BOM)",
    data=b,
    file_name="filtered.csv",
    mime="text/csv",
)