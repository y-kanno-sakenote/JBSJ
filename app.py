# -*- coding: utf-8 -*-
"""
app.py
- メインCSV: keywords_summary4.csv（論文DB）
- 読みCSV:   authors_readings.csv（author, reading, initial）
著者フィルタは論文CSVの「著者」列から抽出した候補を使う（autocomplete有効）。
authors_readings.csv は候補の正規化（表記ゆれ吸収）にのみ使用。
"""

import re
import pandas as pd
import streamlit as st

# ======================
# 設定
# ======================
MAIN_CSV_PATH = "keywords_summary4.csv"
READING_CSV_PATH = "authors_readings.csv"   # author, reading, initial

# ======================
# ユーティリティ
# ======================
_SEP_PAT = re.compile(r"[;；、,/／]+")

def split_authors_cell(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = [p.strip() for p in _SEP_PAT.split(cell) if p.strip()]
    return parts

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    # 文字化け防止で複数エンコ試行
    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

@st.cache_data
def build_author_vocab_from_df(df: pd.DataFrame, readings_df: pd.DataFrame | None) -> list[str]:
    """
    論文CSVの「著者」列から候補を抽出。
    readings があれば author 列と突合 → 同一表記の重複を解消する程度に使う。
    """
    uniq = set()
    if "著者" in df.columns:
        for v in df["著者"].fillna("").astype(str):
            for a in split_authors_cell(v):
                if a:
                    uniq.add(a)

    # readings がある場合は author の前後空白・全角半角差を緩く吸収
    if readings_df is not None and "author" in readings_df.columns:
        norm_map = {}
        for a in readings_df["author"].dropna().astype(str):
            key = re.sub(r"\s+", "", a)  # 空白除去でキー化
            norm_map.setdefault(key, a)  # 代表表記保持

        normalized = set()
        for a in uniq:
            key = re.sub(r"\s+", "", a)
            normalized.add(norm_map.get(key, a))
        uniq = normalized

    # 表示は50音→英字の順に近い感じで
    def sort_key(x: str):
        # 先頭がASCIIなら後ろへ
        is_ascii = bool(re.match(r"[A-Za-z]", x))
        return (is_ascii, x)

    return sorted(uniq, key=sort_key)

def filter_by_authors(df: pd.DataFrame, selected_authors: list[str]) -> pd.DataFrame:
    if not selected_authors:
        return df
    # セル内の著者に一人でも一致したら残す
    mask = df["著者"].fillna("").astype(str).apply(
        lambda s: any(a in s for a in selected_authors)
    )
    return df[mask]

# ======================
# アプリ本体
# ======================
st.set_page_config(page_title="論文検索", layout="wide")
st.title("論文検索システム")

# データ読込
df = load_csv(MAIN_CSV_PATH)
readings_df = None
try:
    readings_df = load_csv(READING_CSV_PATH)
except Exception:
    readings_df = None

# 欄の存在保証（落ちても動くように）
for col in ["発行年", "論文タイトル", "著者"]:
    if col not in df.columns:
        df[col] = ""

st.subheader("検索フィルタ（メイン画面）")

# 著者（autocomplete）
author_vocab = build_author_vocab_from_df(df, readings_df)
selected_authors = st.multiselect(
    "著者（オートコンプリート）",
    options=author_vocab,
    help="一部を入力すると候補が絞り込まれます。複数選択可。"
)

# 年
years = sorted(df["発行年"].dropna().unique().tolist())
selected_years = st.multiselect("発行年", options=years)

# フリーワード（タイトル）
kw = st.text_input("タイトル キーワード検索", placeholder="例: 乳酸菌 / 山田錦 など")

# フィルタ適用
filtered = df.copy()
filtered = filter_by_authors(filtered, selected_authors)
if selected_years:
    filtered = filtered[filtered["発行年"].isin(selected_years)]
if kw:
    filtered = filtered[filtered["論文タイトル"].fillna("").str.contains(kw, case=False, na=False)]

st.subheader(f"検索結果（{len(filtered)}件）")
st.dataframe(filtered, use_container_width=True)