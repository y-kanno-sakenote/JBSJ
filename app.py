# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re
from functools import lru_cache

st.set_page_config(page_title="論文検索（JBSJ）", layout="wide")

# =========================
# 設定
# =========================
DEFAULT_DATA_PATH = "keywords_summary4.csv"     # 本体データ（あなたの新CSV）
AUTHORS_CSV_PATH  = "authors_readings.csv"      # 著者リスト（author, reading, initial）

# 対象物・研究タイプは top3 を優先、無ければ all にフォールバック
TARGET_COL_PREFS = ["対象物_top3", "対象物_all"]
TYPE_COL_PREFS   = ["研究タイプ_top3", "研究タイプ_all"]

# =========================
# ユーティリティ
# =========================
@st.cache_data(show_spinner=False)
def load_csv(path_or_url: str) -> pd.DataFrame:
    # 文字化け回避で複数エンコーディングを順に試す
    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis"):
        try:
            df = pd.read_csv(path_or_url, encoding=enc)
            return df
        except Exception:
            continue
    return pd.read_csv(path_or_url)

@st.cache_data(show_spinner=False)
def load_authors_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # 想定列: author, reading, initial
    # なければ空で返す
    expected = {"author","reading","initial"}
    if not expected.issubset(set(df.columns)):
        return pd.DataFrame(columns=["author","reading","initial"])
    # 欠損を空文字に
    for c in ["author","reading","initial"]:
        df[c] = df[c].fillna("").astype(str)
    # 重複除去
    df = df.drop_duplicates(subset=["author"]).reset_index(drop=True)
    return df

def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# 五十音の行→その行の頭文字一覧
_GOJUON = {
    "あ": list("あいうえお"),
    "か": list("かきくけこがぎぐげご"),
    "さ": list("さしすせそざじずぜぞ"),
    "た": list("たちつてとだぢづでど"),
    "な": list("なにぬねの"),
    "は": list("はひふへほばびぶべぼぱぴぷぺぽ"),
    "ま": list("まみむめも"),
    "や": list("やゆよ"),
    "ら": list("らりるれろ"),
    "わ": list("わをん"),
}

def get_subinitials(row_label: str) -> list[str]:
    return _GOJUON.get(row_label, [])

def split_semi(s: str) -> list[str]:
    if not isinstance(s, str):
        return []
    return [x.strip() for x in re.split(r"[;；]", s) if x and x.strip()]

# =========================
# UI: データ読み込み
# =========================
st.title("論文検索（JBSJ）")

with st.sidebar:
    st.subheader("データ読み込み")
    csv_path = st.text_input("データCSVパス/URL", value=DEFAULT_DATA_PATH)
    authors_path = st.text_input("著者CSV（authors_readings.csv）", value=AUTHORS_CSV_PATH)

df = load_csv(csv_path)
authors_df = load_authors_csv(authors_path)

if df.empty:
    st.error("データCSVが読み込めませんでした。パス/URLをご確認ください。")
    st.stop()

# 列存在チェック（対象物/研究タイプ）
target_col = pick_first_existing_column(df, TARGET_COL_PREFS)
type_col   = pick_first_existing_column(df, TYPE_COL_PREFS)

# =========================
# UI: 検索フィルタ
# =========================
st.sidebar.subheader("検索フィルタ")

# キーワード（全文/タイトル/キーワード列に対して簡易AND）
query = st.sidebar.text_input("キーワード（スペース区切りでAND）", value="").strip()

# 対象物・研究タイプ（top3優先でプルダウン）
if target_col:
    target_all = sorted({t for row in df[target_col].fillna("").astype(str) for t in split_semi(row)} - {""})
    sel_target = st.sidebar.multiselect("対象物で絞り込み（top3優先）", target_all, default=[])
else:
    sel_target = []

if type_col:
    type_all = sorted({t for row in df[type_col].fillna("").astype(str) for t in split_semi(row)} - {""})
    sel_type = st.sidebar.multiselect("研究タイプで絞り込み（top3優先）", type_all, default=[])
else:
    sel_type = []

# ===== 著者フィルタ（あかさたな + 行内頭文字 + オートコンプリート） =====
st.sidebar.markdown("---")
st.sidebar.markdown("#### 著者フィルタ（段階絞り込み）")

row_labels = ["あ","か","さ","た","な","は","ま","や","ら","わ","A〜Z","その他"]
selected_row = st.sidebar.radio("頭文字行", options=row_labels, horizontal=True, index=0)

# 行の候補作成
filtered_authors_df = authors_df.copy()
if selected_row == "A〜Z":
    # 英字A〜Zの initial 想定
    filtered_authors_df = filtered_authors_df[filtered_authors_df["initial"].str.match(r"^[A-Z]$")]
elif selected_row == "その他":
    # 既存initialが「その他」
    filtered_authors_df = filtered_authors_df[filtered_authors_df["initial"] == "その他"]
else:
    filtered_authors_df = filtered_authors_df[filtered_authors_df["initial"] == selected_row]

# 二段階目（その行の中の頭文字 selectbox）
suboptions = get_subinitials(selected_row)
selected_sub = None
if suboptions:
    selected_sub = st.sidebar.selectbox("行内の頭文字", ["（すべて）"] + suboptions, index=0)
    if selected_sub != "（すべて）":
        # reading がその仮名で始まる著者に限定
        mask = filtered_authors_df["reading"].str.startswith(selected_sub, na=False)
        filtered_authors_df = filtered_authors_df[mask]

# 最終候補（authorのユニークリスト）
author_candidates = sorted(filtered_authors_df["author"].unique().tolist())
selected_authors = st.sidebar.multiselect("著者で絞り込み", options=author_candidates, default=[])

# =========================
# データ絞り込みロジック
# =========================
view = df.copy()

# キーワード（AND）
if query:
    terms = [t for t in re.split(r"\s+", query) if t]
    for t in terms:
        # タイトル・要約・キーワード的な列にヒットさせる（存在すれば）
        hits = pd.Series(False, index=view.index)
        for col in ["論文タイトル", "llm_keywords", "primary_keywords", "secondary_keywords"]:
            if col in view.columns:
                hits = hits | view[col].fillna("").astype(str).str.contains(re.escape(t), case=False, na=False)
        # 本文列があれば（任意）
        if "__text__" in view.columns:
            hits = hits | view["__text__"].fillna("").astype(str).str.contains(re.escape(t), case=False, na=False)
        view = view[hits]

# 対象物
if sel_target and target_col:
    mask = pd.Series(False, index=view.index)
    vc = view[target_col].fillna("").astype(str)
    for t in sel_target:
        mask = mask | vc.str.contains(rf"(^|[;；])\s*{re.escape(t)}\s*($|[;；])")
    view = view[mask]

# 研究タイプ
if sel_type and type_col:
    mask = pd.Series(False, index=view.index)
    vc = view[type_col].fillna("").astype(str)
    for t in sel_type:
        mask = mask | vc.str.contains(rf"(^|[;；])\s*{re.escape(t)}\s*($|[;；])")
    view = view[mask]

# 著者（部分一致 OR；セル内セパレータに依らず）
if selected_authors and "著者" in view.columns:
    mask = pd.Series(False, index=view.index)
    col = view["著者"].fillna("").astype(str)
    for a in selected_authors:
        mask = mask | col.str.contains(re.escape(a))
    view = view[mask]

st.write(f"ヒット件数: {len(view):,}")

# =========================
# 表示
# =========================
# よく見る列を並べ替え（存在するものだけ）
prefer_cols = [
    "No.","発行年","巻数","号数","開始ページ","終了ページ",
    "論文タイトル","著者","file_name","HPリンク先","PDFリンク先",
    "llm_keywords","primary_keywords","secondary_keywords",
    "対象物_top3","研究タイプ_top3","対象物_all","研究タイプ_all"
]
cols = [c for c in prefer_cols if c in view.columns] + [c for c in view.columns if c not in prefer_cols]
st.dataframe(view[cols], use_container_width=True, height=600)

# クリックしやすいリンク化（オプション）
st.caption("※ HPリンク先 / PDFリンク先 は列をコピーしてブラウザで開いてください。")