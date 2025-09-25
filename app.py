# app.py
# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import re
from io import StringIO
from urllib.parse import urlparse

# ========= 設定 =========
CSV_URL = "keywords_summary4.csv"  # 直接同梱ファイルを読む。URLにしてもOK
AUTHORS_CSV = "authors_readings.csv"  # あれば使用（author, initial 列）
AUTHOR_SEP_CANDIDATES = [";", "；", "、", ",", "／", "/"]

# ========= ユーティリティ =========
@st.cache_data(show_spinner=False)
def load_csv(path_or_url: str) -> pd.DataFrame:
    return pd.read_csv(path_or_url, encoding="utf-8")

@st.cache_data(show_spinner=False)
def load_authors_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="utf-8")
        if {"author", "initial"}.issubset(df.columns):
            # 重複や空白を整理
            df["author"] = df["author"].astype(str).str.strip()
            df["initial"] = df["initial"].astype(str).str.strip()
            df = df[(df["author"] != "") & (df["initial"] != "")]
            return df.drop_duplicates(subset=["author", "initial"])
    except Exception:
        pass
    return None

def split_authors_cell(cell: str) -> list[str]:
    if not isinstance(cell, str):
        return []
    s = cell.strip()
    if not s:
        return []
    pat = r"|".join(map(re.escape, AUTHOR_SEP_CANDIDATES))
    parts = [re.sub(r"[ \u3000]+", " ", p.strip()) for p in re.split(pat, s)]
    return [p for p in parts if p]

def build_clickable_link(url: str, label: str) -> str:
    if isinstance(url, str) and url.strip():
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
    return ""

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

# ========= 画面 =========
st.set_page_config(page_title="論文検索", layout="wide")
st.title("論文検索（JBSJ）")

# データ読み込み
df = load_csv(CSV_URL)

# 想定列の補完
needed_cols = [
    "発行年","巻数","号数","開始ページ","終了ページ",
    "論文タイトル","著者","file_name","HPリンク先","PDFリンク先",
    "llm_keywords","primary_keywords","secondary_keywords",
    "対象物_top3","研究タイプ_top3","対象物_all","研究タイプ_all"
]
df = ensure_columns(df, needed_cols)

# 数値っぽい列は表示用に整数フォーマット文字列を別途用意
for c in ["発行年","巻数","号数","開始ページ","終了ページ"]:
    if c in df.columns:
        # NaN対策して整数化（できるもののみ）
        def _fmt_int(x):
            try:
                if pd.isna(x) or str(x).strip()=="":
                    return ""
                return str(int(float(x)))
            except Exception:
                return str(x)
        df[f"{c}_表示"] = df[c].apply(_fmt_int)

# クリック用リンク列（HTML）
df["_HP"] = df["HPリンク先"].apply(lambda x: build_clickable_link(x, "HP"))
df["_PDF"] = df["PDFリンク先"].apply(lambda x: build_clickable_link(x, "PDF"))

# 著者候補（authors_readings.csv があればそれを使用）
authors_df = load_authors_csv(AUTHORS_CSV)
use_external_authors = authors_df is not None

# === フィルタ（メインエリア） ===
st.subheader("検索条件")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    kw = st.text_input("フリーワード（タイトル / 要約 / キーワード）", "")
with col2:
    years = sorted([v for v in set(df["発行年"]) if pd.notna(v)])
    year_sel = st.multiselect("発行年", [str(int(float(y))) if str(y).replace('.0','').isdigit() else str(y) for y in years])
with col3:
    targets_sel = st.multiselect("対象物（top3）", sorted([x for x in set(sum([str(v).split(";") for v in df["対象物_top3"].fillna("")], [])) if x.strip()]))
with col4:
    types_sel = st.multiselect("研究タイプ（top3）", sorted([x for x in set(sum([str(v).split(";") for v in df["研究タイプ_top3"].fillna("")], [])) if x.strip()]))

# 著者フィルタ（頭文字 → 候補 → 選択）
st.markdown("#### 著者で絞り込み")
colA, colB = st.columns([1, 3])

if use_external_authors:
    # 頭文字集合（日本語：あ/か/…/わ、英語：A〜Z）
    initials_order = ["あ","か","さ","た","な","は","ま","や","ら","わ"] + [chr(c) for c in range(ord("A"), ord("Z")+1)]
    initials_in_data = [i for i in initials_order if i in set(authors_df["initial"])]

    with colA:
        init_selected = st.radio("頭文字", initials_in_data, horizontal=True)
    # その頭文字の著者候補
    cand_authors = authors_df[authors_df["initial"]==init_selected]["author"].drop_duplicates().tolist()
    with colB:
        author_sel = st.multiselect("著者（頭文字で候補絞り込み済）", cand_authors)
else:
    st.caption("authors_readings.csv が見つからないため、著者はセル分割の簡易推定です。")
    # 著者セルを分割して一意化
    all_auths = set()
    for cell in df["著者"].fillna("").astype(str):
        all_auths.update(split_authors_cell(cell))
    cand_authors = sorted(all_auths)
    with colB:
        author_sel = st.multiselect("著者", cand_authors)

st.divider()

# ====== フィルタ適用 ======
f = df.copy()

# 発行年
if year_sel:
    f = f[f["発行年"].astype(str).isin(set(year_sel))]

# 対象物 / 研究タイプ（top3）
def contains_any(cell: str, terms: list[str]) -> bool:
    if not terms:
        return True
    s = str(cell or "")
    return any(t.strip() and t.strip() in s for t in terms)

if targets_sel:
    f = f[f["対象物_top3"].apply(lambda x: contains_any(x, targets_sel))]
if types_sel:
    f = f[f["研究タイプ_top3"].apply(lambda x: contains_any(x, types_sel))]

# 著者
if author_sel:
    def _has_author(cell: str) -> bool:
        # セル内に author_sel のいずれかが含まれるか
        authors = split_authors_cell(cell)
        return any(a in authors for a in author_sel)
    f = f[f["著者"].apply(_has_author)]

# フリーワード（タイトル/要約/キーワードでざっくり）
if kw.strip():
    pat = re.compile(re.escape(kw.strip()), re.IGNORECASE)
    def _hit(row) -> bool:
        hay = " ".join([
            str(row.get("論文タイトル","")),
            str(row.get("llm_keywords","")),
            str(row.get("primary_keywords","")),
            str(row.get("secondary_keywords","")),
        ])
        return bool(pat.search(hay))
    f = f[f.apply(_hit, axis=1)]

# ====== 表示 ======
st.subheader("検索結果")

# 表示用の列順・ラベル調整
display_cols = [
    "発行年_表示","巻数_表示","号数_表示","開始ページ_表示","終了ページ_表示",
    "論文タイトル","著者","_HP","_PDF",
    "対象物_top3","研究タイプ_top3","llm_keywords","primary_keywords","secondary_keywords",
]
display_names = {
    "発行年_表示":"発行年","巻数_表示":"巻数","号数_表示":"号数",
    "開始ページ_表示":"開始ページ","終了ページ_表示":"終了ページ",
    "論文タイトル":"論文タイトル","著者":"著者","_HP":"HP","_PDF":"PDF",
    "対象物_top3":"対象物(top3)","研究タイプ_top3":"研究タイプ(top3)",
    "llm_keywords":"llm_keywords",
    "primary_keywords":"primary_keywords",
    "secondary_keywords":"secondary_keywords",
}

f_show = f.copy()
for c in display_cols:
    if c not in f_show.columns:
        f_show[c] = ""

f_show = f_show[display_cols].rename(columns=display_names)

# HTMLテーブルでリンクを有効化
st.markdown(
    f_show.to_html(escape=False, index=False),
    unsafe_allow_html=True
)

st.caption(f"件数: {len(f_show)} / 全{len(df)}")