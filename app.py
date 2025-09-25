# -*- coding: utf-8 -*-
"""
論文検索（統一UI版：お気に入りにタグを“表で直接入力”）
＋ 著者フィルタを authors_readings.csv ベースのオートコンプリートに対応
"""

import io, re, time
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

# -------------------- ページ設定 --------------------
st.set_page_config(page_title="論文検索（統一UI版）", layout="wide")

# -------------------- 定数 --------------------
KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]
BASE_COLS = [
    "No.","相対PASS","発行年","巻数","号数","開始ページ","終了ページ",
    "論文タイトル","著者","file_name","HPリンク先","PDFリンク先",
    "対象物_top3","研究タイプ_top3",
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
]
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","アミノ酸・タンパク質","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

# -------------------- ユーティリティ --------------------
def norm_space(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_key(s: str) -> str:
    return norm_space(s).lower()

AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")
def split_authors(cell):
    if not cell: return []
    return [w.strip() for w in AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def split_multi(s):
    if not s: return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()]

def tokens_from_query(q):
    q = norm_key(q)
    return [t for t in re.split(r"[ ,，、；;　]+", q) if t]

def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding="utf-8")

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def haystack(row, include_fulltext: bool):
    parts = [
        str(row.get("論文タイトル","")),
        str(row.get("著者","")),
        str(row.get("file_name","")),
        " ".join(str(row.get(c,"")) for c in KEY_COLS if c in row),
    ]
    if include_fulltext and "pdf_text" in row:
        parts.append(str(row.get("pdf_text","")))
    return norm_key(" \n ".join(parts))

def to_int_or_none(x):
    try: return int(str(x).strip())
    except Exception:
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else None

def order_by_template(values, template):
    vs = list(dict.fromkeys(values))
    tmpl_set = set(template)
    head = [v for v in template if v in vs and "その他" not in v]
    mid  = sorted([v for v in vs if v not in tmpl_set and "その他" not in v])
    tail = [v for v in template if "その他" in v and v in vs] + \
           [v for v in vs if "その他" in v and v not in template]
    return head + mid + tail

def make_visible_cols(df: pd.DataFrame) -> list[str]:
    base_hide = {"相対PASS", "終了ページ", "file_path", "num_pages", "file_name"}
    cols = [str(c) for c in df.columns]
    hide = set(c for c in cols if c in base_hide)
    if "llm_keywords" in cols:
        idx = cols.index("llm_keywords")
        hide.update(cols[idx:])
    return [c for c in cols if c not in hide]

def make_row_id(row):
    no = str(row.get("No.", "")).strip()
    if no and no.lower() not in {"none", "nan"}:
        return f"NO:{no}"
    ttl = str(row.get("論文タイトル", "")).strip()
    yr  = str(row.get("発行年", "")).strip()
    return f"T:{ttl}|Y:{yr}"

# -------------------- データ読み込み --------------------
st.title("醸造協会誌　論文検索")

DEMO_CSV_PATH = Path("data/keywords_summary4.csv")
AUTHORS_CSV_PATH = Path("data/authors_readings.csv")  # 著者リストCSV

SECRET_URL = st.secrets.get("GSHEET_CSV_URL", "")

@st.cache_data(ttl=600, show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame:
    return ensure_cols(pd.read_csv(path, encoding="utf-8"))

df = None
if DEMO_CSV_PATH.exists():
    df = load_local_csv(DEMO_CSV_PATH)

# 著者候補のロード（authors_readings.csv 主軸）
authors_df = None
authors_all = []
if AUTHORS_CSV_PATH.exists():
    authors_df = load_local_csv(AUTHORS_CSV_PATH)
    if "author" in authors_df.columns:
        authors_all = sorted(authors_df["author"].dropna().unique().tolist())

# -------------------- 年・巻・号フィルタ --------------------
st.subheader("年・巻・号フィルタ")
year_vals = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
ymin_all, ymax_all = int(year_vals.min()), int(year_vals.max())

c_y, c_v, c_i = st.columns([1, 1, 1])
with c_y:
    y_from, y_to = st.slider("発行年（範囲）", min_value=ymin_all, max_value=ymax_all, value=(ymin_all, ymax_all))
with c_v:
    vols_sel = st.multiselect("巻", sorted({to_int_or_none(v) for v in df["巻数"] if pd.notna(v)}))
with c_i:
    issues_sel = st.multiselect("号", sorted({to_int_or_none(v) for v in df["号数"] if pd.notna(v)}))

# -------------------- 著者・対象物・研究タイプフィルタ --------------------
st.subheader("検索フィルタ")
c_a, c_tg, c_tp = st.columns([1.2, 1.2, 1.2])
with c_a:
    authors_sel = st.multiselect("著者（オートコンプリート）", authors_all, default=[], placeholder="著者名を入力…")
with c_tg:
    raw_targets = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    targets_sel = st.multiselect("対象物", order_by_template(list(raw_targets), TARGET_ORDER))
with c_tp:
    raw_types = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    types_sel = st.multiselect("研究タイプ", order_by_template(list(raw_types), TYPE_ORDER))

# -------------------- キーワード検索 --------------------
c_kw1, c_kw2, c_kw3 = st.columns([3, 1, 1])
with c_kw1:
    kw_query = st.text_input("キーワード（空白/カンマ区切り）", value="")
with c_kw2:
    kw_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True)
with c_kw3:
    include_fulltext = st.checkbox("本文も検索（pdf_text）", value=True)

# -------------------- フィルタ適用 --------------------
def apply_filters(_df: pd.DataFrame) -> pd.DataFrame:
    df2 = _df.copy()
    y = pd.to_numeric(df2["発行年"], errors="coerce")
    df2 = df2[(y >= y_from) & (y <= y_to) | y.isna()]
    if vols_sel: df2 = df2[df2["巻数"].map(to_int_or_none).isin(set(vols_sel))]
    if issues_sel: df2 = df2[df2["号数"].map(to_int_or_none).isin(set(issues_sel))]
    if authors_sel:
        sel = {norm_key(a) for a in authors_sel}
        def hit_author(v): return any(norm_key(x) in sel for x in split_authors(v))
        df2 = df2[df2["著者"].apply(hit_author)]
    if targets_sel:
        t_norm = [norm_key(t) for t in targets_sel]
        df2 = df2[df2["対象物_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    if types_sel:
        t_norm = [norm_key(t) for t in types_sel]
        df2 = df2[df2["研究タイプ_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    toks = tokens_from_query(kw_query)
    if toks:
        def hit_kw(row):
            hs = haystack(row, include_fulltext=include_fulltext)
            return all(t in hs for t in toks) if kw_mode == "AND" else any(t in hs for t in toks)
        df2 = df2[df2.apply(hit_kw, axis=1)]
    return df2

filtered = apply_filters(df)

# -------------------- 検索結果 --------------------
st.markdown("### 検索結果")
st.caption(f"{len(filtered)} / {len(df)} 件")
st.dataframe(filtered[make_visible_cols(filtered)], use_container_width=True)