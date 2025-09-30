# -*- coding: utf-8 -*-
"""
論文検索（統一UI版：お気に入りにタグを“表で直接入力”）＋ 分析タブ

機能（検索タブ：従来どおり）
- 発行年レンジ、巻・号（複数選択）、著者（正規化・複数選択/読みの頭文字ラジオ＋オートコンプリート）、
  対象物/研究タイプ（部分一致・複数選択）
- キーワード AND/OR 検索（空白/カンマ区切り）
- 検索結果テーブル（不要列の非表示、HP/PDF のリンク化、★でお気に入り）
- お気に入り一覧（常設・★で解除/追加）、tags 列を直接編集、❌ 全て外す
- summaries.csv の summary を「著者」の右列に表示

機能（分析タブ：新規）
- 対象データ：検索結果 or 全件 を選択
- 著者共起ネットワーク（同一論文の著者同士をエッジ）
- 中心性ランキング（Degree / Betweenness / Eigenvector）
- PyVis によるインタラクティブ可視化（未インストール時は matplotlib フォールバック）
"""

import io, re, time, math
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

# 追加：分析用
import itertools
import networkx as nx
import numpy as np

# PyVis は任意
try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False

# -------------------- ページ設定 --------------------
st.set_page_config(page_title="論文検索（統一UI＋分析）", layout="wide")

# -------------------- コントラスト（著者ドロップダウン強化版：従来CSS） --------------------
st.markdown(
    """
    <style>
    .stTextInput input, .stNumberInput input, textarea {
      background-color: #e0e0e0 !important;
      color: #000 !important;
      border: 1px solid #666 !important;
      border-radius: 6px !important;
      padding: 4px 8px !important;
    }
    .stMultiSelect div[data-baseweb="select"],
    .stSelectbox  div[data-baseweb="select"] {
      background-color: #e0e0e0 !important;
      border: 1px solid #666 !important;
      border-radius: 6px !important;
    }
    div[data-baseweb="select"] > div { background: transparent !important; }
    div[data-baseweb="select"] span { color: #000 !important; }
    div[data-baseweb="select"] svg  { color: #000 !important; fill: #000 !important; }
    div[data-baseweb="tag"] {
      background: #d5d5d5 !important;
      color: #000 !important;
      border-radius: 12px !important;
    }
    input:focus, textarea:focus,
    .stTextInput input:focus, .stNumberInput input:focus {
      border: 2px solid #1a73e8 !important;
      box-shadow: 0 0 4px #1a73e8 !important;
      outline: none !important;
    }
    .stMultiSelect div[data-baseweb="select"]:focus-within,
    .stSelectbox  div[data-baseweb="select"]:focus-within {
      border: 2px solid #1a73e8 !important;
      box-shadow: 0 0 4px #1a73e8 !important;
    }
    .stMultiSelect input:focus,
    .stSelectbox  input:focus {
      border: none !important;
      box-shadow: none !important;
      outline: none !important;
    }
    ul[role="listbox"] {
      background: #f5f5f5 !important;
      border: 1px solid #666 !important;
      max-height: 70vh !important;
      min-height: 360px !important;
      overflow-y: auto !important;
      padding-right: 6px !important;
      scrollbar-width: auto;
      scrollbar-color: #555 #e9e9e9;
    }
    li[role="option"] {
      padding: 8px 12px !important;
      line-height: 1.4 !important;
      font-size: 0.95rem !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
      background: #e0e0e0 !important;
      color: #000 !important;
    }
    ul[role="listbox"]::-webkit-scrollbar { width: 16px; }
    ul[role="listbox"]::-webkit-scrollbar-track { background: #e9e9e9; border-radius: 8px; }
    ul[role="listbox"]::-webkit-scrollbar-thumb {
      background: #555; border-radius: 8px; border: 3px solid #e9e9e9;
    }
    ul[role="listbox"]::-webkit-scrollbar-thumb:hover { background: #333; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- 定数（従来） --------------------
KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
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

# -------------------- ユーティリティ（従来） --------------------
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

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_author_candidates(df: pd.DataFrame):
    rep = {}
    for v in df.get("著者", pd.Series(dtype=str)).fillna(""):
        for name in split_authors(v):
            k = norm_key(name)
            if k and k not in rep:
                rep[k] = name
    return [rep[k] for k in sorted(rep.keys())]

def haystack(row):
    parts = [
        str(row.get("論文タイトル","")),
        str(row.get("著者","")),
        str(row.get("file_name","")),
        " ".join(str(row.get(c,"")) for c in KEY_COLS if c in row),
    ]
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
    tail = [v for v in template if v in vs and "その他" in v] + \
           [v for v in vs if ("その他" in v and v not in template)]
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

# -------------------- データ読み込み（従来） --------------------
st.title("醸造協会誌　論文検索 / 分析")

DEMO_CSV_PATH = Path("data/keywords_summary5.csv")   # メインCSV
SUMMARY_CSV_PATH = Path("data/summaries.csv")         # summary（file_name, summary）
AUTHORS_CSV_PATH = Path("data/authors_readings.csv")  # 著者読み（author, reading）
SECRET_URL = st.secrets.get("GSHEET_CSV_URL", "")

@st.cache_data(ttl=600, show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame:
    return ensure_cols(pd.read_csv(path, encoding="utf-8"))

@st.cache_data(ttl=600, show_spinner=False)
def load_url_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding="utf-8")

@st.cache_data(ttl=600, show_spinner=False)
def load_summaries(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists(): return None
        df_s = pd.read_csv(path, encoding="utf-8")
        df_s.columns = [str(c).strip() for c in df_s.columns]
        if not {"file_name", "summary"}.issubset(df_s.columns): return None
        return df_s[["file_name", "summary"]]
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def load_authors_readings(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists(): return None
        adf = pd.read_csv(path, encoding="utf-8")
        adf.columns = [str(c).strip() for c in adf.columns]
        if not {"author", "reading"}.issubset(adf.columns): return None
        adf["author"]  = adf["author"].astype(str).str.strip()
        adf["reading"] = adf["reading"].astype(str).str.strip()
        adf = adf[(adf["author"]!="") & (adf["reading"]!="")]
        adf = adf.drop_duplicates(subset=["reading"], keep="first")
        return adf
    except Exception:
        return None

with st.sidebar:
    st.header("データ読み込み")
    st.caption("※ まずはデモCSVを自動ロード。URL/ファイル指定で上書きできます。")
    use_demo = st.toggle("デモCSVを自動ロードする", value=True)
    url = st.text_input("公開CSVのURL（Googleスプレッドシート output=csv）", value=SECRET_URL)
    up  = st.file_uploader("CSVをローカルから読み込み", type=["csv"])
    load_clicked = st.button("読み込み（URL/ファイルを優先）", type="primary", key="load_btn")

df = None; err = None
try:
    if load_clicked:
        if up is not None:
            df = ensure_cols(pd.read_csv(up, encoding="utf-8")); st.toast("ローカルCSVを読み込みました")
        elif url.strip():
            df = load_url_csv(url.strip()); st.toast("URLのCSVを読み込みました")
        else:
            st.warning("URL または CSV を指定してください。")
    elif use_demo and DEMO_CSV_PATH.exists():
        df = load_local_csv(DEMO_CSV_PATH); st.caption(f"✅ デモCSVを自動ロード中: {DEMO_CSV_PATH}")
    elif SECRET_URL:
        df = load_url_csv(SECRET_URL); st.caption("✅ SecretsのURLから自動ロード中")
except Exception as e:
    err = e

if df is None:
    if err: st.error(f"読み込みエラー: {err}")
    st.info("左のサイドバーで CSV を指定するか、デモCSVを有効にしてください。")
    st.stop()

# summary マージ
sum_df = load_summaries(SUMMARY_CSV_PATH)
if sum_df is not None:
    df = df.merge(sum_df, on="file_name", how="left")

# 著者読み候補（全体）を先にロードして session に保持（検索/分析で共用）
if "author_candidates" not in st.session_state:
    st.session_state.author_candidates = load_authors_readings(AUTHORS_CSV_PATH)

# -------------------- 共通：フィルタ関数（従来ロジック） --------------------
def apply_filters_for_current_state(_df: pd.DataFrame) -> pd.DataFrame:
    # 下記キーは検索タブで設定（ここでは存在しなければデフォルト扱い）
    y_from = st.session_state.get("y_from", None)
    y_to   = st.session_state.get("y_to", None)
    vols_sel   = st.session_state.get("vols_sel", [])
    issues_sel = st.session_state.get("issues_sel", [])
    authors_sel = st.session_state.get("authors_sel", [])
    targets_sel = st.session_state.get("targets_sel", [])
    types_sel   = st.session_state.get("types_sel", [])
    kw_query = st.session_state.get("kw_query", "")
    kw_mode  = st.session_state.get("kw_mode", "OR")

    df2 = _df.copy()
    # 年
    if y_from is not None and y_to is not None and "発行年" in df2.columns:
        y = pd.to_numeric(df2["発行年"], errors="coerce")
        df2 = df2[(y >= y_from) & (y <= y_to) | y.isna()]
    # 巻・号
    if vols_sel and "巻数" in df2.columns:
        df2 = df2[df2["巻数"].map(to_int_or_none).isin(set(vols_sel))]
    if issues_sel and "号数" in df2.columns:
        df2 = df2[df2["号数"].map(to_int_or_none).isin(set(issues_sel))]
    # 著者
    if authors_sel and "著者" in df2.columns:
        sel = {norm_key(a) for a in authors_sel}
        def hit_author(v): return any(norm_key(x) in sel for x in split_authors(v))
        df2 = df2[df2["著者"].apply(hit_author)]
    # 対象物/研究タイプ（top3列に対して部分一致）
    if targets_sel and "対象物_top3" in df2.columns:
        t_norm = [norm_key(t) for t in targets_sel]
        df2 = df2[df2["対象物_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    if types_sel and "研究タイプ_top3" in df2.columns:
        t_norm = [norm_key(t) for t in types_sel]
        df2 = df2[df2["研究タイプ_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    # キーワード
    toks = tokens_from_query(kw_query)
    if toks:
        def hit_kw(row):
            hs = haystack(row)
            return all(t in hs for t in toks) if kw_mode == "AND" else any(t in hs for t in toks)
        df2 = df2[df2.apply(hit_kw, axis=1)]
    return df2

# -------------------- タブ --------------------
tab_search, tab_analysis = st.tabs(["🔍 検索", "📊 分析"])

# ==================== 🔍 検索タブ ====================
with tab_search:
    # -------------------- 年・巻・号フィルタ（従来UI） --------------------
    st.subheader("検索フィルタ")
    year_vals = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
    if year_vals.notna().any():
        ymin_all, ymax_all = int(year_vals.min()), int(year_vals.max())
    else:
        ymin_all, ymax_all = 1980, 2025

    c_y, c_v, c_i = st.columns([1, 1, 1])
    with c_y:
        y_from, y_to = st.slider("発行年（範囲）", min_value=ymin_all, max_value=ymax_all, value=(ymin_all, ymax_all))
        st.session_state.y_from, st.session_state.y_to = y_from, y_to
    with c_v:
        vol_candidates = sorted({v for v in (df.get("巻数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
        vols_sel = st.multiselect("巻（複数選択）", vol_candidates, default=[])
        st.session_state.vols_sel = vols_sel
    with c_i:
        iss_candidates = sorted({v for v in (df.get("号数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
        issues_sel = st.multiselect("号（複数選択）", iss_candidates, default=[])
        st.session_state.issues_sel = issues_sel

    # ---- 1段目：対象物 / 研究タイプ ----
    row1_tg, row1_tp = st.columns([1.2, 1.2])
    with row1_tg:
        raw_targets = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        targets_all = order_by_template(list(raw_targets), TARGET_ORDER)
        targets_sel = st.multiselect("対象物（複数選択／部分一致）", targets_all, default=[])
        st.session_state.targets_sel = targets_sel
    with row1_tp:
        raw_types = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        types_all = order_by_template(list(raw_types), TYPE_ORDER)
        types_sel = st.multiselect("研究タイプ（複数選択／部分一致）", types_all, default=[])
        st.session_state.types_sel = types_sel

    # ---- 2段目：著者（ラジオ＋オートコンプリート） ----
    if "authors_sel" not in st.session_state:
        st.session_state.authors_sel = []

    def handle_author_multiselect_change():
        selected_readings = st.session_state.authors_multiselect_key
        reading2author = dict(zip(st.session_state.author_candidates["reading"], st.session_state.author_candidates["author"]))
        st.session_state.authors_sel = sorted({reading2author[r] for r in selected_readings}) if selected_readings else []
        st.rerun()

    row2_author, row2_radio = st.columns([1.0, 2.0])

    with row2_author:
        adf = st.session_state.author_candidates
        if adf is not None and not adf.empty:
            cand = adf.copy()

            GOJUON = {
                "あ": "あいうえお",
                "か": "かきくけこがぎぐげご",
                "さ": "さしすせそざじずぜぞ",
                "た": "たちつてとだぢづでど",
                "な": "なにぬねの",
                "は": "はひふへほばびぶべぼぱぴぷぺぽ",
                "ま": "まみむめも",
                "や": "やゆよ",
                "ら": "らりるれろ",
                "わ": "わをん",
            }
            def kata_to_hira(s: str) -> str:
                out = []
                for ch in str(s or ""):
                    o = ord(ch)
                    if 0x30A1 <= o <= 0x30F6: out.append(chr(o - 0x60))
                    else: out.append(ch)
                return "".join(out)
            def hira_head(s: str) -> str | None:
                s = str(s or ""); return kata_to_hira(s)[0] if s else None
            def is_roman_head(s: str) -> bool:
                return bool(re.match(r"[A-Za-z]", str(s or "")))

            ini = st.session_state.get("author_initial", "すべて")
            if ini == "英字":
                cand = cand[cand["reading"].astype(str).str.match(r"[A-Za-z]")]
            elif ini != "すべて":
                allowed = set(GOJUON.get(ini, ""))
                cand = cand[cand["reading"].apply(
                    lambda s: (not is_roman_head(s)) and (hira_head(s) in allowed if hira_head(s) else False)
                )]

            AIUEO_ORDER = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
            def sort_tuple(reading: str):
                if not reading: return (3, 999, "")
                ch = reading[0]
                if re.match(r"[A-Za-z]", ch): return (2, 999, ch.lower())
                if re.match(r"[\u30A0-\u30FF]", ch): return (1, 999, reading)
                idx = AIUEO_ORDER.find(ch)
                return (0, idx if idx != -1 else 998, reading)

            cand = cand.assign(
                _grp=[sort_tuple(r)[0] for r in cand["reading"]],
                _key=[sort_tuple(r)[1] for r in cand["reading"]],
                _sub=[sort_tuple(r)[2] for r in cand["reading"]],
            ).sort_values(by=["_grp","_key","_sub"], kind="mergesort").drop(columns=["_grp","_key","_sub"])

            reading2author = dict(zip(cand["reading"], cand["author"]))
            options_readings = list(reading2author.keys())
            selected_readings = []
            if 'authors_sel' in st.session_state:
                selected_author_names = set(st.session_state.authors_sel)
                for r, a in reading2author.items():
                    if a in selected_author_names:
                        selected_readings.append(r)

            st.caption("著者の読み頭文字でサジェストを絞り込み")
            authors_sel_readings = st.multiselect(
                "著者（読みで検索可 / 表示は漢字＋読み）",
                options=options_readings,
                default=selected_readings,
                format_func=lambda r: f"{reading2author.get(r, r)}｜{r}",
                placeholder="例：やまだ / さとう / たかはし ...",
                on_change=handle_author_multiselect_change,
                key="authors_multiselect_key"
            )
        else:
            authors_all = build_author_candidates(df)
            st.session_state.authors_sel = st.multiselect(
                "著者", authors_all, default=st.session_state.authors_sel
            )

    with row2_radio:
        initials = ["すべて","あ","か","さ","た","な","は","ま","や","ら","わ","英字"]
        if "author_initial" not in st.session_state:
            st.session_state.author_initial = "すべて"
        st.radio(
            "著者イニシャル選択",
            options=initials,
            horizontal=True,
            key="author_initial",
        )

    # ---- 3段目：キーワード ----
    kw_row1, kw_row2 = st.columns([3, 1])
    with kw_row1:
        kw_query = st.text_input("キーワード（空白/カンマで複数可）", value="", key="kw_query")
    with kw_row2:
        st.session_state.kw_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True, key="kw_mode")

    # -------------------- フィルタ適用と検索結果 --------------------
    filtered = apply_filters_for_current_state(df)

    st.markdown("### 検索結果")
    st.caption(f"{len(filtered)} / {len(df)} 件")

    visible_cols = make_visible_cols(filtered)
    if "著者" in visible_cols and "summary" in filtered.columns:
        idx = visible_cols.index("著者")
        if "summary" not in visible_cols:
            visible_cols.insert(idx + 1, "summary")

    disp = filtered.loc[:, visible_cols].copy()
    disp["_row_id"] = disp.apply(make_row_id, axis=1)

    # セッション初期化：お気に入り集合／タグ辞書
    if "favs" not in st.session_state:
        st.session_state.favs = set()
    if "fav_tags" not in st.session_state:
        st.session_state.fav_tags = {}

    disp["★"] = disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)

    def handle_main_editor_change():
        edited_rows_dict = st.session_state.main_editor['edited_rows']
        for row_index_str, changes in edited_rows_dict.items():
            row_index = int(row_index_str)
            row_id = disp.iloc[row_index]['_row_id']
            if '★' in changes:
                if changes['★']:
                    st.session_state.favs.add(row_id)
                else:
                    st.session_state.favs.discard(row_id)

    column_config = {
        "★": st.column_config.CheckboxColumn("★", help="気になる論文にチェック/解除", default=False, width="small"),
    }
    if "HPリンク先" in disp.columns:
        column_config["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", help="外部サイトへ移動", display_text="HP")
    if "PDFリンク先" in disp.columns:
        column_config["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", help="PDFを開く", display_text="PDF")

    display_order = ["★"] + [c for c in disp.columns if c not in ["★", "_row_id"]] + ["_row_id"]

    st.subheader("論文リスト")
    st.data_editor(
        disp[display_order],
        key="main_editor",
        on_change=handle_main_editor_change,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        disabled=[c for c in display_order if c != "★"],
        height=520,
        num_rows="fixed",
    )

    # --- お気に入り一覧 ---
    c1, c2 = st.columns([6, 1])
    with c1:
        st.subheader(f"⭐ お気に入り（{len(st.session_state.favs)} 件）")
    with c2:
        if st.button("❌ 全て外す", key="clear_favs_header", use_container_width=True):
            st.session_state.favs = set()
            st.rerun()

    visible_cols_full = make_visible_cols(df)
    if "著者" in visible_cols_full and "summary" in df.columns:
        idx = visible_cols_full.index("著者")
        if "summary" not in visible_cols_full:
            visible_cols_full.insert(idx + 1, "summary")

    fav_disp_full = df.loc[:, visible_cols_full].copy()
    fav_disp_full["_row_id"] = fav_disp_full.apply(make_row_id, axis=1)
    fav_disp = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()

    def tags_str_for(rid: str) -> str:
        s = st.session_state.fav_tags.get(rid, set())
        return ", ".join(sorted(s)) if s else ""

    def update_fav_and_tags_from_favs():
        edited_favs = st.session_state.fav_editor['edited_rows']
        fav_ids_in_view = fav_disp['_row_id'].tolist()
        for row_index_str, changes in edited_favs.items():
            row_index = int(row_index_str)
            row_id = fav_ids_in_view[row_index]
            if '★' in changes:
                if changes['★']:
                    st.session_state.favs.add(row_id)
                else:
                    st.session_state.favs.discard(row_id)
            if 'tags' in changes:
                tag_set = {t.strip() for t in re.split(r"[ ,，、；;　]+", str(changes['tags'])) if t.strip()}
                if tag_set:
                    st.session_state.fav_tags[row_id] = tag_set
                elif row_id in st.session_state.fav_tags:
                    del st.session_state.fav_tags[row_id]
        st.rerun()

    if not fav_disp.empty:
        fav_disp["★"] = fav_disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)
        fav_disp["tags"] = fav_disp["_row_id"].apply(tags_str_for)

        fav_display_order = ["★"] + [c for c in fav_disp.columns if c not in ["★", "_row_id"]] + ["_row_id"]
        fav_column_config = {
            "★": st.column_config.CheckboxColumn("★", help="チェックで解除/追加", default=True, width="small"),
            "tags": st.column_config.TextColumn("tags（カンマ/空白区切り）", help="例: 清酒, 乳酸菌"),
        }
        if "HPリンク先" in fav_disp.columns:
            fav_column_config["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", display_text="HP")
        if "PDFリンク先" in fav_disp.columns:
            fav_column_config["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", display_text="PDF")

        st.data_editor(
            fav_disp[fav_display_order],
            key="fav_editor",
            on_change=update_fav_and_tags_from_favs,
            use_container_width=True,
            hide_index=True,
            column_config=fav_column_config,
            disabled=[c for c in fav_display_order if c not in ["★", "tags"]],
            height=420,
            num_rows="fixed"
        )
    else:
        st.info("お気に入りは未選択です。上の表の『★』にチェックしてから反映してください。")

    with st.expander("🔎 タグでお気に入りを絞り込み（AND/OR）", expanded=False):
        tag_query = st.text_input("タグ検索（カンマ/空白区切り）", key="tag_query")
        tag_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True, key="tag_mode")
        fav_disp_for_filter = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()
        if tag_query.strip():
            tags = [t.strip() for t in re.split(r"[ ,，、；;　]+", tag_query) if t.strip()]
            def match_tags_row(row):
                row_tags = st.session_state.fav_tags.get(row["_row_id"], set())
                return all(t in row_tags for t in tags) if tag_mode == "AND" else any(t in row_tags for t in tags)
            fav_disp_for_filter = fav_disp_for_filter[fav_disp_for_filter.apply(match_tags_row, axis=1)]
        def tags_str_for_filter(rid: str) -> str:
            s = st.session_state.fav_tags.get(rid, set())
            return ", ".join(sorted(s)) if s else ""
        fav_disp_for_filter["tags"] = fav_disp_for_filter["_row_id"].apply(tags_str_for_filter)
        show_cols = ["No.","発行年","巻数","号数","論文タイトル","著者","対象物_top3","研究タイプ","HPリンク先","PDFリンク先","tags"]
        show_cols = [c for c in show_cols if c in fav_disp_for_filter.columns]
        st.dataframe(fav_disp_for_filter[show_cols], use_container_width=True, hide_index=True)

    st.caption(
        f"現在のお気に入り：{len(st.session_state.favs)} 件 / "
        f"タグ数：{len({t for s in st.session_state.fav_tags.values() for t in s})} 稀"
    )

    # 出力
    filtered_export_df = disp.drop(columns=["★", "_row_id"], errors="ignore")
    fav_export = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()
    def _tags_join(rid: str) -> str:
        s = st.session_state.fav_tags.get(rid, set())
        return ", ".join(sorted(s)) if s else ""
    fav_export["tags"] = fav_export["_row_id"].map(_tags_join)
    fav_export = fav_export.drop(columns=["_row_id"], errors="ignore")

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "📥 絞り込み結果をCSV出力（表示列のみ）",
            data=filtered_export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"filtered_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_dl2:
        st.download_button(
            "⭐ お気に入りをCSV出力（tags付き）",
            data=fav_export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"favorites_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=fav_export.empty
        )

# ==================== 📊 分析タブ ====================
with tab_analysis:
    st.subheader("著者共起ネットワーク（中心性ランキング付き）")

    # 対象データ選択
    scope = st.radio("対象データ", ["検索結果（現在の条件）", "全件"], horizontal=True)
    if scope == "検索結果（現在の条件）":
        df_scope = apply_filters_for_current_state(df)
    else:
        df_scope = df

    st.caption(f"対象レコード数：{len(df_scope)}")

    # 著者リスト抽出
    def authors_list_from_row(v):
        return split_authors(v) if isinstance(v, str) else []

    author_rows = df_scope.get("著者", pd.Series(dtype=str)).fillna("").apply(authors_list_from_row)
    # ノード・エッジ
    G = nx.Graph()
    for authors in author_rows:
        authors = [a for a in authors if a]
        # 自己ループ回避、重複除去
        uniq = list(dict.fromkeys(authors))
        for a in uniq:
            if not G.has_node(a):
                G.add_node(a)
        for u, v in itertools.combinations(uniq, 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    if G.number_of_nodes() == 0:
        st.info("著者データがありません。検索条件を広げるか、全件を選択してください。")
    else:
        # 中心性
        deg_c = nx.degree_centrality(G)
        try:
            btw_c = nx.betweenness_centrality(G, weight="weight", normalized=True)
        except Exception:
            btw_c = nx.betweenness_centrality(G, normalized=True)
        try:
            eig_c = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            # fallback（収束しない場合など）
            eig_c = {n: np.nan for n in G.nodes()}

        rank_df = pd.DataFrame({
            "著者": list(G.nodes()),
            "Degree": [deg_c.get(n, 0.0) for n in G.nodes()],
            "Betweenness": [btw_c.get(n, 0.0) for n in G.nodes()],
            "Eigenvector": [eig_c.get(n, np.nan) for n in G.nodes()],
            "共同数（総計）": [int(sum(d["weight"] for *_e, d in G.edges(n, data=True))) for n in G.nodes()]
        })
        # 並び替えキー（Degree→Betweenness→Eigenvector）
        rank_df = rank_df.sort_values(
            by=["Degree","Betweenness","Eigenvector","共同数（総計）"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        topk = st.slider("ランキング表示件数", min_value=10, max_value=200, value=50, step=10)
        st.dataframe(rank_df.head(topk), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### ネットワーク可視化")

        # PyVis → フォールバックで matplotlib
        if _HAS_PYVIS:
            nt = Network(height="680px", width="100%", notebook=False, bgcolor="#FFFFFF", font_color="#000000")
            # 物理演算
            nt.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150, spring_strength=0.05, damping=0.9)

            # ノードサイズ（Degree）
            max_deg = max(deg_c.values()) if deg_c else 1.0
            for n in G.nodes():
                size = 10 + 30 * (deg_c.get(n, 0) / max_deg if max_deg else 0)
                label = n
                title = f"{n}<br>Degree:{deg_c.get(n,0):.3f} / Bet:{btw_c.get(n,0):.3f} / Eig:{eig_c.get(n,0) if not math.isnan(eig_c.get(n,np.nan)) else 'NA'}"
                nt.add_node(n, label=label, title=title, value=size)

            for u, v, d in G.edges(data=True):
                w = d.get("weight", 1)
                nt.add_edge(u, v, value=w, title=f"共著回数: {w}")

            html_file = "author_network.html"
            nt.show(html_file)
            with open(html_file, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            # matplotlib フォールバック
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, k=0.6, seed=42, weight="weight")
            # ノードサイズ：Degree
            deg_vals = np.array([deg_c.get(n, 0.0) for n in G.nodes()])
            sizes = 100 + 1200 * (deg_vals / (deg_vals.max() if deg_vals.max() > 0 else 1))
            nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#69b3a2", alpha=0.8, ax=ax)
            # エッジは重みで太さ
            widths = [0.5 + 2.5 * (G[u][v].get("weight", 1) / max(1, max(nx.get_edge_attributes(G, "weight").values()))) for u,v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            ax.axis("off")
            st.pyplot(fig)