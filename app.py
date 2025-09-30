# -*- coding: utf-8 -*-
"""
醸造協会誌 論文検索（統一UI版）
- 既存機能は変更なし
- 追加: 検索結果の著者名をクリックすると、著者フィルタ（readingベースの multiselect）へ自動追加
"""

import io, re, time, urllib.parse
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# -------------------- ページ設定 --------------------
st.set_page_config(page_title="論文検索（統一UI版）", layout="wide")

# -------------------- 定数 --------------------
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

def consolidate_authors_column(df: pd.DataFrame) -> pd.DataFrame:
    """著者列：空白では分割せず、区切り記号のみで分割→同名異表記を代表表記に統合"""
    if "著者" not in df.columns:
        return df
    df = df.copy()
    def unify(cell: str) -> str:
        names = split_authors(cell)
        seen = set()
        result = []
        for n in names:
            k = norm_key(n)
            if not k or k in seen:
                continue
            seen.add(k)
            result.append(n)
        return ", ".join(result)
    df["著者"] = df["著者"].astype(str).apply(unify)
    return df

def build_author_candidates(df: pd.DataFrame):
    rep = {}
    for v in df.get("著者", pd.Series(dtype=str)).fillna(""):
        for name in split_authors(v):
            k = norm_key(name)
            if k and k not in rep:
                rep[k] = name
    return [rep[k] for k in sorted(rep.keys())]

def haystack(row, include_fulltext: bool):
    parts = [
        str(row.get("論文タイトル","")),
        str(row.get("著者","")),
        str(row.get("file_name","")),
        " ".join(str(row.get(c,"")) for c in KEY_COLS if c in row),
    ]
    # 本文検索は現状なし（UIから削除済み）
    return norm_key(" \n ".join(parts))

def to_int_or_none(x):
    try: return int(str(x).strip())
    except Exception:
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else None

def order_by_template(values, template):
    """1) テンプレの順 2) 未収載はアルファ順 3) その他は最後"""
    vs = list(dict.fromkeys(values))
    tmpl_set = set(template)
    head = [v for v in template if v in vs and "その他" not in v]
    mid  = sorted([v for v in vs if v not in tmpl_set and "その他" not in v])
    tail = [v for v in template if v in vs and "その他" in v] + \
           [v for v in vs if ("その他" in v and v not in template)]
    return head + mid + tail

def make_visible_cols(df: pd.DataFrame) -> list[str]:
    """df の列から『相対PASS/終了ページ/file_path/num_pages/file_name』と
       『llm_keywords 以降の全列』を非表示対象にして、表示列リストを返す。
    """
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

# === 著者読みCSVのローダ（authors_readings.csv） ===
AUTHORS_CSV_PATH = Path("data/authors_readings.csv")

@st.cache_data(ttl=600, show_spinner=False)
def load_authors_readings(path: Path) -> pd.DataFrame | None:
    try:
        df_a = pd.read_csv(path, encoding="utf-8")
        df_a.columns = [str(c).strip() for c in df_a.columns]
        if not {"author", "reading"}.issubset(set(df_a.columns)):
            return None
        df_a["author"]  = df_a["author"].astype(str).str.strip()
        df_a["reading"] = df_a["reading"].astype(str).str.strip()
        df_a = df_a[(df_a["author"]!="") & (df_a["reading"]!="")]
        df_a = df_a.drop_duplicates(subset=["reading"], keep="first")
        return df_a
    except Exception:
        return None

# -------------------- タイトル --------------------
st.title("醸造協会誌　論文検索")

# -------------------- データ読み込み --------------------
DEMO_CSV_PATH = Path("data/keywords_summary4.csv")
SECRET_URL = st.secrets.get("GSHEET_CSV_URL", "")

@st.cache_data(ttl=600, show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame:
    return ensure_cols(pd.read_csv(path, encoding="utf-8"))

@st.cache_data(ttl=600, show_spinner=False)
def load_url_csv(url: str) -> pd.DataFrame:
    return ensure_cols(fetch_csv(url))

with st.sidebar:
    st.header("データ読み込み")
    st.caption("※ まずはデモ用CSVを自動ロード。URL/ファイル指定で上書きできます。")
    use_demo = st.toggle("デモCSVを自動ロードする", value=True)
    url = st.text_input("公開CSVのURL（Googleスプレッドシート output=csv）", value=SECRET_URL)
    up  = st.file_uploader("CSVをローカルから読み込み", type=["csv"])
    load_clicked = st.button("読み込み（URL/ファイルを優先）", type="primary", key="load_btn")

df = None
err = None
try:
    if load_clicked:
        if up is not None:
            df = ensure_cols(pd.read_csv(up, encoding="utf-8"))
            st.toast("ローカルCSVを読み込みました")
        elif url.strip():
            df = load_url_csv(url.strip())
            st.toast("URLのCSVを読み込みました")
        else:
            st.warning("URL または CSV を指定してください。")
    elif use_demo and DEMO_CSV_PATH.exists():
        df = load_local_csv(DEMO_CSV_PATH)
        st.caption(f"✅ デモCSVを自動ロード中: {DEMO_CSV_PATH}")
    elif SECRET_URL:
        df = load_url_csv(SECRET_URL)
        st.caption("✅ SecretsのURLから自動ロード中")
except Exception as e:
    err = e

if df is None:
    if err:
        st.error(f"読み込みエラー: {err}")
    st.info("左のサイドバーで CSV を指定するか、デモCSVを有効にしてください。")
    st.stop()

# -------------------- 年・巻・号フィルタ --------------------
st.subheader("年・巻・号フィルタ")
year_vals = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
if year_vals.notna().any():
    ymin_all, ymax_all = int(year_vals.min()), int(year_vals.max())
else:
    ymin_all, ymax_all = 1980, 2025

c_y, c_v, c_i = st.columns([1, 1, 1])
with c_y:
    y_from, y_to = st.slider(
        "発行年（範囲）", min_value=ymin_all, max_value=ymax_all,
        value=(ymin_all, ymax_all)
    )
with c_v:
    vol_candidates = sorted({v for v in (df.get("巻数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    vols_sel = st.multiselect("巻（複数選択）", vol_candidates, default=[])
with c_i:
    iss_candidates = sorted({v for v in (df.get("号数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    issues_sel = st.multiselect("号（複数選択）", iss_candidates, default=[])

# -------------------- 検索フィルタ --------------------
st.subheader("検索フィルタ")
# 1段目：対象物 / 研究タイプ
c_tg, c_tp = st.columns([1.2, 1.2])
with c_tg:
    raw_targets = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    targets_all = order_by_template(list(raw_targets), TARGET_ORDER)
    targets_sel = st.multiselect("対象物（複数選択／部分一致）", targets_all, default=[])
with c_tp:
    raw_types = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    types_all = order_by_template(list(raw_types), TYPE_ORDER)
    types_sel = st.multiselect("研究タイプ（複数選択／部分一致）", types_all, default=[])

# 2段目：著者（右にイニシャル）
c_a1, c_a2 = st.columns([2.0, 1.2])
with c_a1:
    # 著者（読みで検索・表示は漢字＋読み）
    adf = load_authors_readings(AUTHORS_CSV_PATH)
    authors_sel = []
    authors_sel_readings_default = st.session_state.get("authors_sel_readings", [])
    if adf is not None:
        # イニシャル（あかさたな/英字）
        if "author_initial" not in st.session_state:
            st.session_state.author_initial = "すべて"
        initials = ["すべて","あ","か","さ","た","な","は","ま","や","ら","わ","英字"]

        # 候補生成
        cand = adf.copy()

        GOJUON = {
            "あ": "あいうえおがぎぐげご",
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
            for ch in str(s):
                code = ord(ch)
                if 0x30A1 <= code <= 0x30F6:
                    out.append(chr(code - 0x60))
                else:
                    out.append(ch)
            return "".join(out)

        def is_roman_head(s: str) -> bool:
            return bool(re.match(r"[A-Za-z]", str(s or "")))

        def hira_head(s: str) -> str | None:
            s = str(s or "")
            if not s: return None
            h = kata_to_hira(s)[0]
            return h

        ini = st.session_state.author_initial

        # 並び替え：ひらがな→カタカナ→英字
        AIUEO_ORDER = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
        def sort_tuple_reading(r):
            r = str(r or "")
            if not r: return (3, 999, r)
            ch = r[0]
            if re.match(r"[A-Za-z]", ch):
                return (2, 999, ch.lower())
            if re.match(r"[\u30A0-\u30FF]", ch):
                return (1, 999, r)
            idx = AIUEO_ORDER.find(ch)
            if idx == -1: idx = 998
            return (0, idx, r)

        # ラジオで範囲絞り込み
        def filter_by_initial(df_cand: pd.DataFrame, initial: str) -> pd.DataFrame:
            if initial == "英字":
                m = df_cand["reading"].astype(str).str.match(r"[A-Za-z]")
                return df_cand[m]
            if initial == "すべて":
                return df_cand
            allowed = set(GOJUON.get(initial, ""))
            def in_row(s):
                if not s or is_roman_head(s):
                    return False
                h = hira_head(s)
                return bool(h and h in allowed)
            return df_cand[df_cand["reading"].apply(in_row)]

        cand = filter_by_initial(cand, ini)
        # 並べ替え
        _tmp = cand["reading"].apply(lambda r: pd.Series(sort_tuple_reading(r), index=["_grp","_key","_sub"]))
        cand = pd.concat([cand.reset_index(drop=True), _tmp], axis=1)\
                 .sort_values(by=["_grp","_key","_sub"], kind="mergesort")\
                 .drop(columns=["_grp","_key","_sub"]).reset_index(drop=True)

        reading2author = dict(zip(cand["reading"], cand["author"]))
        options_readings = list(reading2author.keys())

        authors_sel_readings = st.multiselect(
            "著者（読みで検索可 / 表示は漢字＋読み）",
            options=options_readings,
            format_func=lambda r: f"{reading2author.get(r, r)}｜{r}",
            placeholder="例：やまだ / さとう / たかはし ...",
            default=[r for r in authors_sel_readings_default if r in options_readings]
        )
        authors_sel = sorted({reading2author[r] for r in authors_sel_readings}) if authors_sel_readings else []
        st.session_state["authors_sel_readings"] = authors_sel_readings
        st.session_state["authors_sel"] = authors_sel
        st.session_state["reading2author_latest_map"] = reading2author  # クリック時の逆引きに使う
    else:
        # フォールバック
        authors_all = build_author_candidates(df)
        authors_sel = st.multiselect("著者", authors_all, default=[])

with c_a2:
    st.caption("著者の読み・イニシャル")
    initials = ["すべて","あ","か","さ","た","な","は","ま","や","ら","わ","英字"]
    if "author_initial" not in st.session_state:
        st.session_state.author_initial = "すべて"
    st.session_state.author_initial = st.radio(
        "著者イニシャル選択",
        initials,
        horizontal=True,
        index=initials.index(st.session_state.author_initial) if st.session_state.author_initial in initials else 0,
        key="author_initial_radio"
    )

# 3段目：キーワード
c_kw1, c_kw2 = st.columns([3, 1])
with c_kw1:
    kw_query = st.text_input("キーワード（空白/カンマで複数可）", value="")
with c_kw2:
    kw_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True, key="kw_mode")

# -------------------- フィルタ適用 --------------------
def apply_filters(_df: pd.DataFrame) -> pd.DataFrame:
    df2 = _df.copy()
    if "発行年" in df2.columns:
        y = pd.to_numeric(df2["発行年"], errors="coerce")
        df2 = df2[(y >= y_from) & (y <= y_to) | y.isna()]
    if "巻数" in df2.columns and vols_sel:
        df2 = df2[df2["巻数"].map(to_int_or_none).isin(set(vols_sel))]
    if "号数" in df2.columns and issues_sel:
        df2 = df2[df2["号数"].map(to_int_or_none).isin(set(issues_sel))]
    if authors_sel and "著者" in df2.columns:
        sel = {norm_key(a) for a in authors_sel}
        def hit_author(v): return any(norm_key(x) in sel for x in split_authors(v))
        df2 = df2[df2["著者"].apply(hit_author)]
    if targets_sel and "対象物_top3" in df2.columns:
        t_norm = [norm_key(t) for t in targets_sel]
        df2 = df2[df2["対象物_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    if types_sel and "研究タイプ_top3" in df2.columns:
        t_norm = [norm_key(t) for t in types_sel]
        df2 = df2[df2["研究タイプ_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    toks = tokens_from_query(kw_query)
    if toks:
        def hit_kw(row):
            hs = haystack(row, include_fulltext=False)
            return all(t in hs for t in toks) if kw_mode == "AND" else any(t in hs for t in toks)
        df2 = df2[df2.apply(hit_kw, axis=1)]
    return df2

filtered = apply_filters(df)

# -------------------- 検索結果テーブル（既存そのまま） --------------------
st.markdown("### 検索結果")
st.caption(f"{len(filtered)} / {len(df)} 件")

visible_cols = make_visible_cols(filtered)
disp = filtered.loc[:, visible_cols].copy()
disp["_row_id"] = disp.apply(make_row_id, axis=1)

# summary の可読性（省略してツールチップ）：既存の列名 'summary' があれば加工
if "summary" in disp.columns:
    def clip_sum(s, n=120):
        s = str(s or "")
        return s if len(s) <= n else (s[:n] + "…")
    disp["_summary_tip"] = disp["summary"].astype(str)
    disp["summary"] = disp["summary"].apply(clip_sum)

# セッション初期化：お気に入り集合／タグ辞書
if "favs" not in st.session_state:
    st.session_state.favs = set()
if "fav_tags" not in st.session_state:
    st.session_state.fav_tags = {}   # row_id -> set(tags)

# メイン表：お気に入りチェック列
disp["★"] = disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)

# LinkColumn 設定
column_config = {
    "★": st.column_config.CheckboxColumn("★", help="気になる論文にチェック/解除", default=False, width="small"),
}
if "HPリンク先" in disp.columns:
    column_config["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", help="外部サイトへ移動", display_text="HP")
if "PDFリンク先" in disp.columns:
    column_config["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", help="PDFを開く", display_text="PDF")
if "summary" in disp.columns and "_summary_tip" in disp.columns:
    column_config["summary"] = st.column_config.TextColumn("summary（ホバーで全文）", help="", disabled=True)

display_order = ["★"] + [c for c in disp.columns if c not in ["★", "_row_id", "_summary_tip"]] + ["_row_id"]

st.subheader("論文リスト")
with st.form("main_table_form", clear_on_submit=False):
    edited_main = st.data_editor(
        disp[display_order],
        key="main_editor",
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        disabled=[c for c in display_order if c != "★"],
        height=520,
        num_rows="fixed",
    )
    apply_main = st.form_submit_button("チェックした論文をお気に入りリストに追加", use_container_width=True)

if apply_main:
    subset_ids_main = set(disp["_row_id"].tolist())
    checked_subset_main = set(edited_main.loc[edited_main["★"] == True, "_row_id"].tolist())
    st.session_state.favs = (st.session_state.favs - subset_ids_main) | checked_subset_main

# -------------------- 追加：著者クリック補助テーブル（著者名をリンク化） --------------------
def _author_links_cell(authors_cell: str) -> str:
    names = split_authors(authors_cell)
    links = []
    for a in names:
        a_clean = a.strip()
        if not a_clean:
            continue
        href = "?add_author=" + urllib.parse.quote(a_clean)
        links.append(f'<a href="{href}" style="color:#1a73e8; text-decoration:underline;">{a_clean}</a>')
    return " / ".join(links) if links else ""

# 著者リンクのみの軽量テーブル（No. / 論文タイトル / 著者リンク）
mini = filtered.copy()
keep_cols = []
if "No." in mini.columns: keep_cols.append("No.")
if "論文タイトル" in mini.columns: keep_cols.append("論文タイトル")
if "著者" in mini.columns: keep_cols.append("著者")
mini = mini.loc[:, keep_cols].copy() if keep_cols else pd.DataFrame(columns=["著者"])

if not mini.empty and "著者" in mini.columns:
    mini["著者（クリックでフィルタ追加）"] = mini["著者"].apply(_author_links_cell)
    show_cols = [c for c in ["No.","論文タイトル","著者（クリックでフィルタ追加）"] if c in mini.columns]
    st.markdown("##### 著者クリック用（補助表示）")
    st.markdown(
        mini[show_cols].to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
    st.caption("※ 上の「論文リスト」は従来通り編集可、こちらは“著者クリックでフィルタ追加”専用の補助表示です。")

# ---- クエリパラメータから著者追加（readingへ逆引きして multiselect に反映） ----
if "add_author" in st.query_params:
    add_name = st.query_params.get("add_author")
    # 最新の候補マッピング（reading -> author）を取得（無い場合は authors_readings.csv から復元）
    reading2author = st.session_state.get("reading2author_latest_map")
    if reading2author is None and adf is not None:
        reading2author = dict(zip(adf["reading"], adf["author"]))

    added_readings = []
    if reading2author:
        # author -> reading の逆引き（複数一致はすべて）
        for r, a in reading2author.items():
            if a == add_name:
                added_readings.append(r)

    if added_readings:
        cur = st.session_state.get("authors_sel_readings", [])
        new_list = list(dict.fromkeys(cur + [r for r in added_readings if r not in cur]))
        st.session_state["authors_sel_readings"] = new_list
        # authors_sel も同期
        st.session_state["authors_sel"] = sorted({reading2author[r] for r in new_list})
        # クエリを消して反映
        st.query_params.clear()
        st.rerun()
    else:
        # 逆引きできない場合は何もしない（クエリだけ消す）
        st.query_params.clear()

# -------------------- お気に入り一覧 --------------------
c1, c2 = st.columns([6, 1])
with c1:
    st.subheader(f"⭐ お気に入り（{len(st.session_state.favs)} 件）")
with c2:
    if st.button("❌ 全て外す", key="clear_favs_header", use_container_width=True):
        st.session_state.favs = set()
        st.rerun()

visible_cols_full = make_visible_cols(df)
fav_disp_full = df.loc[:, visible_cols_full].copy()
fav_disp_full["_row_id"] = fav_disp_full.apply(make_row_id, axis=1)
fav_disp = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()

def tags_str_for(rid: str) -> str:
    s = st.session_state.fav_tags.get(rid, set())
    return ", ".join(sorted(s)) if s else ""

if not fav_disp.empty:
    fav_disp["★"] = fav_disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)
    fav_disp["tags"] = fav_disp["_row_id"].apply(tags_str_for)

    fav_display_order = ["★"] + [c for c in fav_disp.columns if c not in ["★", "_row_id"]] + ["_row_id"]

    fav_column_config = {
        "★": st.column_config.CheckboxColumn("★", help="チェックで解除/追加（下のボタンで反映）", default=True, width="small"),
        "tags": st.column_config.TextColumn("tags（カンマ/空白区切り）", help="例: 清酒, 乳酸菌"),
    }
    if "HPリンク先" in fav_disp.columns:
        fav_column_config["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", display_text="HP")
    if "PDFリンク先" in fav_disp.columns:
        fav_column_config["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", display_text="PDF")

    with st.form("fav_table_form", clear_on_submit=False):
        fav_edited = st.data_editor(
            fav_disp[fav_display_order],
            key="fav_editor",
            use_container_width=True,
            hide_index=True,
            column_config=fav_column_config,
            disabled=[c for c in fav_display_order if c not in ["★", "tags"]],
            height=420,
            num_rows="fixed",
        )
        apply_fav = st.form_submit_button("お気に入りの変更（★/tags）を更新", use_container_width=True)

    if apply_fav:
        subset_ids_fav = set(fav_disp["_row_id"].tolist())
        fav_checked_subset = set(fav_edited.loc[fav_edited["★"] == True, "_row_id"].tolist())
        st.session_state.favs = (st.session_state.favs - subset_ids_fav) | fav_checked_subset

        def parse_tags(s):
            if not isinstance(s, str): s = str(s or "")
            parts = [t.strip() for t in re.split(r"[ ,，、；;　]+", s) if t.strip()]
            return set(parts)
        for _, r in fav_edited.iterrows():
            rid = r["_row_id"]
            tag_set = parse_tags(r.get("tags", ""))
            if tag_set:
                st.session_state.fav_tags[rid] = tag_set
            elif rid in st.session_state.fav_tags:
                del st.session_state.fav_tags[rid]

        st.success("お気に入り（★/tags）を反映しました")
        st.rerun()
else:
    st.info("お気に入りは未選択です。上の表の『★』にチェックしてから反映してください。")

# -------------------- タグでお気に入りを絞り込み（折り畳み） --------------------
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

# -------------------- 下部アクション（CSV出力：2種類） --------------------
st.caption(
    f"現在のお気に入り：{len(st.session_state.favs)} 件 / "
    f"タグ数：{len({t for s in st.session_state.fav_tags.values() for t in s})} 種"
)

filtered_export_df = disp.drop(columns=["★", "_row_id", "_summary_tip"], errors="ignore")

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