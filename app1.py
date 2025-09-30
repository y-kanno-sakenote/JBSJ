# -*- coding: utf-8 -*-
"""
論文検索デモ（著者クリックでフィルタ追加版）
- 検索結果に出てきた「著者」をクリックで著者フィルタに追加できる仕組み
- st.dataframe はセルクリックイベントを取れないため、結果テーブル下に著者ボタン群を生成して対応
- CSV列: 論文タイトル, 著者, HPリンク先, PDFリンク先, summary（任意）
"""

import re
from pathlib import Path
import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="論文検索（著者クリック追加デモ）", layout="wide")

AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell):
    if cell is None:
        return []
    return [w.strip() for w in AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def norm_key(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower().strip()

def build_author_candidates(df: pd.DataFrame):
    raw = df.get("著者", pd.Series(dtype=str)).fillna("").tolist()
    authors = []
    for cell in raw:
        authors.extend(split_authors(cell))
    seen, uniq = set(), []
    for a in authors:
        k = norm_key(a)
        if k and k not in seen:
            seen.add(k)
            uniq.append(a)
    return sorted(uniq)

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_demo_df() -> pd.DataFrame:
    demo = Path("data/keywords_summary4.csv")
    if demo.exists():
        try:
            df = pd.read_csv(demo, encoding="utf-8")
            return ensure_cols(df)
        except Exception:
            pass
    return pd.DataFrame([
        {"論文タイトル":"清酒中のアミノ酸に関する研究","著者":"山田太郎; 佐藤花子","HPリンク先":"https://example.com/hp1","PDFリンク先":"https://example.com/p1.pdf","summary":"清酒のアミノ酸組成と官能特性の関係を概説。"},
        {"論文タイトル":"味噌発酵における乳酸菌ダイナミクス","著者":"田中一郎","HPリンク先":"https://example.com/hp2","PDFリンク先":"https://example.com/p2.pdf","summary":"味噌発酵中の乳酸菌群集の推移と代謝産物を解析。"},
        {"論文タイトル":"ビールのホップ香成分の変動","著者":"Suzuki Ken; 佐藤花子","HPリンク先":"https://example.com/hp3","PDFリンク先":"https://example.com/p3.pdf","summary":"ドライホッピング条件と香気成分の関係を調査。"}
    ])

with st.sidebar:
    st.header("データ読み込み")
    up = st.file_uploader("CSVを選択（UTF-8）", type=["csv"])
    use_demo = st.toggle("デモデータを使う", value=True)

if up is not None:
    try:
        df = ensure_cols(pd.read_csv(up, encoding="utf-8"))
        st.toast("CSVを読み込みました")
    except Exception as e:
        st.error(f"CSVの読み込みに失敗: {e}")
        st.stop()
elif use_demo:
    df = load_demo_df()
else:
    st.info("左サイドバーでCSVを指定するかデモを有効にしてください。")
    st.stop()

if "summary" not in df.columns:
    df["summary"] = ""

st.title("論文検索")

if "authors_sel" not in st.session_state:
    st.session_state["authors_sel"] = []

col_a, col_kw = st.columns([1.2, 2.0])

with col_a:
    authors_all = build_author_candidates(df)
    authors_sel = st.multiselect(
        "著者（複数選択）",
        authors_all,
        default=st.session_state["authors_sel"],
        key="authors_multiselect"
    )

with col_kw:
    kw = st.text_input("キーワード（タイトル／summary を部分一致）", value="")

def apply_filters(_df: pd.DataFrame) -> pd.DataFrame:
    df2 = _df.copy()
    if st.session_state["authors_sel"]:
        sel = {norm_key(a) for a in st.session_state["authors_sel"]}
        def hit_author(v):
            return any(norm_key(x) in sel for x in split_authors(v))
        df2 = df2[df2["著者"].apply(hit_author)]
    if kw.strip():
        q = norm_key(kw)
        mask = df2["論文タイトル"].astype(str).str.lower().str.contains(q) | \
               df2["summary"].astype(str).str.lower().str.contains(q)
        df2 = df2[mask]
    return df2

filtered = apply_filters(df)

def short_summary(s: str, n=120):
    s = str(s or "").strip()
    return (s[:n] + "…") if len(s) > n else s

disp = filtered.copy()
disp["summary(要約)"] = disp["summary"].apply(short_summary)

cols_order = [c for c in ["論文タイトル","著者","HPリンク先","PDFリンク先","summary(要約)"] if c in disp.columns]
disp = disp[cols_order]

st.markdown(f"### 検索結果（{len(disp)} 件）")
st.dataframe(disp, use_container_width=True, hide_index=True)

st.divider()
with st.expander("👆 結果に出た著者をクリックしてフィルタに追加", expanded=False):
    author_cells = filtered.get("著者", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    cand_authors = []
    for cell in author_cells:
        cand_authors.extend(split_authors(cell))
    uniq_authors, seen = [], set()
    for a in cand_authors:
        k = norm_key(a)
        if k and k not in seen:
            seen.add(k)
            uniq_authors.append(a)
    if not uniq_authors:
        st.caption("（この検索結果には著者名が見つかりませんでした）")
    else:
        st.caption("クリックすると上部の『著者（複数選択）』に追加されます。")
        def _add_author(name: str):
            st.session_state["authors_sel"] = sorted(set(st.session_state["authors_sel"]) | {name})
            st.session_state["authors_multiselect"] = st.session_state["authors_sel"]
        per_row = 6
        rows = [uniq_authors[i:i+per_row] for i in range(0, len(uniq_authors), per_row)]
        for row in rows:
            cols = st.columns(per_row, gap="small")
            for col, name in zip(cols, row):
                with col:
                    st.button(
                        f"＋ {name}",
                        key=f"add_author_{hash(name)}",
                        use_container_width=True,
                        on_click=_add_author,
                        args=(name,)
                    )
        if st.session_state["authors_sel"]:
            st.info("現在選択中の著者: " + " / ".join(st.session_state["authors_sel"]))

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.download_button(
        "📥 現在の結果をCSVでダウンロード",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"results_{time.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
with col_d2:
    st.download_button(
        "📥 全データをCSVでダウンロード",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"all_{time.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
