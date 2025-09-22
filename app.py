# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

# ======================
# データ読み込み
# ======================
@st.cache_data
def load_csv(path_or_url: str) -> pd.DataFrame:
    # Excel用CSV（BOM付き）を想定
    df = pd.read_csv(path_or_url, encoding="utf-8-sig")
    # 欠損は空文字で埋める
    df = df.fillna("")
    return df


# ======================
# メインアプリ
# ======================
def main():
    st.set_page_config(page_title="論文検索DB", layout="wide")

    st.title("📑 日本醸造協会誌 論文検索DB")

    # CSVパス（固定 or 入力）
    csv_path = "keywords_summary4.csv"  # ←ここを差し替え
    df = load_csv(csv_path)

    # ======================
    # フィルタUI（サイドバーではなく上部に）
    # ======================
    with st.expander("🔍 フィルタ条件", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.multiselect("発行年", sorted(df["発行年"].unique()))
        with col2:
            vol = st.multiselect("巻数", sorted(df["巻数"].unique()))
        with col3:
            issue = st.multiselect("号数", sorted(df["号数"].unique()))

        col4, col5 = st.columns(2)
        with col4:
            obj = st.multiselect("対象物", sorted(df["対象物_top3"].unique()))
        with col5:
            rtype = st.multiselect("研究タイプ", sorted(df["研究タイプ_top3"].unique()))

        keyword = st.text_input("キーワード検索（タイトル・著者・llm_keywords）", "")

    # ======================
    # フィルタ適用
    # ======================
    q = df.copy()
    if year:
        q = q[q["発行年"].isin(year)]
    if vol:
        q = q[q["巻数"].isin(vol)]
    if issue:
        q = q[q["号数"].isin(issue)]
    if obj:
        q = q[q["対象物_top3"].isin(obj)]
    if rtype:
        q = q[q["研究タイプ_top3"].isin(rtype)]
    if keyword:
        q = q[
            q["論文タイトル"].str.contains(keyword, case=False, na=False)
            | q["著者"].str.contains(keyword, case=False, na=False)
            | q["llm_keywords"].str.contains(keyword, case=False, na=False)
        ]

    # ======================
    # 結果表示（お気に入り付き）
    # ======================
    st.markdown(f"### 検索結果 ({len(q)} 件)")

    # お気に入り管理
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = set()

    def toggle_fav(fname):
        if fname in st.session_state["favorites"]:
            st.session_state["favorites"].remove(fname)
        else:
            st.session_state["favorites"].add(fname)

    # 表示用テーブル
    for _, row in q.iterrows():
        cols = st.columns([0.3, 1, 1, 1])
        fav_button = "★" if row["file_name"] in st.session_state["favorites"] else "☆"
        if cols[0].button(fav_button, key=row["file_name"]):
            toggle_fav(row["file_name"])
        cols[1].markdown(f"**{row['論文タイトル']}**")
        cols[2].markdown(row["著者"])
        cols[3].markdown(f"{row['発行年']}年 {row['巻数']}巻{row['号数']}号")

    # ======================
    # お気に入り一覧
    # ======================
    st.markdown("### ⭐ お気に入り一覧")
    fav_df = df[df["file_name"].isin(st.session_state["favorites"])]
    st.dataframe(fav_df[["発行年", "巻数", "号数", "論文タイトル", "著者"]])


if __name__ == "__main__":
    main()