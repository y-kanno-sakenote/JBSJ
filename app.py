import streamlit as st
import pandas as pd
import re
from typing import List
import os

# 外部データ（CSV）を読み込む
@st.cache_data
def load_data(url: str):
    """
    指定されたURLからCSVデータをロードする。
    """
    df = pd.read_csv(url)
    return df

# データファイル
DATA_FILE_DEMO = "./data/demo_journal_data.csv"
DATA_FILE_DEFAULT = "https://docs.google.com/spreadsheets/d/1X5l8g3bX_x7_p-p6-aK5_n4_2_K-z9_f/gviz/tq?tqx=out:csv&sheet=Sheet1"
YOMI_FILE = "authors_readings.csv"

# --- アプリケーションのメイン部分 ---
st.title("醸協誌論文検索")
st.markdown("---")

# データ読み込みオプション
data_source = st.radio(
    "データの読み込み元を選択してください。",
    ("デフォルトデータ（Google Sheet）", "デモデータ（ローカル）", "CSVファイル"),
    index=0
)

# データのロード
df = None
if data_source == "デフォルトデータ（Google Sheet）":
    df = load_data(DATA_FILE_DEFAULT)
elif data_source == "デモデータ（ローカル）":
    if os.path.exists(DATA_FILE_DEMO):
        df = load_data(DATA_FILE_DEMO)
    else:
        st.error("デモデータファイルが見つかりません。")
elif data_source == "CSVファイル":
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("CSVファイルをアップロードしてください。")

# データが読み込まれたら処理を進める
if df is not None:
    # --- データ前処理 ---
    df.columns = [col.strip() for col in df.columns]
    
    # 必須の列が存在するか確認
    required_columns = ['論文名', '著者', '発行年', '巻', '号']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        st.error(f"必須の列が見つかりません: {missing_cols}")
        st.stop() # プログラムの実行を停止

    df = df.rename(columns={'hp_link': 'HP', 'pdf_link': 'PDF'})
    df['発行年'] = df['発行年'].astype(str)
    
    # 著者名の読み仮名をCSVから結合
    if os.path.exists(YOMI_FILE):
        try:
            df_yomi = pd.read_csv(YOMI_FILE)
            df_yomi.columns = [col.strip() for col in df_yomi.columns]
            
            # `著者`と`yomi`列が存在するかチェック
            if '著者' in df_yomi.columns and 'yomi' in df_yomi.columns:
                df = pd.merge(df, df_yomi[['著者', 'yomi']], on='著者', how='left')
                df['yomi'].fillna('', inplace=True)
            else:
                st.warning("`authors_readings.csv`に`著者`または`yomi`列が見つかりません。")
                df['yomi'] = ''
        except Exception as e:
            st.warning(f"`authors_readings.csv`の読み込み中にエラーが発生しました: {e}")
            df['yomi'] = ''
    else:
        st.warning("`authors_readings.csv`が見つかりません。著者名の読み仮名フィルタリングは使用できません。")
        df['yomi'] = ''
    
    # NaNを適切に扱う
    df.fillna('', inplace=True)

    # --- フィルタリング機能 ---
    st.sidebar.header("論文の検索・絞り込み")
    st.sidebar.markdown("---")

    # 1. 発行年でフィルタ
    min_year, max_year = df['発行年'].min(), df['発行年'].max()
    if min_year and max_year:
        min_year_val, max_year_val = st.sidebar.slider(
            "発行年で絞り込み",
            int(min_year), int(max_year), (int(min_year), int(max_year))
        )
        df = df[(df['発行年'].astype(int) >= min_year_val) & (df['発行年'].astype(int) <= max_year_val)]

    # 2. 巻・号でフィルタ
    unique_volumes = sorted(df['巻'].unique())
    selected_volumes = st.sidebar.multiselect("巻を選択", options=unique_volumes, default=unique_volumes)
    if selected_volumes:
        df = df[df['巻'].isin(selected_volumes)]
    
    unique_issues = sorted(df['号'].unique())
    selected_issues = st.sidebar.multiselect("号を選択", options=unique_issues, default=unique_issues)
    if selected_issues:
        df = df[df['号'].isin(selected_issues)]

    # 3. 著者でフィルタ
    st.sidebar.subheader("著者")

    # 著者名の読みでフィルタリングするためのラジオボタン
    kana_list = ['すべて', 'あ', 'か', 'さ', 'た', 'な', 'は', 'ま', 'や', 'ら', 'わ', '他']
    selected_kana = st.sidebar.radio("著者名の読みで絞り込み", kana_list, index=0, horizontal=True)

    # 読み仮名でフィルタ
    if selected_kana != 'すべて':
        if 'yomi' in df.columns and df['yomi'].any():
            if selected_kana == 'あ':
                hiragana_range = ['あ', 'い', 'う', 'え', 'お']
            elif selected_kana == 'か':
                hiragana_range = ['か', 'き', 'く', 'け', 'こ']
            elif selected_kana == 'さ':
                hiragana_range = ['さ', 'し', 'す', 'せ', 'そ']
            elif selected_kana == 'た':
                hiragana_range = ['た', 'ち', 'つ', 'て', 'と']
            elif selected_kana == 'な':
                hiragana_range = ['な', 'に', 'ぬ', 'ね', 'の']
            elif selected_kana == 'は':
                hiragana_range = ['は', 'ひ', 'ふ', 'へ', 'ほ']
            elif selected_kana == 'ま':
                hiragana_range = ['ま', 'み', 'む', 'め', 'も']
            elif selected_kana == 'や':
                hiragana_range = ['や', 'ゆ', 'よ']
            elif selected_kana == 'ら':
                hiragana_range = ['ら', 'り', 'る', 'れ', 'ろ']
            elif selected_kana == 'わ':
                hiragana_range = ['わ', 'を', 'ん']
            else: # '他' の場合
                hiragana_chars = set(['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'を', 'ん'])
                df = df[~df['yomi'].str.startswith(tuple(hiragana_chars))]
            
            if selected_kana != '他':
                df = df[df['yomi'].str.startswith(tuple(hiragana_range))]
        else:
            st.warning("`authors_readings.csv`が見つからないか、yomi列が空のため、読み仮名での絞り込みはできません。")
            
    # フィルタリング後の著者リストを更新
    filtered_authors = sorted([author for author in df['著者'].unique() if pd.notna(author) and author.strip() != ''])
    selected_authors = st.sidebar.multiselect("著者を選択", options=filtered_authors)
    if selected_authors:
        df = df[df['著者'].isin(selected_authors)]

    # 4. キーワードでフィルタ
    st.sidebar.markdown("---")
    st.sidebar.subheader("キーワード検索")
    search_query = st.sidebar.text_input("タイトル、著者、キーワード、要旨を検索 (例: ビール, 酵母)")
    and_or_logic = st.sidebar.radio("検索ロジック", ("AND", "OR"), horizontal=True)
    search_full_text = st.sidebar.checkbox("PDF本文を検索対象に含める")

    if search_query:
        df = apply_keyword_filter(df, search_query, and_or_logic, search_full_text)

    # 5. 対象物・研究タイプでフィルタ
    st.sidebar.subheader("トピック・研究タイプで絞り込み")
    selected_subjects = st.sidebar.multiselect(
        "対象物",
        options=sorted(df['対象物_top3'].explode().unique().tolist())
    )
    if selected_subjects:
        df = df[df['対象物_top3'].apply(lambda x: any(s in x for s in selected_subjects))]
        
    selected_types = st.sidebar.multiselect(
        "研究タイプ",
        options=sorted(df['研究タイプ_top3'].explode().unique().tolist())
    )
    if selected_types:
        df = df[df['研究タイプ_top3'].apply(lambda x: any(t in x for t in selected_types))]

    # --- フィルタリング結果表示 ---
    st.markdown("---")
    st.subheader(f"検索結果 ({len(df)}件)")
    st.markdown("★チェックボックスでお気に入りに追加")

    # 検索結果とfavoritesの結合
    if 'favorites' not in st.session_state:
        st.session_state.favorites = pd.DataFrame(columns=df.columns)
    
    merged_df = pd.merge(df, st.session_state.favorites[['論文名']], on='論文名', how='left', indicator=True)
    df['★'] = merged_df['_merge'] == 'both'

    # 結果テーブル表示
    st.dataframe(df[['★', '論文名', '著者', '発行年', 'HP', 'PDF']], hide_index=True)

    # ダウンロード機能
    st.markdown("---")
    st.subheader("データのダウンロード")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="絞り込み結果をCSVでダウンロード",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="search_results.csv",
            mime="text/csv",
        )

    # --- お気に入り機能 ---
    st.markdown("---")
    st.header("★お気に入りリスト")
    st.markdown("タグ列を直接編集できます（カンマまたはスペース区切り）。")

    # favoritesの更新
    st.session_state.favorites = df[df['★']].copy()
    if 'tags' not in st.session_state.favorites.columns:
        st.session_state.favorites['tags'] = ''

    # お気に入りリストの表示と編集
    if not st.session_state.favorites.empty:
        # タグでフィルタ
        fav_search_query = st.text_input("お気に入りリストをタグで検索 (例: 清酒, 発酵)")
        fav_and_or_logic = st.radio("タグ検索ロジック", ("AND", "OR"), key='fav_logic', horizontal=True)
        
        filtered_favs = st.session_state.favorites.copy()
        if fav_search_query:
            filtered_favs = apply_keyword_filter(filtered_favs, fav_search_query, fav_and_or_logic, search_columns=['tags'])
        
        edited_df = st.data_editor(filtered_favs[['論文名', '著者', '発行年', 'tags', 'HP', 'PDF']], hide_index=True, num_rows="dynamic", use_container_width=True)
        st.session_state.favorites['tags'] = edited_df['tags']
        
        # ダウンロード
        with col2:
            st.download_button(
                label="お気に入りをCSVでダウンロード",
                data=st.session_state.favorites.to_csv(index=False).encode('utf-8'),
                file_name="favorites.csv",
                mime="text/csv",
            )
    
    if st.button("お気に入りを全て削除"):
        st.session_state.favorites = pd.DataFrame(columns=df.columns)
        st.experimental_rerun()
    
    st.markdown("---")

# --- ヘルパー関数 ---
def apply_keyword_filter(df, query, logic, search_full_text=False, search_columns=None):
    keywords = re.split(r'[,\s]+', query.strip())
    if not keywords:
        return df

    search_cols = ['論文名', '著者', 'キーワード', '要旨']
    if search_full_text:
        search_cols.append('pdf_text')
    if search_columns:
        search_cols = search_columns

    if logic == "AND":
        for keyword in keywords:
            if keyword:
                df = df[df.apply(
                    lambda row: any(keyword in str(row[col]) for col in search_cols), axis=1
                )]
    elif logic == "OR":
        condition = df.apply(
            lambda row: any(keyword in str(row[col]) for keyword in keywords for col in search_cols), axis=1
        )
        df = df[condition]
    
    return df