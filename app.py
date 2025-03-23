import streamlit as st
import time
from app.components.form import render_travel_form
from app.components.results import render_loading_state, render_travel_plans
from app.services.langgraph_service import TravelPlannerWorkflow
from app.utils.env_loader import load_env_variables

# ページ設定
st.set_page_config(
    page_title="日本旅行プランナー",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# スタイル設定
st.markdown(
    """
<style>
    .main {
        background-color: #F5F5F5;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {
        color: #1E3A8A;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 環境変数の読み込み
env_vars = load_env_variables()


@st.cache_resource
def get_travel_planner_workflow():
    """TravelPlannerWorkflowのインスタンスを作成してキャッシュする"""
    return TravelPlannerWorkflow(
        openai_api_key=env_vars.get("OPENAI_API_KEY"),
        serpapi_key=env_vars.get("SERPAPI_API_KEY"),
    )


def main():
    # サイドバー
    with st.sidebar:
        st.image(
            "https://www.japan.travel/en/wp-content/uploads/2021/07/header-logo.svg"
        )
        st.title("🏯 日本旅行プランナー")
        st.markdown(
            """
        このアプリは、LangGraphとOpenAI APIを使用して、日本国内の旅行プランを提案します。
        
        あなたの条件に合わせたオリジナルの旅行プランを生成します。
        """
        )

        st.subheader("使い方")
        st.markdown(
            """
        1. 右側のフォームに旅行の条件を入力
        2. 「旅行プランを生成」ボタンをクリック
        3. AIが条件に合った旅行プランを提案
        """
        )

        # ワークフロー図を表示
        with st.expander("ワークフロー図"):
            st.markdown(
                """
            ```mermaid
            graph TD
                A[開始] --> B{情報収集が必要?}
                B -->|はい| C[リサーチ]
                B -->|いいえ| D[プラン生成]
                C --> E{リサーチ成功?}
                E -->|はい| D
                E -->|いいえ| F[エラー処理]
                D --> G{プラン生成成功?}
                G -->|はい| H[追加情報]
                G -->|いいえ| F
                H --> I{追加情報成功?}
                I -->|はい| J[終了]
                I -->|いいえ| F
                F --> J
            ```
            """
            )

        st.caption("© 2023 日本旅行プランナー")

    # メインコンテンツ
    st.title("あなただけの日本旅行プランを作成")

    # セッション状態の初期化
    if "travel_result" not in st.session_state:
        st.session_state.travel_result = None
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False

    # フォームの表示
    form_data = render_travel_form()

    # フォームが送信された場合
    if form_data and not st.session_state.form_submitted:
        st.session_state.form_submitted = True
        render_loading_state()

        try:
            # 旅行プランナーワークフローの取得
            travel_planner = get_travel_planner_workflow()

            # LangGraphワークフローを実行して旅行プランの生成
            result = travel_planner.generate_travel_plans(
                current_location=form_data["current_location"],
                destination=form_data["destination"],
                budget=form_data["budget"],
                duration=form_data["duration"],
                purpose=form_data["purpose"],
            )

            st.session_state.travel_result = result
            st.experimental_rerun()

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            st.session_state.form_submitted = False

    # 結果の表示
    if st.session_state.travel_result:
        render_travel_plans(st.session_state.travel_result)

        # 新しいプランの作成ボタン
        if st.button("新しいプランを作成"):
            st.session_state.travel_result = None
            st.session_state.form_submitted = False
            st.experimental_rerun()


if __name__ == "__main__":
    main()
