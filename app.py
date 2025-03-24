import streamlit as st
import time
import logging
import traceback
import subprocess
import sys
from app.components.form import render_travel_form
from app.components.results import render_loading_state, render_travel_plans
from app.services.langgraph_service import TravelPlannerWorkflow
from app.utils.env_loader import load_env_variables

# from app.utils.langsmith_utils import render_langsmith_dashboard

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # 標準出力へのハンドラ
)
logger = logging.getLogger("TripPlannerApp")

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
logger.info("環境変数の読み込み開始")
try:
    env_vars = load_env_variables()
    logger.info("環境変数の読み込み完了")
except Exception as e:
    logger.error(f"環境変数の読み込み中にエラーが発生: {e}")
    logger.error(traceback.format_exc())
    st.error(f"環境変数の読み込み中にエラーが発生しました: {str(e)}")
    env_vars = {}


def install_package(package_name):
    """必要なパッケージをインストールする"""
    try:
        logger.info(f"{package_name}のインストールを試みます")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"{package_name}のインストールに成功しました")
        return True
    except Exception as e:
        logger.error(f"{package_name}のインストール中にエラーが発生: {e}")
        return False


@st.cache_resource
def get_travel_planner_workflow():
    """TravelPlannerWorkflowのインスタンスを作成してキャッシュする"""
    logger.info("TravelPlannerWorkflowのインスタンスを作成")
    try:
        openai_api_key = env_vars.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI APIキーが設定されていません")
            st.error(
                "OpenAI APIキーが設定されていません。.envファイルを確認してください。"
            )
            return None

        workflow = TravelPlannerWorkflow(
            openai_api_key=openai_api_key,
            serpapi_key=env_vars.get("SERPAPI_API_KEY"),
        )
        logger.info("TravelPlannerWorkflowの作成成功")
        return workflow
    except ImportError as e:
        error_message = str(e)
        logger.error(f"インポートエラー: {error_message}")

        # faissのインポートエラーの場合
        if "faiss" in error_message:
            logger.info("faissのインポートエラーを検出")
            st.error(
                """
                FAISSライブラリがインストールされていないため、ベクトル検索機能が使用できません。
                
                **解決策**:
                1. ターミナル/コマンドプロンプトで以下のコマンドを実行してください：
                   ```
                   pip install faiss-cpu
                   ```
                2. または、以下のボタンをクリックして自動インストールを試みることができます：
                """
            )

            if st.button("faiss-cpuをインストール"):
                if install_package("faiss-cpu"):
                    st.success(
                        "faiss-cpuのインストールに成功しました。アプリを再起動してください。"
                    )
                else:
                    st.error(
                        "faiss-cpuのインストールに失敗しました。手動でインストールしてください。"
                    )

                    # Pythonバージョンに関する情報を表示
                    st.info(f"Python バージョン: {sys.version}")
                    st.info(
                        "注意: faiss-cpuはPythonバージョンによって異なるインストール方法が必要な場合があります。"
                    )
        else:
            st.error(
                f"ワークフローの初期化中にインポートエラーが発生しました: {str(e)}"
            )
        return None
    except Exception as e:
        logger.error(f"TravelPlannerWorkflowの作成中にエラーが発生: {e}")
        logger.error(traceback.format_exc())
        st.error(f"ワークフローの初期化中にエラーが発生しました: {str(e)}")
        return None


def main():
    logger.info("アプリケーション起動")
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
                B -->|いいえ| E[プラン生成]
                C --> D{リサーチ成功?}
                D -->|はい| N[RAG]
                D -->|いいえ| F[エラー処理]
                N --> O{RAG成功?}
                O -->|はい| E
                O -->|いいえ| E
                E --> G{プラン生成成功?}
                G -->|はい| H[追加情報]
                G -->|いいえ| F
                H --> I{追加情報成功?}
                I -->|はい| J[終了]
                I -->|いいえ| F
                F --> J
            ```
            """
            )

            # # デバッグセクションを追加
            # with st.expander("デバッグ情報"):
            #     if st.button("ログレベルをDEBUGに設定"):
            #         logging.getLogger().setLevel(logging.DEBUG)
            #         logger.debug("ログレベルをDEBUGに設定しました")
            #         st.success(
            #             "ログレベルをDEBUGに設定しました。詳細なログが出力されます。"
            #         )

            #     if st.button("ナレッジベースの再初期化"):
            #         try:
            #             workflow = get_travel_planner_workflow()
            #             if workflow:
            #                 workflow.knowledge_base.initialize_knowledge_base()
            #                 st.success("ナレッジベースを再初期化しました")
            #                 logger.info("ナレッジベースの再初期化に成功")
            #         except Exception as e:
            #             logger.error(f"ナレッジベースの再初期化中にエラー: {e}")
            #             st.error(f"ナレッジベースの再初期化中にエラー: {str(e)}")

            #     # 依存関係の管理セクション
            #     st.subheader("依存関係の管理")

            #     # Pythonバージョン情報
            #     st.info(f"Python バージョン: {sys.version}")

            # # faissのインストール状態チェック
            # try:
            #     import faiss

            #     st.success(f"FAISS インストール済み: {faiss.__version__}")
            # except ImportError:
            #     st.warning("FAISS がインストールされていません")
            #     if st.button("FAISS (faiss-cpu) をインストール"):
            #         if install_package("faiss-cpu"):
            #             st.success(
            #                 "faiss-cpuのインストールに成功しました。アプリを再起動してください。"
            #             )
            #         else:
            #             st.error(
            #                 "faiss-cpuのインストールに失敗しました。手動でインストールしてください。"
            #             )

            # # 現在のベクトルストアの種類を表示
            # if "travel_planner" in st.session_state:
            #     workflow = st.session_state.travel_planner
            #     if (
            #         hasattr(workflow, "knowledge_base")
            #         and workflow.knowledge_base.vector_store
            #     ):
            #         st.info(
            #             f"現在使用中のベクトルストア: {type(workflow.knowledge_base.vector_store).__name__}"
            #         )

        st.caption("© 2023 日本旅行プランナー")

    # メインコンテンツ
    st.title("あなただけの日本旅行プランを作成")

    # タブを作成
    # tab1, tab2 = st.tabs(["旅行プラン生成", "LangSmith実行トレース"])
    (tab1,) = st.tabs(["旅行プラン生成"])

    with tab1:
        # セッション状態の初期化
        if "travel_result" not in st.session_state:
            st.session_state.travel_result = None
        if "form_submitted" not in st.session_state:
            st.session_state.form_submitted = False

        # フォームの表示
        form_data = render_travel_form()

        # フォームが送信された場合
        if form_data and not st.session_state.form_submitted:
            logger.info(f"フォーム送信: 目的地={form_data['destination']}")
            st.session_state.form_submitted = True
            render_loading_state()

            try:
                # 旅行プランナーワークフローの取得
                travel_planner = get_travel_planner_workflow()

                if not travel_planner:
                    st.error("旅行プランナーワークフローの初期化に失敗しました。")
                    logger.error("旅行プランナーワークフローの初期化に失敗")
                    st.session_state.form_submitted = False
                    return

                # セッションに保存
                st.session_state.travel_planner = travel_planner

                # LangGraphワークフローを実行して旅行プランの生成
                logger.info("旅行プラン生成を実行")
                result = travel_planner.generate_travel_plans(
                    current_location=form_data["current_location"],
                    destination=form_data["destination"],
                    budget=form_data["budget"],
                    duration=form_data["duration"],
                    purpose=form_data["purpose"],
                )

                logger.info("旅行プラン生成完了")

                # エラーチェック
                if "error" in result:
                    logger.error(f"旅行プラン生成でエラー: {result['error']}")
                    st.error(
                        f"旅行プラン生成中にエラーが発生しました: {result['error']}"
                    )

                st.session_state.travel_result = result
                st.experimental_rerun()

            except Exception as e:
                logger.error(f"旅行プラン生成中に例外が発生: {e}")
                logger.error(traceback.format_exc())
                st.error(f"エラーが発生しました: {str(e)}")
                st.session_state.form_submitted = False

        # 結果の表示
        if st.session_state.travel_result:
            logger.info("旅行プラン結果を表示")
            render_travel_plans(st.session_state.travel_result)

            # LangSmithトレースURLがある場合は表示
            if "trace_url" in st.session_state.travel_result:
                trace_url = st.session_state.travel_result["trace_url"]
                st.info(
                    f"このプランの生成プロセスの詳細は[LangSmithトレース]({trace_url})で確認できます。"
                )

            # 新しいプランの作成ボタン
            if st.button("新しいプランを作成"):
                logger.info("新しいプラン作成ボタンがクリックされました")
                st.session_state.travel_result = None
                st.session_state.form_submitted = False
                st.experimental_rerun()

    # with tab2:
    #     # LangSmithダッシュボードを表示
    #     render_langsmith_dashboard()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"アプリケーション実行中に未処理の例外が発生: {e}")
        logger.error(traceback.format_exc())
        st.error(f"アプリケーションエラー: {str(e)}")
