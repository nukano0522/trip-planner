import os
import streamlit as st
from langsmith import Client
from langsmith.schemas import Run, RunTree
from typing import List, Dict, Any, Optional


def get_langsmith_client() -> Optional[Client]:
    """LangSmith APIクライアントを取得する"""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return None

    try:
        return Client(api_key=api_key)
    except Exception as e:
        print(f"LangSmith APIクライアントの初期化エラー: {e}")
        return None


def get_latest_runs(project_name: str = None, limit: int = 5) -> List[RunTree]:
    """指定したプロジェクトの最新の実行トレースを取得する"""
    client = get_langsmith_client()
    if not client:
        return []

    project_name = project_name or os.getenv("LANGSMITH_PROJECT")
    if not project_name:
        return []

    try:
        # 最新の実行を取得
        runs = client.list_runs(
            project_name=project_name,
            execution_order=1,  # トップレベルの実行のみ
            limit=limit,
        )

        # RunTreeに変換
        run_trees = []
        for run in runs:
            try:
                tree = client.get_run_tree(run.id)
                run_trees.append(tree)
            except Exception as e:
                print(f"実行トレースの取得エラー {run.id}: {e}")

        return run_trees
    except Exception as e:
        print(f"LangSmith実行リストの取得エラー: {e}")
        return []


def render_run_info(run: Run) -> None:
    """実行情報をStreamlitに表示する"""
    with st.expander(f"実行: {run.name} ({run.id[:8]}...)"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**開始時間:**", run.start_time)
            st.write(
                "**所要時間:**",
                f"{(run.end_time - run.start_time).total_seconds():.2f}秒",
            )
            st.write("**ステータス:**", run.status)

        with col2:
            st.write("**入力:**")
            st.json(run.inputs)

        st.write("**出力:**")
        st.json(run.outputs)

        if run.error:
            st.error(f"エラー: {run.error}")


def render_langsmith_dashboard(project_name: str = None) -> None:
    """LangSmithのダッシュボードを表示する"""
    project_name = project_name or os.getenv("LANGSMITH_PROJECT")

    st.subheader("🔍 LangSmith実行トレース")

    if not os.getenv("LANGSMITH_API_KEY"):
        st.warning(
            "LangSmith APIキーが設定されていません。.envファイルにLANGSMITH_API_KEYを追加してください。"
        )
        return

    if not project_name:
        st.warning(
            "LangSmithプロジェクト名が設定されていません。.envファイルにLANGSMITH_PROJECTを追加してください。"
        )
        return

    # 最新の実行を取得
    run_trees = get_latest_runs(project_name)

    if not run_trees:
        st.info(
            "実行トレースがまだありません。旅行プランを生成すると、ここにトレースが表示されます。"
        )
        return

    # 実行トレースを表示
    for i, run_tree in enumerate(run_trees):
        st.markdown(f"### 実行 #{i+1}")
        render_run_info(run_tree.root)

        # 子ノードの表示
        if run_tree.root.child_runs:
            st.markdown("#### ワークフローステップ")
            for child in run_tree.root.child_runs:
                render_run_info(child)

    # LangSmithダッシュボードへのリンク
    st.markdown("---")
    langsmith_url = f"https://smith.langchain.com/projects/{project_name}"
    st.markdown(
        f"詳細な実行トレースは [LangSmithダッシュボード]({langsmith_url}) で確認できます。"
    )


def get_langsmith_trace_url(run_id: str) -> str:
    """LangSmithでのトレースURLを取得する"""
    return f"https://smith.langchain.com/traces/{run_id}"
