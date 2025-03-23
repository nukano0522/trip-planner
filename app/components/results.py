import streamlit as st


def render_loading_state():
    """ローディング状態を表示"""
    with st.spinner("旅行プランを生成しています..."):
        st.info("この処理には1分程度かかる場合があります。しばらくお待ちください。")


def render_travel_plans(result):
    """旅行プランの結果を表示"""
    if "error" in result:
        st.error(result["error"])
        return

    st.success("旅行プランが生成されました！")

    # メインの旅行プラン
    with st.expander("提案された旅行プラン", expanded=True):
        st.markdown(result["travel_plans"])

    # 追加情報
    if "additional_info" in result and result["additional_info"]:
        with st.expander("追加情報"):
            st.markdown(result["additional_info"])

    # 免責事項
    st.info(
        """
    **免責事項**: このプランはAIによって生成されたものです。
    実際の旅行計画を立てる際は、最新の情報や状況を確認することをお勧めします。
    特に予算や営業時間、交通状況などは変動する可能性があります。
    """
    )
