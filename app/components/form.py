import streamlit as st


def render_travel_form():
    """旅行プランのためのフォームを表示"""
    with st.form("travel_plan_form"):
        st.subheader("あなたの旅行条件を入力してください")

        col1, col2 = st.columns(2)

        with col1:
            current_location = st.text_input("現在地", "東京")
            destination = st.text_input("目的地", "京都")

        with col2:
            budget_options = [
                "~5万円",
                "5万円~10万円",
                "10万円~15万円",
                "15万円~20万円",
                "20万円~",
            ]
            budget = st.selectbox("予算", budget_options)

            duration_options = [
                "日帰り",
                "1泊2日",
                "2泊3日",
                "3泊4日",
                "4泊5日",
                "5泊以上",
            ]
            duration = st.selectbox("滞在期間", duration_options)

        purpose_options = [
            "観光",
            "グルメ",
            "温泉",
            "自然",
            "歴史・文化",
            "ショッピング",
            "その他",
        ]
        purpose = st.multiselect("旅行の目的", purpose_options)

        additional_requests = st.text_area("その他のリクエスト", "")

        submit_button = st.form_submit_button("旅行プランを生成")

        if submit_button:
            return {
                "current_location": current_location,
                "destination": destination,
                "budget": budget,
                "duration": duration,
                "purpose": ", ".join(purpose) if purpose else "観光",
                "additional_requests": additional_requests,
            }

        return None
