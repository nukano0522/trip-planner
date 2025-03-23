from typing import Dict, List, Any, TypedDict, Annotated, Literal
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper

from langgraph.graph import StateGraph, END, START


# ステート（状態）の型定義
class TravelPlanningState(TypedDict):
    """旅行プランニングの状態を表す型定義"""

    current_location: str
    destination: str
    budget: str
    duration: str
    purpose: str
    research_done: bool
    research_results: Dict[str, str]
    travel_plan: str
    additional_info: str
    error: str
    next_step: Literal["research", "plan_generation", "recommendation", "end"]


# 各ノードの名前を定義
class TravelPlanningNodes(str, Enum):
    RESEARCH = "research"
    PLAN_GENERATION = "plan_generation"
    RECOMMENDATION = "recommendation"
    ERROR_HANDLER = "error_handler"


class TravelPlannerWorkflow:
    def __init__(self, openai_api_key: str, serpapi_key: str = None):
        """旅行プランニングワークフローの初期化"""
        self.openai_api_key = openai_api_key
        self.serpapi_key = serpapi_key

        # 各ステップで使用するLLMを初期化
        self.llm = ChatOpenAI(
            temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key
        )

        # 検索用のインスタンスを作成
        self.wikipedia = WikipediaAPIWrapper(lang="ja")

        # SerpAPIラッパーも初期化
        self.serpapi_wrapper = None
        if serpapi_key:
            self.serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key)

        # ワークフローグラフを構築
        self.workflow = self._build_workflow()

    def _research(self, state: TravelPlanningState) -> TravelPlanningState:
        """目的地に関する情報を収集するノード"""
        try:
            destination = state["destination"]

            # 検索クエリの作成
            wiki_query = f"{destination}の観光情報、見どころ、アクセス"

            # Wikipedia検索を実行
            wiki_result = self.wikipedia.run(wiki_query)

            # 検索結果を状態に格納
            research_results = {
                "wikipedia": wiki_result,
            }

            # SerpAPIキーがある場合はWeb検索も実行
            if self.serpapi_wrapper:
                web_query = f"{destination} 観光 おすすめ スポット 2023"
                web_result = self.serpapi_wrapper.run(web_query)
                research_results["web_search"] = web_result

            # 状態を更新
            return {
                **state,
                "research_done": True,
                "research_results": research_results,
                "next_step": "plan_generation",
            }
        except Exception as e:
            return {
                **state,
                "error": f"研究中にエラーが発生しました: {str(e)}",
                "next_step": "error_handler",
            }

    def _plan_generation(self, state: TravelPlanningState) -> TravelPlanningState:
        """収集した情報に基づいて旅行プランを生成するノード"""
        try:
            # プロンプトの作成
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="""
                あなたは日本の旅行プランを提案する専門家です。
                提供された情報に基づいて、最適な旅行プランを3つ提案してください。
                各プランには以下の情報を含めてください：
                - プランの概要と特徴
                - 訪問する場所のリスト（各場所の簡単な説明を含む）
                - おすすめの宿泊施設
                - 食事のおすすめ
                - 予想される費用の内訳
                - 季節に合わせたアドバイス
                
                回答は日本語でマークダウン形式にしてください。
                """
                    ),
                    HumanMessage(
                        content=f"""
                以下の条件と収集した情報に基づいて旅行プランを作成してください。
                
                現在地: {state["current_location"]}
                目的地: {state["destination"]}
                予算: {state["budget"]}
                滞在期間: {state["duration"]}
                旅行の目的: {state["purpose"]}
                
                収集した情報:
                {state["research_results"].get("wikipedia", "情報なし")}
                
                {state["research_results"].get("web_search", "")}
                """
                    ),
                ]
            )

            # format_promptメソッドを使ってPromptValueを取得
            prompt_value = prompt.format_prompt()

            # LLMを使用してプランを生成
            response = self.llm.invoke(prompt_value.to_messages())

            # 生成されたプランを状態に格納
            return {
                **state,
                "travel_plan": response.content,
                "next_step": "recommendation",
            }
        except Exception as e:
            return {
                **state,
                "error": f"旅行プラン生成中にエラーが発生しました: {str(e)}",
                "next_step": "error_handler",
            }

    def _recommendation(self, state: TravelPlanningState) -> TravelPlanningState:
        """追加のアドバイスや観光情報を提供するノード"""
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="""
                あなたは日本旅行のエキスパートです。
                旅行プランに加えて、追加のアドバイスや現地の最新情報、文化的なヒントなどを提供してください。
                回答は日本語でマークダウン形式にしてください。
                """
                    ),
                    HumanMessage(
                        content=f"""
                以下の旅行条件について、追加のアドバイスや現地の最新情報、文化的なヒントなどを提供してください：
                
                目的地: {state["destination"]}
                滞在期間: {state["duration"]}
                旅行の目的: {state["purpose"]}
                予算: {state["budget"]}
                
                特に以下の点について触れてください：
                - 現地の気候と服装のアドバイス
                - 現地の交通手段
                - 現地のマナーや慣習
                - おすすめのお土産
                - 旅行保険や安全に関するアドバイス
                """
                    ),
                ]
            )

            # format_promptメソッドを使ってPromptValueを取得
            prompt_value = prompt.format_prompt()

            # LLMを使用して追加情報を生成
            response = self.llm.invoke(prompt_value.to_messages())

            # 生成された追加情報を状態に格納
            return {**state, "additional_info": response.content, "next_step": "end"}
        except Exception as e:
            return {
                **state,
                "error": f"追加情報生成中にエラーが発生しました: {str(e)}",
                "next_step": "error_handler",
            }

    def _error_handler(self, state: TravelPlanningState) -> TravelPlanningState:
        """エラーハンドリングノード"""
        error_message = state.get("error", "不明なエラーが発生しました")

        # エラーが発生した場合でも最小限の情報を提供
        fallback_plan = f"""
        # 旅行プラン生成中にエラーが発生しました
        
        申し訳ありませんが、以下のエラーにより完全な旅行プランを生成できませんでした：
        
        ```
        {error_message}
        ```
        
        ### 基本的な{state["destination"]}旅行情報
        
        * 滞在期間: {state["duration"]}
        * 予算: {state["budget"]}
        * 目的: {state["purpose"]}
        
        一般的な{state["destination"]}旅行のアドバイス：
        
        1. 事前に主要な観光スポットを調査してください
        2. 現地の天気に適した服装を準備してください
        3. 現地の交通手段を確認してください
        4. 旅行保険への加入を検討してください
        """

        return {**state, "travel_plan": fallback_plan, "next_step": "end"}

    def _should_research(
        self, state: TravelPlanningState
    ) -> Literal["research", "plan_generation"]:
        """研究ステップに進むべきかを判断するルーター"""
        if not state.get("research_done", False):
            return "research"
        return "plan_generation"

    def _router(self, state: TravelPlanningState) -> str:
        """次のステップを決定するルーター"""
        return state["next_step"]

    def _build_workflow(self) -> StateGraph:
        """ワークフローグラフを構築"""
        # 新しいグラフを作成
        workflow = StateGraph(TravelPlanningState)

        # ノードを追加
        workflow.add_node(TravelPlanningNodes.RESEARCH, self._research)
        workflow.add_node(TravelPlanningNodes.PLAN_GENERATION, self._plan_generation)
        workflow.add_node(TravelPlanningNodes.RECOMMENDATION, self._recommendation)
        workflow.add_node(TravelPlanningNodes.ERROR_HANDLER, self._error_handler)

        # エッジを追加（状態に基づいてノード間のルーティングを定義）
        workflow.add_conditional_edges(
            START,
            self._should_research,
            {
                "research": TravelPlanningNodes.RESEARCH,
                "plan_generation": TravelPlanningNodes.PLAN_GENERATION,
            },
        )

        workflow.add_conditional_edges(
            TravelPlanningNodes.RESEARCH,
            self._router,
            {
                "plan_generation": TravelPlanningNodes.PLAN_GENERATION,
                "error_handler": TravelPlanningNodes.ERROR_HANDLER,
            },
        )

        workflow.add_conditional_edges(
            TravelPlanningNodes.PLAN_GENERATION,
            self._router,
            {
                "recommendation": TravelPlanningNodes.RECOMMENDATION,
                "error_handler": TravelPlanningNodes.ERROR_HANDLER,
            },
        )

        workflow.add_conditional_edges(
            TravelPlanningNodes.RECOMMENDATION,
            self._router,
            {"end": END, "error_handler": TravelPlanningNodes.ERROR_HANDLER},
        )

        workflow.add_edge(TravelPlanningNodes.ERROR_HANDLER, END)

        # グラフをコンパイル
        return workflow.compile()

    def generate_travel_plans(
        self,
        current_location: str,
        destination: str,
        budget: str,
        duration: str,
        purpose: str,
    ) -> Dict[str, str]:
        """旅行プランを生成する"""
        try:
            # 初期状態を設定
            initial_state = TravelPlanningState(
                current_location=current_location,
                destination=destination,
                budget=budget,
                duration=duration,
                purpose=purpose,
                research_done=False,
                research_results={},
                travel_plan="",
                additional_info="",
                error="",
                next_step="research",
            )

            # ワークフローを実行
            final_state = self.workflow.invoke(initial_state)

            # 結果を返す
            return {
                "travel_plans": final_state["travel_plan"],
                "additional_info": final_state["additional_info"],
            }
        except Exception as e:
            return {"error": f"旅行プランの生成中にエラーが発生しました: {str(e)}"}
