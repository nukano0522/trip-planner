from typing import Dict, List, Any, TypedDict, Annotated, Literal
from enum import Enum
import os
import uuid
import logging
import traceback

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

from langgraph.graph import StateGraph, END, START

from app.utils.rag_utils import RAGKnowledgeBase

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 標準出力へのハンドラ
    ],
)
logger = logging.getLogger("TravelPlannerWorkflow")


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
    rag_results: List[Dict[str, Any]]  # RAG検索結果を格納するフィールドを追加
    travel_plan: str
    additional_info: str
    error: str
    next_step: Literal["research", "rag", "plan_generation", "recommendation", "end"]


# 各ノードの名前を定義
class TravelPlanningNodes(str, Enum):
    RESEARCH = "research"
    RAG = "rag"  # RAGノードを追加
    PLAN_GENERATION = "plan_generation"
    RECOMMENDATION = "recommendation"
    ERROR_HANDLER = "error_handler"


class TravelPlannerWorkflow:
    def __init__(self, openai_api_key: str, serpapi_key: str = None):
        """旅行プランニングワークフローの初期化"""
        logger.info("TravelPlannerWorkflowの初期化を開始")
        self.openai_api_key = openai_api_key
        self.serpapi_key = serpapi_key

        # トレーシング設定
        self.project_name = os.getenv("LANGSMITH_PROJECT", "trip-planner-japan")
        self.tracing_enabled = (
            os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true"
        )

        try:
            # 各ステップで使用するLLMを初期化
            logger.info("ChatOpenAIの初期化")
            model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            logger.info(f"使用するモデル: {model_name}")
            self.llm = ChatOpenAI(
                temperature=0.7,
                model_name=model_name,
                openai_api_key=openai_api_key,
            )

            # 検索用のインスタンスを作成
            logger.info("WikipediaAPIWrapperの初期化")
            self.wikipedia = WikipediaAPIWrapper(lang="ja")

            # SerpAPIラッパーも初期化
            self.serpapi_wrapper = None
            if serpapi_key:
                logger.info("SerpAPIWrapperの初期化")
                self.serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key)
            else:
                logger.warning("SerpAPI APIキーが設定されていないため、Web検索は無効")

            # RAGナレッジベースの初期化 - OpenAI埋め込みを使用
            logger.info("RAGナレッジベースの初期化")
            self.knowledge_base = RAGKnowledgeBase(use_openai=True)

            # ワークフローグラフを構築
            logger.info("ワークフローグラフの構築")
            self.workflow = self._build_workflow()
            logger.info("TravelPlannerWorkflowの初期化完了")
        except Exception as e:
            logger.error(f"TravelPlannerWorkflowの初期化中にエラーが発生: {e}")
            logger.error(traceback.format_exc())
            raise

    def _research(self, state: TravelPlanningState) -> TravelPlanningState:
        """目的地に関する情報を収集するノード"""
        logger.info(f"researchノード開始: 目的地={state['destination']}")
        try:
            destination = state["destination"]

            # 検索クエリの作成
            wiki_query = f"{destination}の観光情報、見どころ、アクセス"
            logger.info(f"Wikipedia検索クエリ: {wiki_query}")

            # Wikipedia検索を実行
            wiki_result = self.wikipedia.run(wiki_query)
            logger.info(f"Wikipedia検索結果: {len(wiki_result)} 文字")

            # 検索結果を状態に格納
            research_results = {
                "wikipedia": wiki_result,
            }

            # SerpAPIキーがある場合はWeb検索も実行
            if self.serpapi_wrapper:
                web_query = f"{destination} 観光 おすすめ スポット 2024"
                logger.info(f"Web検索クエリ: {web_query}")
                web_result = self.serpapi_wrapper.run(web_query)
                logger.info(f"Web検索結果: {len(web_result)} 文字")
                research_results["web_search"] = web_result

            # 状態を更新
            logger.info("researchノード完了: 次のステップ=rag")
            return {
                **state,
                "research_done": True,
                "research_results": research_results,
                "next_step": "rag",  # 次のステップをRAGに変更
            }
        except Exception as e:
            logger.error(f"researchノードでエラー発生: {e}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": f"research中にエラーが発生しました: {str(e)}",
                "next_step": "error_handler",
            }

    def _rag(self, state: TravelPlanningState) -> TravelPlanningState:
        """内部ナレッジベースからの情報検索を行うRAGノード"""
        logger.info(f"RAGノード開始: 目的地={state['destination']}")
        try:
            destination = state["destination"]
            purpose = state["purpose"]
            duration = state["duration"]

            # RAG検索クエリの作成
            rag_query = f"{destination}の旅行情報 {purpose} 滞在期間:{duration}"
            logger.info(f"RAG検索クエリ: {rag_query}")

            # ナレッジベースに問い合わせ
            rag_results = self.knowledge_base.query_knowledge_base(rag_query, top_k=3)
            logger.info(f"RAG検索結果: {len(rag_results)}件")

            # RAG検索結果を状態に格納
            logger.info("RAGノード完了: 次のステップ=plan_generation")
            return {
                **state,
                "rag_results": rag_results,
                "next_step": "plan_generation",
            }
        except Exception as e:
            logger.error(f"RAGノードでエラー発生: {e}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": f"内部ナレッジベース検索中にエラーが発生しました: {str(e)}",
                "rag_results": [],
                "next_step": "plan_generation",  # エラーがあっても次のステップに進む
            }

    def _plan_generation(self, state: TravelPlanningState) -> TravelPlanningState:
        """収集した情報に基づいて旅行プランを生成するノード"""
        logger.info(f"プラン生成ノード開始: 目的地={state['destination']}")
        try:
            # RAG結果からのコンテンツを抽出
            rag_content = ""
            if state.get("rag_results"):
                logger.info(f"RAG結果を処理: {len(state['rag_results'])}件")
                for idx, result in enumerate(state["rag_results"]):
                    rag_content += (
                        f"\n内部ナレッジベース {idx+1}:\n{result['content']}\n"
                    )
                    source = result.get("source", "不明")
                    if isinstance(source, str) and os.path.exists(source):
                        source = os.path.basename(source)
                    logger.info(
                        f"RAG結果 {idx+1}: ソース={source}, スコア={result.get('similarity_score', 'N/A')}"
                    )

            # プロンプトの作成
            logger.info("プロンプトを作成")
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
                
                外部から収集した情報:
                {state["research_results"].get("wikipedia", "情報なし")}
                
                {state["research_results"].get("web_search", "")}
                
                内部ナレッジベースからの情報:
                {rag_content}
                
                外部情報と内部ナレッジを組み合わせて、最適な旅行プランを作成してください。
                """
                    ),
                ]
            )

            # format_promptメソッドを使ってPromptValueを取得
            logger.info("プロンプトをフォーマット")
            prompt_value = prompt.format_prompt()

            # LLMを使用してプランを生成
            logger.info("LLMを呼び出してプラン生成")
            response = self.llm.invoke(prompt_value.to_messages())
            logger.info(f"LLM応答: {len(response.content)} 文字")

            # 生成されたプランを状態に格納
            logger.info("プラン生成ノード完了: 次のステップ=recommendation")
            return {
                **state,
                "travel_plan": response.content,
                "next_step": "recommendation",
            }
        except Exception as e:
            logger.error(f"プラン生成ノードでエラー発生: {e}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": f"旅行プラン生成中にエラーが発生しました: {str(e)}",
                "next_step": "error_handler",
            }

    def _recommendation(self, state: TravelPlanningState) -> TravelPlanningState:
        """追加のアドバイスや観光情報を提供するノード"""
        logger.info(f"レコメンデーションノード開始: 目的地={state['destination']}")
        try:
            logger.info("追加情報プロンプトを作成")
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
            logger.info("レコメンデーションプロンプトをフォーマット")
            prompt_value = prompt.format_prompt()

            # LLMを使用して追加情報を生成
            logger.info("LLMを呼び出して追加情報生成")
            response = self.llm.invoke(prompt_value.to_messages())
            logger.info(f"LLM応答: {len(response.content)} 文字")

            # 生成された追加情報を状態に格納
            logger.info("レコメンデーションノード完了: 次のステップ=end")
            return {**state, "additional_info": response.content, "next_step": "end"}
        except Exception as e:
            logger.error(f"レコメンデーションノードでエラー発生: {e}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": f"追加情報生成中にエラーが発生しました: {str(e)}",
                "next_step": "error_handler",
            }

    def _error_handler(self, state: TravelPlanningState) -> TravelPlanningState:
        """エラーハンドリングノード"""
        error_message = state.get("error", "不明なエラーが発生しました")
        logger.error(f"エラーハンドラーノード実行: {error_message}")

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

        logger.info("エラーハンドラーノード完了: フォールバックプラン生成")
        return {**state, "travel_plan": fallback_plan, "next_step": "end"}

    def _should_research(
        self, state: TravelPlanningState
    ) -> Literal["research", "plan_generation"]:
        """researchステップに進むべきかを判断するルーター"""
        if not state.get("research_done", False):
            logger.info("ルーター: research→research")
            return "research"
        logger.info("ルーター: research→plan_generation")
        return "plan_generation"

    def _router(self, state: TravelPlanningState) -> str:
        """次のステップを決定するルーター"""
        next_step = state["next_step"]
        logger.info(f"ルーター: 次のステップ={next_step}")
        return next_step

    def _build_workflow(self) -> StateGraph:
        """ワークフローグラフを構築"""
        logger.info("ワークフローグラフの構築開始")
        try:
            # 新しいグラフを作成
            workflow = StateGraph(TravelPlanningState)
            logger.info("グラフオブジェクト作成完了")

            # ノードを追加
            logger.info("ノードを追加")
            workflow.add_node(TravelPlanningNodes.RESEARCH, self._research)
            workflow.add_node(TravelPlanningNodes.RAG, self._rag)  # RAGノードを追加
            workflow.add_node(
                TravelPlanningNodes.PLAN_GENERATION, self._plan_generation
            )
            workflow.add_node(TravelPlanningNodes.RECOMMENDATION, self._recommendation)
            workflow.add_node(TravelPlanningNodes.ERROR_HANDLER, self._error_handler)

            # エッジを追加（状態に基づいてノード間のルーティングを定義）
            logger.info("エッジを追加")
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
                    "rag": TravelPlanningNodes.RAG,  # リサーチからRAGへのエッジを追加
                    "error_handler": TravelPlanningNodes.ERROR_HANDLER,
                },
            )

            workflow.add_conditional_edges(
                TravelPlanningNodes.RAG,
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

            # LangSmithチェックポインターの設定
            logger.info("LangSmithチェックポインターの設定")
            if self.tracing_enabled:
                try:
                    logger.info("LangSmithトレース有効")
                    # 最新のlanggraphバージョンではチェックポインターは不要
                    # 環境変数でLangSmith統合が設定されていれば自動的に有効になります
                except Exception as e:
                    logger.error(f"LangSmithトレーサーの設定エラー: {e}")

            # グラフをコンパイル
            logger.info("グラフをコンパイル")
            compiled_workflow = workflow.compile()
            logger.info("ワークフローグラフの構築完了")
            return compiled_workflow
        except Exception as e:
            logger.error(f"ワークフローグラフの構築中にエラーが発生: {e}")
            logger.error(traceback.format_exc())
            raise

    def generate_travel_plans(
        self,
        current_location: str,
        destination: str,
        budget: str,
        duration: str,
        purpose: str,
    ) -> Dict[str, str]:
        """旅行プランを生成する"""
        logger.info(
            f"旅行プラン生成開始: 目的地={destination}, 予算={budget}, 期間={duration}"
        )
        try:

            # LangChainトレーサーの初期化
            tracer = None
            if self.tracing_enabled:
                try:
                    logger.info("LangChainトレーサーを初期化")
                    tracer = LangChainTracer(project_name=self.project_name)
                    logger.info(
                        f"LangChainトレーサー初期化完了: プロジェクト={self.project_name}"
                    )
                    logger.info(
                        f"LangChainトレーサー初期化完了: プロジェクト={self.project_name}"
                    )
                except Exception as e:
                    logger.error(f"LangChainトレーサーの初期化エラー: {e}")

            # 初期状態を設定
            logger.info("初期状態を設定")
            initial_state = TravelPlanningState(
                current_location=current_location,
                destination=destination,
                budget=budget,
                duration=duration,
                purpose=purpose,
                research_done=False,
                research_results={},
                rag_results=[],  # RAG結果の初期値を追加
                travel_plan="",
                additional_info="",
                error="",
                next_step="research",
            )

            # ワークフローを実行
            logger.info("ワークフローを実行")
            callbacks = [tracer] if tracer else None
            final_state = self.workflow.invoke(
                initial_state, config={"callbacks": callbacks}
            )
            logger.info("ワークフロー実行完了")

            # トレーシングの完了を待機
            if self.tracing_enabled and tracer:
                logger.info("トレーシングの完了を待機")
                wait_for_all_tracers()
                logger.info("トレーシング完了")

            # 結果を返す
            result = {
                "travel_plans": final_state["travel_plan"],
                "additional_info": final_state["additional_info"],
            }

            # エラーがあれば記録
            if final_state.get("error"):
                logger.error(f"最終状態にエラーあり: {final_state['error']}")
                result["error"] = final_state["error"]

            # トレーシング情報も追加
            # if self.tracing_enabled:
            # trace_url = f"https://smith.langchain.com/traces/{self.run_id}"
            # result["trace_url"] = trace_url
            # result["run_id"] = self.run_id
            # logger.info(f"トレースURL: {trace_url}")

            logger.info("旅行プラン生成完了")
            return result
        except Exception as e:
            logger.error(f"旅行プラン生成中に例外が発生: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"旅行プランの生成中にエラーが発生しました: {str(e)}"}
