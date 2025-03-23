from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType
import os


class TravelPlannerService:
    def __init__(self, openai_api_key, serpapi_key=None):
        self.openai_api_key = openai_api_key
        self.serpapi_key = serpapi_key

        # 修正：最新バージョンに合わせてChatOpenAIの初期化方法を変更
        self.llm = ChatOpenAI(
            temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key
        )

        # ツールの初期化
        self.tools = self._initialize_tools()

        # エージェントの初期化
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

        # 旅行プラン生成用のチェーン
        self.travel_plan_chain = self._create_travel_plan_chain()

    def _initialize_tools(self):
        """利用可能なツールを初期化"""
        tools = []

        # Wikipedia検索ツール
        wikipedia = WikipediaAPIWrapper(lang="ja")
        wiki_tool = Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="日本語のWikipediaで特定の場所や観光地について検索するときに使用します。",
        )
        tools.append(wiki_tool)

        # Web検索ツール (SerpAPI)
        if self.serpapi_key:
            search = SerpAPIWrapper(serpapi_api_key=self.serpapi_key)
            search_tool = Tool(
                name="Search",
                func=search.run,
                description="インターネットで最新の旅行情報、観光地、ホテル、レストランなどを検索するときに使用します。",
            )
            tools.append(search_tool)

        return tools

    def _create_travel_plan_chain(self):
        """旅行プラン作成のためのチェーンを作成"""
        prompt_template = """
        あなたは日本の旅行プランを提案する専門家です。以下の条件に基づいて、最適な旅行プランを3つ提案してください。
        
        現在地: {current_location}
        目的地: {destination}
        予算: {budget}
        滞在期間: {duration}
        旅行の目的: {purpose}
        
        各プランには以下の情報を含めてください：
        - プランの概要と特徴
        - 訪問する場所のリスト（各場所の簡単な説明を含む）
        - おすすめの宿泊施設
        - 食事のおすすめ
        - 予想される費用の内訳
        - 季節に合わせたアドバイス
        
        回答は日本語でお願いします。
        """

        prompt = PromptTemplate(
            input_variables=[
                "current_location",
                "destination",
                "budget",
                "duration",
                "purpose",
            ],
            template=prompt_template,
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def generate_travel_plans(
        self, current_location, destination, budget, duration, purpose
    ):
        """旅行プランを生成する"""
        try:
            # まずは基本的な旅行プランを生成
            plan_result = self.travel_plan_chain.run(
                current_location=current_location,
                destination=destination,
                budget=budget,
                duration=duration,
                purpose=purpose,
            )

            # エージェントを使用して追加情報を取得（時間がかかるため省略可能）
            agent_query = f"{destination}の観光情報、おすすめスポット、現在のイベント情報を教えてください。"
            additional_info = ""

            try:
                additional_info = self.agent.run(agent_query)
            except Exception as e:
                additional_info = f"追加情報の取得中にエラーが発生しました: {str(e)}"

            # 最終的な結果を返す
            return {"travel_plans": plan_result, "additional_info": additional_info}

        except Exception as e:
            return {"error": f"旅行プランの生成中にエラーが発生しました: {str(e)}"}
