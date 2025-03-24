import os
import glob
import logging
import sys
from typing import List, Dict, Any

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 標準出力へのハンドラ
    ],
)
logger = logging.getLogger("RAGKnowledgeBase")
from langchain_community.vectorstores import FAISS

# faissのインポートをtry-exceptで囲む
# try:
#     from langchain_community.vectorstores import FAISS

#     FAISS_AVAILABLE = True
#     logger.info("FAISSのインポートに成功しました")
# except ImportError:
#     FAISS_AVAILABLE = False
#     logger.warning("FAISSのインポートに失敗しました。代替手段を使用します")

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAGKnowledgeBase:
    """旅行プランニングのためのRAGナレッジベースクラス"""

    def __init__(self, knowledge_base_path: str = None, use_openai: bool = True):
        """
        Args:
            knowledge_base_path: ナレッジベースディレクトリのパス
            use_openai: OpenAI埋め込みモデルを使用するかどうか
        """
        # デフォルトのナレッジベースパス
        self.knowledge_base_path = knowledge_base_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "knowledge_base",
        )

        logger.info(f"Python バージョン: {sys.version}")
        logger.info(f"ナレッジベースパス: {self.knowledge_base_path}")
        logger.info(f"OpenAI埋め込みモデルを使用: {use_openai}")

        # 埋め込みモデルの選択
        if use_openai:
            try:
                # OpenAIの埋め込みモデルを使用（APIキーが必要）
                logger.info("OpenAI埋め込みモデルを初期化中...")
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small"  # 小さい方が経済的
                )
                logger.info("OpenAI埋め込みモデルの初期化に成功")
            except Exception as e:
                logger.error(f"OpenAI埋め込みモデルの初期化エラー: {e}")
                raise
        else:
            # HuggingFace埋め込みモデルを使用
            try:
                # 多言語モデルを試す
                logger.info(
                    "HuggingFace埋め込みモデル(distiluse-base-multilingual-cased-v1)を初期化中..."
                )
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="distiluse-base-multilingual-cased-v1"
                )
                logger.info("HuggingFace埋め込みモデルの初期化に成功")
            except Exception as e:
                logger.warning(f"最初のHuggingFaceモデルの読み込みエラー: {e}")
                try:
                    # バックアップとして別の埋め込みを使用
                    logger.info(
                        "代替HuggingFace埋め込みモデル(all-MiniLM-L6-v2)を初期化中..."
                    )
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2"
                    )
                    logger.info("代替HuggingFace埋め込みモデルの初期化に成功")
                except Exception as e2:
                    logger.error(f"代替HuggingFaceモデルの読み込みエラー: {e2}")
                    raise

        # ベクトルストア
        self.vector_store = None

        # ナレッジベースの初期化
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self) -> None:
        """ナレッジベースを初期化し、ベクトルストアを作成する"""
        try:
            # ナレッジベースディレクトリがなければ作成
            os.makedirs(self.knowledge_base_path, exist_ok=True)
            logger.info(f"ナレッジベースディレクトリを確認: {self.knowledge_base_path}")

            # ナレッジベースディレクトリから全てのマークダウンファイルを読み込む
            markdown_files = glob.glob(os.path.join(self.knowledge_base_path, "*.md"))
            logger.info(f"マークダウンファイルを{len(markdown_files)}件見つけました")

            for file in markdown_files:
                logger.info(f"ナレッジファイル: {os.path.basename(file)}")

            # マークダウンファイルが存在する場合、ベクトルストアを作成
            if markdown_files:
                # テキストローダーの作成
                documents = []
                for file_path in markdown_files:
                    try:
                        loader = TextLoader(file_path, encoding="utf-8")
                        file_docs = loader.load()
                        documents.extend(file_docs)
                        logger.info(
                            f"ファイル読み込み成功: {file_path} ({len(file_docs)}ドキュメント)"
                        )
                    except Exception as e:
                        logger.error(f"ファイル読み込みエラー: {file_path} - {e}")

                # テキストスプリッターの作成（日本語に適した設定）
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=[
                        "\n## ",
                        "\n### ",
                        "\n#### ",
                        "\n",
                        "。",
                        "、",
                        " ",
                        "",
                    ],
                )

                logger.info(f"テキスト分割を開始: {len(documents)}ドキュメント")

                # テキストを分割
                chunks = text_splitter.split_documents(documents)
                logger.info(f"テキスト分割完了: {len(chunks)}チャンク")

                # ベクトルストアの作成
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                logger.info(
                    f"FAISSベクトルストア初期化完了: {len(chunks)}チャンクを登録"
                )

            else:
                logger.warning(
                    "警告: ナレッジベースにマークダウンファイルが見つかりません"
                )
        except Exception as e:
            logger.error(f"ナレッジベース初期化エラー: {e}")
            raise

    def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        ナレッジベースに対してクエリを実行し、関連する情報を取得する

        Args:
            query: 検索クエリ
            top_k: 返す結果の最大数

        Returns:
            関連する情報のリスト
        """
        logger.info(f"ナレッジベース検索: '{query}' (top_k={top_k})")

        if self.vector_store is None:
            logger.warning("ベクトルストアが初期化されていません")
            return [{"content": "ナレッジベースが初期化されていません", "source": ""}]

        try:
            # ベクトルストアから類似度検索
            if isinstance(self.vector_store, DocArrayInMemorySearch):
                # DocArrayInMemorySearchはscoreを返さないので、similarity_searchのみを使用
                logger.info("DocArrayInMemorySearchで検索を実行")
                documents = self.vector_store.similarity_search(query, k=top_k)
                results = [(doc, 0.0) for doc in documents]  # ダミースコア
            else:
                # FAISSはscoreを返す
                logger.info("FAISSで検索を実行")
                results = self.vector_store.similarity_search_with_score(query, k=top_k)

            logger.info(f"検索結果: {len(results)}件見つかりました")

            # 検索結果をフォーマット
            formatted_results = []
            for idx, (doc, score) in enumerate(results):
                source = doc.metadata.get("source", "不明")
                content_summary = (
                    doc.page_content[:50] + "..."
                    if len(doc.page_content) > 50
                    else doc.page_content
                )
                logger.info(
                    f"検索結果 {idx+1}: スコア={score:.4f}, ソース={os.path.basename(source)}, 内容={content_summary}"
                )

                formatted_results.append(
                    {
                        "content": doc.page_content,
                        "source": source,
                        "similarity_score": score,
                    }
                )

            return formatted_results
        except Exception as e:
            logger.error(f"ナレッジベース検索エラー: {e}")
            return [
                {
                    "content": f"ナレッジベース検索中にエラーが発生しました: {str(e)}",
                    "source": "",
                }
            ]
