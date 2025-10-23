from pydantic import BaseModel, PrivateAttr
from typing import Optional, Type, Tuple, List
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.callbacks import CallbackManagerForToolRun
import os
import requests
from .vectorizer_cnlthd import run_vectorizer
from dotenv import load_dotenv
from pathlib import Path
import logging

load_dotenv()

# Đây là file của RetrieverTool hay RAG
class RetrieveInput(BaseModel):
    query: str

class RetrieverTool(BaseTool):
    name: str = "retrieve_tool"
    description: str = (
        "Công cụ này được sử dụng để truy xuất và cung cấp thông tin đáng tin cậy "
        "liên quan đến tâm lý học và sức khỏe tinh thần. "
        "Nó hỗ trợ người dùng hiểu rõ hơn về cảm xúc, nguyên nhân của căng thẳng, "
        "và các kỹ năng ứng phó hoặc tư duy tích cực dựa trên tài liệu chuyên ngành. "
        "Lưu ý: Công cụ chỉ mang tính chất tham khảo, không thay thế cho tư vấn hoặc trị liệu tâm lý chuyên nghiệp."
    )
    args_schema: Type[BaseModel] = RetrieveInput
    return_direct: bool = False

    _vector_store: Optional[FAISS] = PrivateAttr(default=None)
    _k: int = PrivateAttr(default=3)
    _llm_url: str = PrivateAttr(default="http://localhost:8000/")
    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    def __init__(self, vector_store_path: Optional[str] = None,
                 k: int = 1,
                 llm_url: str = "http://localhost:8000/",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # configure basic attributes
        object.__setattr__(self, "_k", k)
        object.__setattr__(self, "_llm_url", llm_url.rstrip('/') + '/')

        # configure logger
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

        # determine default vector store path relative to package if not provided
        if vector_store_path is None:
            base = Path(__file__).resolve().parents[1]
            vector_store_path = str((base / 'data' / 'vector_store').resolve())

        try:
            if not os.path.exists(vector_store_path):
                self._logger.info("Vector store chưa tồn tại, đang tạo mới...")
                run_vectorizer()
            else:
                self._logger.info("Vector store đã tồn tại, bỏ qua bước vectorizer.")

            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

            if not os.path.exists(vector_store_path):
                raise FileNotFoundError(f"Vector store path không tồn tại: {vector_store_path}")

            vector_store = FAISS.load_local(
                vector_store_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
            object.__setattr__(self, "_vector_store", vector_store)
        except Exception as e:
            self._logger.exception(f"Lỗi khi load vector store: {e}")
            raise

    def _returndocs(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
         # Hàm lấy những tài liệu liên quan đến câu hỏi
        if self._vector_store is None:
            return "Lỗi: Vector store chưa được khởi tạo"

        try:
            results = self._vector_store.similarity_search(query, k=self._k)
            documents = [getattr(doc, 'page_content', str(doc)) for doc in results]

            if not documents:
                return "Không tìm thấy thông tin phù hợp"
            return "\n\n".join(documents)

        except Exception as e:
            self._logger.exception('Lỗi khi truy xuất thông tin')
            return f"Lỗi khi truy xuất thông tin: {str(e)}"
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Hàm chạy chính
        self._logger.info("đang gọi tới RAG")
        context = self._returndocs(query)
        # print("context: ", context)
        prompt = f"""Dưới đây là một số tài liệu liên quan:\n{context}\n\n hãy dựa vào tài liệu và trả lời cho chia sẻ của người dùng: {query}"""

        try:
            llm_response = requests.post(self._llm_url + "/response", json={
                "message": prompt
            })
            self._last_response=llm_response.json()
            return self._last_response.get("content", " Không có phản hồi ")
        except Exception as e:
            return f"Lỗi gọi LLM Tool: {str(e)}"
