import os
from dotenv import load_dotenv
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model phản hồi giống như bên API
class MessageOut(BaseModel):
    content: str
    sender: str
    timestamp: str

class SearchWebInput(BaseModel):
    query: str = Field(
        ...,
        description="Chủ đề mà người dùng quan tâm, ví dụ: 'động lực học tập', 'niềm tin bản thân', 'nguồn cảm hứng sống'."
    )

class SearchWebTool(BaseTool):
    name: str = "search_web"
    description: str = (
        "Công cụ tìm kiếm các câu chuyện, bộ phim, sách hoặc âm nhạc truyền cảm hứng, "
        "giúp người dùng thư giãn, tìm lại động lực và suy nghĩ tích cực về cuộc sống."
    )
    args_schema: Type[BaseModel] = SearchWebInput
    return_direct: bool = False

    def _run(self, query: str, run_manager: Optional = None) -> MessageOut:
        """Thực hiện truy vấn Tavily và trả về 3 kết quả đầu tiên"""

        logger.info("🔍 Đang gọi tới Tavily Search...")

        try:
            tavily = TavilySearch(api_key=TAVILY_API_KEY)
            results = tavily.invoke(query)

            if not results or "results" not in results or len(results["results"]) == 0:
                return MessageOut(
                    content="Không có kết quả phù hợp. Bạn có muốn nói thêm về cảm xúc của mình để tôi có thể giúp không?",
                    sender="bot",
                    timestamp=datetime.now().isoformat()
                )

            formatted_results = []
            for item in results["results"][:3]:
                title = item.get("title", "Không có tiêu đề")
                snippet = item.get("content", "Không có mô tả")
                url = item.get("url", "")
                formatted_results.append(f"- **{title}**\n  {snippet}\n  🔗 {url}\n")

            summary = "\n".join(formatted_results)
            msg = f"🌐 Kết quả tìm kiếm cho **'{query}'**:\n\n{summary}"

            return MessageOut(
                content=msg,
                sender="bot",
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            error_msg = f"⚠️ Lỗi khi tìm kiếm với Tavily: {str(e)}"
            return MessageOut(
                content=error_msg,
                sender="bot",
                timestamp=datetime.now().isoformat()
            )