from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import requests
import logging


# Class cho tool llm
class LLMInput(BaseModel):
    query: str = Field(..., description = "Câu hỏi hoặc tâm sự của người dùng")
# endpoint llm service để đưa câu hỏi và lấy câu trả lời
llm_url = "http://localhost:8000/"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class LLMTool(BaseTool):
    name: str="llm_normal"
    description: str = ("Trợ lý AI dùng để trả lời các câu hỏi KHÔNG liên quan đến tài liệu nội bộ. "
                        "Trọng tâm là tư vấn, lắng nghe, khuyến khích và hỗ trợ tinh thần người dùng."
    )
    args_schema: Type[BaseModel]= LLMInput
    return_direct: bool = False
    
    def _run(self, query: str, run_manager: Optional = None) -> str:
        #xử lý prompt và câu hỏi
        combined_message = f"Người dùng chia sẻ: {query}"
        
        #hàm  đưa câu hỏi cho endpoin và lấy câu trả lời
        logger.info("Đang gọi tới llm")
        try:
             llm_response = requests.post(llm_url + "/response", json={
                "message": combined_message
            })
             self._last_response= llm_response.json()
             return self._last_response.get("content", "Không có phản hồi")
        except Exception as e:
            return f"Lỗi gọi LLM Tool: {str(e)}"