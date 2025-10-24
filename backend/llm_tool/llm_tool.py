from langchain.tools import BaseTool
from pydantic import BaseModel, PrivateAttr
from typing import Type, Optional
import logging
from llm_utils import generate_response

class LLMInput(BaseModel):
    query: str

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LLMTool(BaseTool):
    name: str = "llm_normal"
    description: str = ("Trợ lý AI dùng để trả lời các câu hỏi bình thường KHÔNG liên quan đến tài liệu nội bộ. "
                        "Trọng tâm là tư vấn, lắng nghe, khuyến khích và hỗ trợ tinh thần người dùng.")
    args_schema: Type[BaseModel] = LLMInput
    return_direct: bool = False

    # ✅ private attributes để lưu model/tokenizer
    _model: Optional[any] = PrivateAttr()
    _tokenizer: Optional[any] = PrivateAttr()

    def __init__(self, tokenizer, model=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_tokenizer", tokenizer)

    def _run(self, query: str, run_manager: Optional = None) -> str:
        combined_message = f"Người dùng chia sẻ: {query}"
        logger.info("Đang gọi tới LLM")
        try:
            response_text = generate_response(self._tokenizer,self._model, combined_message)
            return response_text
        except Exception as e:
            return f"Lỗi gọi LLM Tool: {str(e)}"

# llm = LLMTool()
# print("câu trả lời:", llm._run("bạn là ai"))