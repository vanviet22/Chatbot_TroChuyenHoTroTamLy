import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from datetime import datetime
import logging
from agent.agent import Agent
from caching.Cache_mysql import MySQLCache
from dotenv import load_dotenv
load_dotenv()

# ----------------- SETUP -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo agent một lần duy nhất
agent = Agent()
agent_executor = agent.create_agent()


mysqlcache = MySQLCache()

# ----------------- MODELS -----------------
class MessageIn(BaseModel):
    message: str

class MessageOut(BaseModel):
    content: str
    sender: str = "bot"
    timestamp: str


# ----------------- ENDPOINTS -----------------
@app.post("/chatbot", response_model=MessageOut)
async def chat_reply(msg: MessageIn):
    try:
        # Kiểm tra cache
        cached = mysqlcache.search_with_vectorstore(msg.message)
        if cached:
            logger.info("Lấy câu trả lời từ mysql cache")
            response_text = cached["answer"]

            # return NGAY tại đây → response_text đã được gán
            return MessageOut(
                content=response_text,
                sender="bot",
                timestamp=datetime.now().isoformat()
            )

        # Không có cache → gọi Agent
        result = agent_executor.invoke({"input": msg.message})

        if isinstance(result, dict):
            response_text = (
                result.get("output")
                or (result.get("messages")[-1].content if result.get("messages") else "")
            )
        else:
            response_text = str(result)

        # Lưu cache
        mysqlcache.add_cache(question=msg.message, answer=response_text)

        return MessageOut(
            content=response_text,
            sender="bot",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.exception("Lỗi tại endpoint /chatbot")

        # return riêng khi lỗi → KHÔNG DÙNG response_text
        return MessageOut(
            content="Xin lỗi, hiện hệ thống gặp sự cố. Bạn thử lại giúp mình nhé.",
            sender="bot",
            timestamp=datetime.now().isoformat()
        )

# ----------------- RUN -----------------
" Câu lệnh chạy trên terminal"
# uvicorn main:app --reload
" hoặc  chạy file python"
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)