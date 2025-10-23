import os
os.environ["HF_HOME"] = r"D:\HuggingFace_Cache"
os.environ["HF_HUB_CACHE"] = r"D:\HuggingFace_Cache\hub"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from datetime import datetime
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from agent.agent import Agent
from caching.Cache_mysql import MySQLCache
import torch 

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

# ----------------- MODEL LOAD  -----------------
model_id = "Qwen/Qwen2-1.5B-Instruct"

print(f"🚀 Đang tải mô hình {model_id}...")

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",       
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.eval()
# Tạo pipeline HuggingFace
pipe_agent = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,      
    temperature=0.2,
    top_p=0.9
)
# Bọc pipeline thành LLM cho LangChain
hf_pipeline_agent = HuggingFacePipeline(pipeline=pipe_agent)
llm_agent = ChatHuggingFace(llm=hf_pipeline_agent)

# Khởi tạo agent
agent = Agent(llm=llm_agent)
agent_executor = agent.create_agent()

mysqlcache = MySQLCache()

# ----------------- MODELS -----------------
class MessageIn(BaseModel):
    message: str
    history: list = []

class MessageOut(BaseModel):
    content: str
    sender: str = "bot"
    timestamp: str

# ----------------- HELPERS -----------------
# Hợp lịch sử và chat
def merge_his_mess(user_mess, his):
    messages = []
    if his:
        context = " ".join([f"[Context trước đó] {m['content'].strip()}" for m in his])
        messages.append({"role": "system", "content": context})
        
    messages.append({"role": "user", "content": user_mess.strip()})
    return messages

# Sản sinh câu trả lời
def generate_response(user_input: str) -> str:
    system_prompt = """Bạn là một người bạn đồng hành, luôn lắng nghe và thấu hiểu người dùng.
    Trả lời ngắn gọn, dưới 5 câu, tự nhiên và chân thành.
    Luôn đồng cảm, quan tâm và tôn trọng cảm xúc của người dùng.
    Đưa ra góc nhìn tích cực hoặc lời khuyên nhẹ nhàng để họ bình tĩnh hơn.
    Nếu người dùng nói ngoài chủ đề tâm lý/cảm xúc, hãy nhẹ nhàng gợi ý quay lại chủ đề chính.
    Không lặp lại những gì người dùng đã nói."""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens= 128,
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    response = text.strip()
    if response.lower().startswith("assistant:"):
        response = response.split("assistant:", 1)[-1].strip()
    return response

# print("câu trả lời:",generate_response("Hôm nay tôi hơi buồn bạn có thể trò chuyện với tôi được kh"))
# ----------------- ROUTES -----------------
@app.post("/response", response_model=MessageOut)
async def get_response(message: str):
    try:
        response = generate_response(message)
        return MessageOut(
            content=response,
            sender="bot",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.exception("Lỗi tại endpoint /response")
        return MessageOut(
            content="Xin lỗi, đã xảy ra lỗi khi gọi LLM trả lời. Vui lòng thử lại.",
            sender="bot",
            timestamp=datetime.now().isoformat()
        )

@app.post("/chatbot", response_model=MessageOut)
async def chat_reply(msg: MessageIn):
    try:
        if resp := mysqlcache.search_with_vectorstore(msg.message):
            logger.info("Lấy câu trả lời từ mysql cache")
            response_text = resp["answer"]
        else:
            messages = merge_his_mess(msg.message, msg.history)
            result = agent_executor.invoke({"input": messages})
            if isinstance(result, dict):
                response_text = result.get("output") or result.get("messages", [])[-1].content
            else:
                response_text = str(result)
        return MessageOut(
            content=response_text,
            sender="bot",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.exception("Lỗi tại endpoint /chatbot")
        return MessageOut(
            content="Xin lỗi, đã xảy ra lỗi endpoint chatbot. Vui lòng thử lại.",
            sender="bot",
            timestamp=datetime.now().isoformat()
        )

# ----------------- RUN -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=False)