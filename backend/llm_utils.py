import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def generate_response(user_input: str) -> str:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = (
        "Bạn là một người bạn đồng hành, luôn lắng nghe và thấu hiểu người dùng.\n"
        "Trả lời ngắn gọn, dưới 5 câu, tự nhiên và chân thành.\n"
        "Luôn đồng cảm, quan tâm và tôn trọng cảm xúc của người dùng.\n"
        "Đưa ra góc nhìn tích cực.\n"
        "Nếu người dùng nói ngoài chủ đề tâm lý/cảm xúc, hãy nhẹ nhàng gợi ý quay lại chủ đề chính.\n"
        "Không lặp lại những gì người dùng đã nói."
    )

    messages = [
        ("system", system_prompt),
        ("user", user_input)
    ]

    response = llm.invoke(messages)
    return response.content.strip()
