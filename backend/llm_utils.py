import os
os.environ["HF_HOME"] = r"D:\HuggingFace_Cache"
os.environ["HF_HUB_CACHE"] = r"D:\HuggingFace_Cache\hub"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "Qwen/Qwen2-1.5B-Instruct"
# print(f"🚀 Đang tải mô hình {model_id} bên llm_utils...")

# # Load tokenizer & model
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cpu",
#     offload_folder="D:/HuggingFace_Cache/offload",  # thư mục lưu tạm
#     torch_dtype= torch.float32,
#     trust_remote_code=True,
#     low_cpu_mem_usage=True
# )
# model.eval()

def generate_response(tokenizer, model, user_input: str) -> str:
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
            max_new_tokens=128,
            temperature=0.,
            top_p=0.8,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    response = text.strip()
    if response.lower().startswith("assistant:"):
        response = response.split("assistant:", 1)[-1].strip()
    return response
