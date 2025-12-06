
# 🤖 LangChain-Powered Tâm Lý Support Chatbot

  - Tìm hiểu LangChain Framework & Ứng dụng xây dựng Chatbot hỗ trợ tâm lý

  - Đây là dự án nghiên cứu và phát triển chatbot đồng hành – hỗ trợ tâm lý được xây dựng trên nền tảng LangChain, ứng dụng kiến trúc RAG và Intelligent Agent nhằm cung cấp trải nghiệm trò chuyện tự nhiên, giàu ý nghĩa và đáng tin cậy.

### 🌟 Tính Năng Nổi Bật
  - 🧠 1. Trò chuyện đồng hành (Conversational Memory)

    + Ghi nhớ ngữ cảnh trò chuyện dài hạn.

    + Sử dụng MySQL để lưu trữ và quản lý lịch sử hội thoại.

    + Giúp chatbot phản hồi tự nhiên, cá nhân hóa.

  - 🤖 2. Hệ thống Agent thông minh

    + Agent đóng vai trò “bộ não điều phối”.

    + Tự động chọn tool phù hợp (RAG, web search, cache…).

    + Linh hoạt xử lý đa dạng yêu cầu người dùng.

  - 📚 3. Truy vấn tài liệu chuyên môn (RAG)

    + Tích hợp kiến trúc Retrieval-Augmented Generation.

    + Truy xuất thông tin từ PDF/DOCX liên quan đến tâm lý học.

    + Kết hợp FAISS vector database để truy vấn ngữ nghĩa nhanh & chính xác.

  - 🌐 4. Web Search Tool

    + Tìm kiếm phim truyền cảm hứng, câu chuyện chữa lành, lời khuyên hữu ích.

    + Sử dụng các API web-search được tích hợp trong LangChain.

  - ⚡ 5. Cơ chế Cache nâng cao

    + Cache kết quả LLM và kết quả truy vấn vector.

    + Lưu vào MySQL → tiết kiệm chi phí, tăng tốc độ phản hồi.

### 🛠️ Công Nghệ Sử Dụng
  - Thành phần	Công nghệ
    + Framework LLM	LangChain (Agents, Tools, RAG, Chains)
    + Backend	FastAPI (Python)
    + LLM Provider	GROQ LLM Models
    + Vector DB	FAISS
    + Database	MySQL (cache + conversational memory)
    + Frontend	HTML, CSS, JavaScript
### 📁 Cấu Trúc Dự Án

<img width="286" height="529" alt="image" src="https://github.com/user-attachments/assets/84a2d562-0c6e-4d25-a3ce-c7733c94b371" />

### ⚙️ Hướng Dẫn Cài Đặt & Chạy

  - 1. Clone Repository: git clone https://github.com/vanviet22/Chatbot_TroChuyenHoTroTamLy.git

    + cd Chatbot_TroChuyenHoTroTamLy

  - 2. Tạo môi trường ảo Python

    + Yêu cầu: Python 3.9+
  
    + python -m venv venv

Windows

.\venv\Scripts\activate


macOS / Linux

source venv/bin/activate

  - 3. Cài đặt thư viện: pip install -r requirements.txt

  - 4. Thiết lập biến môi trường
  
    + Tạo file .env (cùng cấp với main.py) và Tham khảo file: .env_example

  - 5. Khởi chạy Backend

    + python main.py

    + FastAPI server sẽ chạy tại: http://127.0.0.1:8000

  - 6. Mở giao diện Chatbot

    + Vào thư mục frontend/ → mở file index.html trên trình duyệt để bắt đầu tương tác với chatbot.
