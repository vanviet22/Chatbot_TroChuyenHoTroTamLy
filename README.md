🤖 LangChain-Powered Tâm Lý Support Chatbot
TÌM HIỂU FRAMEWORK LANGCHAIN VÀ ỨNG DỤNG XÂY DỰNG CHATBOT HỖ TRỢ VỀ TÂM LÍ
Đây là dự án nghiên cứu và phát triển chatbot đồng hành, hỗ trợ tâm lý được xây dựng trên nền tảng LangChain (framework tiêu chuẩn cho ứng dụng LLM), tích hợp kiến trúc Retrieval-Augmented Generation (RAG) và hệ thống Agent thông minh.

🌟 Tính Năng Nổi Bật

Trò chuyện Đồng hành (Conversational Memory): Hỗ trợ trò chuyện dài, ghi nhớ ngữ cảnh bằng cách sử dụng MySQL để quản lý lịch sử hội thoại.


Hệ thống Agent Thông minh: Sử dụng Agent làm bộ não điều phối, tự động lựa chọn công cụ phù hợp để trả lời câu hỏi của người dùng.


Truy vấn Dữ liệu Chuyên môn (RAG):

Sử dụng kiến trúc RAG để truy xuất thông tin từ tài liệu chuyên môn (PDF, DOCX) liên quan đến tâm lý, đảm bảo câu trả lời chính xác, đáng tin cậy.

Tích hợp FAISS (Facebook AI Similarity Search) làm Vector Database để lưu trữ và truy vấn ngữ nghĩa hiệu quả.

Web Search Tool: Tích hợp công cụ tìm kiếm web (search_web_tool) để gợi ý những bộ phim, câu chuyện động lực và thông tin truyền cảm hứng cho người dùng.

Cơ chế Cache: Giảm chi phí và tăng tốc độ phản hồi cho các câu hỏi lặp lại bằng cách sử dụng cơ chế cache dựa trên MySQL và tìm kiếm vector.

Cảnh báo Khủng hoảng: Phát hiện các dấu hiệu nghiêm trọng (ý nghĩ tự hại) và đưa ra cảnh báo, khuyến nghị liên hệ dịch vụ hỗ trợ khẩn cấp.

🛠️ Công Nghệ Sử Dụng

Framework LLM / Orchestration: LangChain (Agents, Tools, Chains, RAG).

Backend Server: Python, FastAPI (đóng vai trò là API server, cầu nối giao tiếp với frontend).

Large Language Model (LLM): các mô hình của GROQ.

Vector Database: FAISS (dùng cho RAG).

Cơ sở dữ liệu: MySQL (dùng để lưu trữ Cache và Memory lịch sử hội thoại).

Frontend: HTML, CSS, JS.

🚀 Cấu Trúc Dự Án
Cấu trúc dự án được phân tách rõ ràng thành hai phần chính: backend (xử lý logic) và frontend (giao diện người dùng).

<img width="401" height="526" alt="image" src="https://github.com/user-attachments/assets/cf5765e6-1357-4720-a773-e6cb60803525" />

⚙️ Hướng Dẫn Cài Đặt và Chạy
Thực hiện theo các bước sau để thiết lập và chạy dự án trên máy cục bộ:

1. Clone Repository
Bash

git clone https://github.com/vanviet22/Chatbot_TroChuyenHoTroTamLy.git
cd Chatbot_TroChuyenHoTroTamLy


2. Thiết lập Môi trường Python
Tạo và kích hoạt môi trường ảo (nên dùng Python 3.9 trở lên ):

Bash

python -m venv venv
# Trên Windows
.\venv\Scripts\activate
# Trên macOS/Linux
source venv/bin/activate
3. Cài đặt Thư viện
Cài đặt tất cả các thư viện cần thiết từ file requirements.txt:

Bash

pip install -r requirements.txt


4. Thiết lập Biến Môi trường
   
Tạo một file .env ở thư mục gốc (cùng cấp với main.py) và điền các khóa API/cấu hình cần thiết, dựa trên mẫu trong .env_example:

GROQ_API_KEY: Khóa API để sử dụng các mô hình của GROQ.

TAVILY_API_KEY: Khóa API cho công cụ tìm kiếm web.

Cấu hình MySQL (Tên DB, User, Password, Host).

5. Khởi chạy Backend
Chạy file main.py để khởi động FastAPI server, tạo các API endpoint cần thiết cho chatbot:

Bash

python main.py


6. Truy cập Giao diện Chatbot
Mở thư mục frontend và click vào file HTML (ví dụ: index.html) bằng trình duyệt web để bắt đầu trò chuyện với chatbot.
