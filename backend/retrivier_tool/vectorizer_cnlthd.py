import os
from pathlib import Path
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from dotenv import load_dotenv
import logging
import argparse

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
"""
file này thực thi quá trình vector embedding dữ liệu nội bộ cho hệ thống RAG
gồm 2 class: chunkingDocs: các kĩ thuật chunking
            Vectorize: thực hiện embedding
"""
class ChunkingDocs:
    def __init__(self, embeddings_model=None):
        self._embeddings_model = embeddings_model or HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def chunk_recursive(self, documents: list[Document], chunk_size: int = 200, overlap: int = 50):
        # kĩ thuật recursive
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents(documents)
    

class Vectorizer:
    def __init__(self, folder_path: str = None,
                 vector_path: str = None,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # default paths relative to package
        base = Path(__file__).resolve().parents[1]
        self._folder_path = Path(folder_path) if folder_path else (base / 'data' / 'raw')
        self._vector_path = Path(vector_path) if vector_path else (base / 'data' / 'vector_store')
        self._documents: list[Document] = []
        
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=model_name
        )

        self._chunking_tool = ChunkingDocs(embeddings_model=self._embedding_model)

        if not self._folder_path.exists():
            raise FileNotFoundError(f"Folder path không tồn tại: {self._folder_path}")

        # ensure output directory exists
        self._vector_path.parent.mkdir(parents=True, exist_ok=True)

    def _encode_pdf(self):
        # Hàm load dữ liệu cho file pdf
        pdf_files = [p for p in self._folder_path.iterdir() if p.is_file() and p.suffix.lower() == '.pdf']
        if not pdf_files:
            logger.warning(f"Không tìm thấy file .pdf nào trong {self._folder_path}")
            return
        logger.info(f"Tìm thấy {len(pdf_files)} file .pdf")
        for full_path in pdf_files:
            try:
                with fitz.open(str(full_path)) as pdf_doc:
                    # iterate pages and add page-level documents with metadata
                    for i, page in enumerate(pdf_doc, start=1):
                        try:
                            text = page.get_text("text").strip()
                        except Exception:
                            text = ''
                        if not text:
                            logger.debug(f"Trang {i} của {full_path.name} rỗng, bỏ qua")
                            continue
                        metadata = {'source': str(full_path.resolve()), 'filename': full_path.name, 'page': i}
                        self._documents.append(Document(page_content=text, metadata=metadata))
                    logger.info(f"Đã load: {full_path.name} ({len(pdf_doc)} trang; tổng đoạn: {len([d for d in self._documents if d.metadata.get('filename')==full_path.name])})")
            except Exception as e:
                logger.exception(f"Lỗi khi đọc file PDF {full_path}: {e}")

    def _split_and_save(self, strategy: str = "recursive"):
        # Hàm sử dụng các kĩ thuật chunking để chia dữ liệu sau đó thức hiện embedding rồi lưu vào FAISS
        if not self._documents:
            logger.error("Không có document nào để split. Hãy chạy _encode() trước.")
            return False
        
        try:
            logger.info(f"Áp dụng chiến lược chunking: {strategy}")
            if strategy == "recursive":
                all_splits = self._chunking_tool.chunk_recursive(self._documents)
            else:
                logger.error(f"Chiến lược không hợp lệ: {strategy}. Chỉ chấp nhận 'recursive'")
                return False

            if not all_splits:
                logger.warning("Không có đoạn văn bản nào sau khi split.")
                return False
            
            vector_db = FAISS.from_documents(all_splits, self._embedding_model)
            vector_db.save_local(str(self._vector_path))
            logger.info(f"Đã lưu vector store tại: {self._vector_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Lỗi khi tạo vector store: {e}")
            return False
    
    def _run(self, strategy: str = "recursive"):
        # Hàm chạy chính
        self._encode_pdf()

        if not self._documents:
            logger.error("Không có document nào được load. Dừng quá trình.")
            return False

        success = self._split_and_save(strategy)

        if success:
            logger.info("=== HOÀN THÀNH VECTORIZATION ===")
        else:
            logger.error("=== VECTORIZATION THẤT BẠI ===")

        return success

def run_vectorizer(folder_path: str = None, vector_path: str = None, model_name: str = None, strategy: str = "recursive") -> bool:
    """Create and run vectorizer.

    Parameters can be passed to override defaults or env variables.
    Returns True on success, False otherwise.
    """
    vectorizer = Vectorizer()
    return vectorizer._run(strategy=strategy)


