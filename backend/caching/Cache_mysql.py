import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["HF_HOME"] = r"D:\HuggingFace_Cache"
os.environ["HF_HUB_CACHE"] = r"D:\HuggingFace_Cache\hub"
import mysql.connector
import hashlib
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.connect_mysql import get_cache_table
import statistics
import time
from tabulate import tabulate
import json

class MySQLCache:
    def __init__(self, embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.conn, self.cursor = get_cache_table()
        self._embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None

    # ------------------- Tiện ích -------------------
    def _generate_cache_id(self, question: str) -> str:
        normalized_question = question.lower().strip()
        return hashlib.md5(normalized_question.encode()).hexdigest()

    def _get_embedding(self, question: str) -> List[float]:
        try:
            return self._embedding_model.embed_query(question)
        except Exception as e:
            print(f"Lỗi khi tạo embedding: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ------------------- CRUD -------------------
    def add_cache(self, question: str, answer: str):
        cache_id = self._generate_cache_id(question)
        processed_question = self._preprocess_text(question)
        embedding = self._get_embedding(processed_question)
        if not embedding:
            raise ValueError("Không tạo được embedding cho câu hỏi")

        now = datetime.utcnow()
        sql = """
        INSERT INTO cache (id, question, answer, embedding, time)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            question=VALUES(question),
            answer=VALUES(answer),
            embedding=VALUES(embedding),
            time=VALUES(time)
        """
        self.cursor.execute(sql, (cache_id, question, answer, json.dumps(embedding), now))
        self.conn.commit()
        return cache_id

    def update_cache(self, question: str, new_answer: str) -> bool:
        cache_id = self._generate_cache_id(question)
        now = datetime.utcnow()
        sql = "UPDATE cache SET answer=%s, time=%s WHERE id=%s"
        self.cursor.execute(sql, (new_answer, now, cache_id))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_cache(self, question: str) -> bool:
        cache_id = self._generate_cache_id(question)
        self.cursor.execute("DELETE FROM cache WHERE id=%s", (cache_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def list_all_cache(self):
        self.cursor.execute("SELECT id, question, answer, time FROM cache")
        rows = self.cursor.fetchall()
        caches = [
            {"id": r[0], "question": r[1], "answer": r[2], "time": r[3].isoformat() if r[3] else None}
            for r in rows
        ]
        return caches

    # ------------------- FAISS VectorStore -------------------
    def build_vectorstore(self):
        texts, embeddings, metadatas = [], [], []

        self.cursor.execute("SELECT id, question, answer, embedding FROM cache")
        rows = self.cursor.fetchall()
        
        for r in rows:
            embedding_str = r["embedding"]
            
            embedding = json.loads(embedding_str)
            embeddings.append(np.array(embedding, dtype="float32"))
            texts.append(r["question"])
            metadatas.append({"answer": r["answer"], "id": r["id"]})

        if not embeddings:
            print("⚠️ Chưa có dữ liệu để build vectorstore")
            return None

        self.vectorstore = FAISS.from_embeddings(
            list(zip(texts, embeddings)), self._embedding_model, metadatas=metadatas
        )
        print(f"✅ Vectorstore built với {len(texts)} documents")
        return self.vectorstore

    def search_with_vectorstore(self, query: str, top_k=3, threshold=0.75, threshold_check=0.1):
        if self.vectorstore is None:
            self.build_vectorstore()
            if self.vectorstore is None:
                print("⚠️ Vectorstore chưa có dữ liệu — bỏ qua tìm kiếm.")
                return None

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        filtered = []

        for r, l2_dist in results:
            cosine_sim = 1 - (l2_dist**2) / 2
            if cosine_sim >= threshold:
                filtered.append({
                    "question": r.page_content,
                    "answer": r.metadata.get("answer"),
                    "id": r.metadata.get("id"),
                    "score": float(cosine_sim)
                })

        if not filtered:
            return None

        filtered.sort(key=lambda x: x["score"], reverse=True)
        if len(filtered) > 1:
            diff = filtered[0]["score"] - filtered[1]["score"]
            if diff < threshold_check:
                return None

        return filtered[0]

