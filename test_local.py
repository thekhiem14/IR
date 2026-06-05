"""
Test nhanh Student Server đang chạy local (port 5000).
Mô phỏng Teacher Server gọi /upload + /ask.
"""
import requests

BASE = "http://127.0.0.1:5000"

DOC = """
RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp truy xuất thông tin
với mô hình ngôn ngữ lớn (LLM). Thay vì để LLM tự sinh câu trả lời chỉ dựa
trên kiến thức huấn luyện, RAG truy xuất các đoạn văn bản liên quan từ một
kho tri thức (vector database) rồi đưa vào prompt làm ngữ cảnh.

Một hệ thống RAG cơ bản gồm 3 bước: (1) chunking - chia tài liệu thành các
đoạn nhỏ; (2) embedding - mã hóa các đoạn thành vector; (3) retrieval -
khi có câu hỏi, tìm các đoạn gần nhất bằng cosine similarity rồi gửi cho LLM.

FAISS là thư viện do Facebook AI Research phát triển, dùng để đánh chỉ mục
và tìm kiếm vector hiệu quả ở quy mô lớn.
"""

def test_upload():
    r = requests.post(f"{BASE}/upload",
                      json={"doc_id": "test_doc", "text": DOC},
                      timeout=120)
    print("UPLOAD:", r.status_code, r.json())

def test_ask():
    q = ("RAG là gì? "
         "A. Một loại mô hình ngôn ngữ lớn "
         "B. Kỹ thuật kết hợp truy xuất thông tin với LLM "
         "C. Một thư viện vector "
         "D. Phương pháp huấn luyện LLM")
    r = requests.post(f"{BASE}/ask", json={"question": q}, timeout=60)
    print("ASK:", r.status_code, r.json())

if __name__ == "__main__":
    test_upload()
    test_ask()
