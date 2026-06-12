"""
Tải model `keepitreal/vietnamese-sbert` về thư mục ./models/vietnamese-sbert
CHẠY 1 LẦN KHI CÒN INTERNET (ở nhà / wifi ngoài), TRƯỚC KHI VÀO PHÒNG THI.

Sau khi chạy xong, kiểm tra:
    ls ./models/vietnamese-sbert
phải thấy: config.json, pytorch_model.bin (hoặc model.safetensors),
            tokenizer files, modules.json, sentence_bert_config.json, ...
"""
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "keepitreal/vietnamese-sbert"
SAVE_DIR   = "./models/vietnamese-sbert"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[*] Downloading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)   # tải từ HF Hub
    print(f"[*] Saving to {SAVE_DIR} ...")
    model.save(SAVE_DIR)

    # Smoke test: load lại từ local + encode 1 câu
    print("[*] Reload from local to verify ...")
    m2 = SentenceTransformer(SAVE_DIR)
    vec = m2.encode(["Xin chào, đây là kiểm tra embedding tiếng Việt."])
    print(f"[OK] Embedding dim = {vec.shape[1]}")
    print(f"[OK] Model sẵn sàng tại: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()
