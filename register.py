"""
Đăng ký Student Server với Teacher Server.

Lệnh:
    python register.py register                # đăng ký URL máy mình  (nhanh)
    python register.py evaluate                # LẦN ĐẦU: document_received=False
                                               #   teacher gửi /upload rồi 100 câu /ask
    python register.py evaluate --received     # lần sau: document_received=True
                                               #   bỏ qua /upload, chỉ bơm 100 câu
    python register.py result                  # check điểm hiện tại    (nhanh)
    python register.py reset                   # reset nếu crash        (nhanh)
"""
import argparse
import os
import requests

TEACHER_BASE = "http://192.168.50.218:8000/api/v1"
STUDENT_ID   = os.getenv("STUDENT_ID", "B22DCDT171")          # <-- MSSV
SERVER_URL   = os.getenv("SERVER_URL", "http://192.168.50.229:5000")  # <-- IP LAN máy mình

HEADERS = {"X-Student-ID": STUDENT_ID}

# /evaluate sẽ block đến khi Teacher gọi xong 100 câu hỏi ngược về Student:
#   /upload (max 120s) + 100 * /ask (max 60s) ~= tối đa 102 phút
# -> timeout phía requests phải >= 105 phút để có buffer
EVAL_TIMEOUT = 6300   # 105 phút
SHORT_TIMEOUT = 15

def register():
    r = requests.post(f"{TEACHER_BASE}/competition/register",
                      headers=HEADERS,
                      json={"server_url": SERVER_URL},
                      timeout=SHORT_TIMEOUT)
    print("STATUS:", r.status_code)
    print("BODY  :", r.text)

def evaluate(received: bool = False):
    payload = {"document_received": received}
    flag = "SKIP UPLOAD (lần sau)" if received else "FULL (upload + 100 câu)"
    print(f"[*] Bắt đầu thi  [{flag}]")
    print(f"[*] Block tối đa {EVAL_TIMEOUT}s (~{EVAL_TIMEOUT // 60} phút), đừng tắt cửa sổ này")
    print(f"[*] Payload: {payload}")
    print(f"[*] Theo dõi log /upload và /ask ở Terminal main.py")
    r = requests.post(f"{TEACHER_BASE}/competition/evaluate",
                      headers=HEADERS,
                      json=payload,
                      timeout=EVAL_TIMEOUT)
    print("STATUS:", r.status_code)
    print("BODY  :", r.text)

def result():
    r = requests.get(f"{TEACHER_BASE}/competition/result",
                     headers=HEADERS,
                     timeout=SHORT_TIMEOUT)
    print("STATUS:", r.status_code)
    print("BODY  :", r.text)

def reset():
    r = requests.post(f"{TEACHER_BASE}/competition/reset",
                      headers=HEADERS,
                      timeout=SHORT_TIMEOUT)
    print("STATUS:", r.status_code)
    print("BODY  :", r.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khiem Teacher Server CLI")
    parser.add_argument("command", choices=["register", "evaluate", "result", "reset"])
    parser.add_argument(
        "--received",
        action="store_true",
        help="Gửi document_received=True (đã upload + embed xong từ lần thi trước).",
    )
    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate(received=args.received)
    else:
        {"register": register, "result": result, "reset": reset}[args.command]()
