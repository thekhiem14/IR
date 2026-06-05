"""
Đăng ký Student Server với Teacher Server.

Lệnh:
    python register.py register     # đăng ký URL máy mình  (nhanh)
    python register.py evaluate     # BẮT ĐẦU THI -- block ~10 phút
    python register.py result       # check điểm hiện tại    (nhanh)
    python register.py reset        # reset nếu crash        (nhanh)
"""
import os
import sys
import requests

TEACHER_BASE = "http://192.168.50.218:8000/api/v1"
STUDENT_ID   = os.getenv("STUDENT_ID", "B22DCDT171")          # <-- MSSV
SERVER_URL   = os.getenv("SERVER_URL", "http://192.168.50.229:5000")  # <-- IP LAN máy mình

HEADERS = {"X-Student-ID": STUDENT_ID}

# /evaluate sẽ block đến khi Teacher gọi xong 10 câu hỏi ngược về Student:
#   /upload (max 120s) + 10 * /ask (max 60s) ~= tối đa 12 phút
# -> timeout phía requests phải >= 15 phút
EVAL_TIMEOUT = 900   # 15 phút
SHORT_TIMEOUT = 15

def register():
    r = requests.post(f"{TEACHER_BASE}/competition/register",
                      headers=HEADERS,
                      json={"server_url": SERVER_URL},
                      timeout=SHORT_TIMEOUT)
    print("STATUS:", r.status_code)
    print("BODY  :", r.text)

def evaluate():
    print(f"[*] Bắt đầu thi... (block tối đa {EVAL_TIMEOUT}s, đừng tắt cửa sổ này)")
    print(f"[*] Theo dõi log /upload và /ask ở Terminal main.py")
    r = requests.post(f"{TEACHER_BASE}/competition/evaluate",
                      headers=HEADERS,
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
    cmd = sys.argv[1] if len(sys.argv) > 1 else "register"
    {"register": register,
     "evaluate": evaluate,
     "result":   result,
     "reset":    reset}[cmd]()
