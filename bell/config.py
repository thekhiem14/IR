import os
from pathlib import Path

from dotenv import load_dotenv


# Load shared.env ở root của repo (../shared.env so với folder bell/)
load_dotenv(Path(__file__).resolve().parent.parent / "shared.env")

STUDENT_ID = os.getenv("STUDENT_ID", "B22DCDT171")

STUDENT_HOST = "0.0.0.0"
STUDENT_PORT = int(os.getenv("SERVER_PORT", "5000"))
STUDENT_LAN_IP = os.getenv("SERVER_PUBLIC_IP") or "192.168.50.234"

STUDENT_SERVER_URL = f"http://{STUDENT_LAN_IP}:{STUDENT_PORT}"

TEACHER_BASE_URL = os.getenv(
    "TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1"
).rstrip("/")
TEACHER_PROXY_URL = os.getenv(
    "TEACHER_PROXY_BASE_URL", f"{TEACHER_BASE_URL}/proxy"
).rstrip("/")

HEADERS = {
    "X-Student-ID": STUDENT_ID,
    "Content-Type": "application/json",
}
