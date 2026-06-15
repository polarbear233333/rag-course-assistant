# chat_store.py
import json
import os
from threading import Lock
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, "chat_log.json")
_lock = Lock()

if not os.path.exists(FILE):
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as file:
        json.dump([], file)


def load_logs() -> List[Dict[str, Any]]:
    if not os.path.exists(FILE):
        return []
    with open(FILE, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []


def save_logs(data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def append_log(record: Dict[str, Any]) -> None:
    with _lock:
        data = load_logs()
        data.append(record)
        save_logs(data)


def update_answer(record_id: str, answer: str) -> None:
    with _lock:
        data = load_logs()
        for record in reversed(data):
            if record.get("id") == record_id:
                record["answer"] = answer
                break
        save_logs(data)


print("chat_log.json will be saved to:", FILE)
