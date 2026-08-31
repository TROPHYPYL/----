from typing import Union
import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
# from dialogue.session import SessionState

import sys
from pathlib import Path
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import config
from schemas import Question, Answer #type: ignore
from service import ChatService

service = ChatService()
fastApi = FastAPI()

# _sessions = {} # 会话管理，记录历史聊天内容的

fastApi.mount("/static", StaticFiles(directory=config.WEB_STATIC_DIR), name="static")
@fastApi.get("/")
def read_root():
    return RedirectResponse("/static/index.html")

@fastApi.post("/api/chat")
def read_item(question: Question):
    # session = _sessions.setdefault(question.session_id, SessionState(session_id=question.session_id))
    result = service.chat(question.message)
    # session.add_turn(question.message, result)
    return Answer(message=result)

if __name__ == "__main__":
    uvicorn.run("app:fastApi", host="0.0.0.0", port=8001, reload=True)