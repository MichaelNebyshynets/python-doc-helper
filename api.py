"""HTTP-API над RAG из rag.py. Запуск:  uvicorn api:app --port 8000

GET  /api/config  -> настройки для интерфейса (модель, режимы, число чанков)
POST /api/ask     -> ответ потоком Server-Sent Events (SSE)

Логику поиска и генерации здесь не дублируем: всё делает класс RAG.
"""
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag as rag_module
from rag import RAG, check_citations, refusal_message

DIST = Path(__file__).resolve().parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # RAG грузится один раз при старте (корпус и подключение к Chroma). В тестах его можно подставить заранее.
    if getattr(app.state, "rag", None) is None:
        app.state.rag = RAG()
    yield


app = FastAPI(title="PyTorch docs RAG", lifespan=lifespan)

# В режиме разработки фронтенд живёт на :5173, поэтому браузеру нужно разрешение (CORS).
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=10)
    mode: str = Field("dense", pattern="^(dense|hybrid)$")
    graph: bool = False
    model: str | None = None


def sse(event: str, data: dict) -> str:
    """Одно SSE-сообщение: строка event, строка data (JSON), пустая строка в конце."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/config")
def config():
    r: RAG = app.state.rag
    return {"model": r.gen_model, "modes": ["dense", "hybrid"], "graph": rag_module.GRAPH_PATH.exists(),
            "chunks": len(r.chunks)}


@app.post("/api/ask")
def ask(req: Ask):
    r: RAG = app.state.rag

    def events():  # обычный (синхронный) генератор: FastAPI гонит его в пуле потоков, сервер не блокируется
        t0 = time.time()
        try:
            reason, res = r.retrieve(req.question, k=req.k, mode=req.mode, graph=req.graph)
            if reason:
                yield sse("refusal", {"message": refusal_message(req.question, reason)})
                yield sse("done", {"cited": [], "invented": [], "seconds": round(time.time() - t0, 1)})
                return
            sources = [{"n": i, "title": c["title"], "file": c["file"], "related": s == 0.0}
                       for i, (s, c) in enumerate(res, 1)]
            yield sse("sources", {"sources": sources})
            text = ""
            for piece in r.generate(req.question, res, model=req.model or None):
                text += piece
                yield sse("token", {"t": piece})
            cited, invented = check_citations(text, len(sources))
            yield sse("done", {"cited": cited, "invented": invented, "seconds": round(time.time() - t0, 1)})
        except Exception as e:  # чаще всего: Ollama не запущена или модель не скачана
            yield sse("error", {"message": f"{e}. Проверь, что Ollama запущена и модели скачаны."})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Если фронтенд собран (npm run build), отдаём его же с этого сервера: всё работает на одном порту.
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
