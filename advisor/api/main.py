"""FastAPI backend с веб-чатом."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.recommender import Recommender


app = FastAPI(title="AI Model Advisor Chat")

static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

recommender = Recommender()


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []


def format_recommendation(result) -> str:
    """Форматирует рекомендации в красивый HTML (пурпурно-тёмная тема)."""
    
    recs = result["recommendations"]
    if not recs:
        return "Не удалось подобрать модель. Попробуй переформулировать."
    
    html = []
    
    for r in recs:
        m = r.model
        
        # Цвет границы по уверенности
        if r.confidence > 0.5:
            border_color = "#22c55e"  # зелёный
        elif r.confidence > 0.3:
            border_color = "#eab308"  # жёлтый
        else:
            border_color = "#ef4444"  # красный
        
        card = f"""
        <div class="model-card" style="border-left-color: {border_color};">
            <h3>{r.rank}. {m.name}</h3>
            <div class="company">🏢 {m.company} | {m.access_label}</div>
            <div class="price">💰 {m.price}</div>
            <div class="plus">✅ {m.plus}</div>
            <div class="explanation">💡 {r.explanation}</div>
            <div class="tags">
                {' '.join(f'<span class="tag">#{tag}</span>' for tag in m.tags[:3])}
            </div>
        </div>
        """
        html.append(card)
    
    # Бесплатные альтернативы
    free = recommender.recommend(result["query"], top_k=2, access_filter="free")
    if free["recommendations"]:
        html.append('<div class="free-alternatives"><h4>🆓 Бесплатные альтернативы</h4>')
        for r in free["recommendations"]:
            html.append(f'<div class="free-item"><b>{r.model.name}</b> — {r.model.price}</div>')
        html.append('</div>')
    
    return "".join(html)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static_path, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat(req: ChatRequest):
    text = req.message.lower()
    
    if any(w in text for w in ["привет", "hello"]):
        return {"response": "Привет! 👋 Опиши задачу, я подберу AI-модель.", "history": []}
    
    if any(w in text for w in ["спасибо", "thanks"]):
        return {"response": "Пожалуйста! 😊", "history": []}
    
    result = recommender.recommend(req.message, top_k=3)
    html_response = format_recommendation(result)
    
    return {"response": html_response, "history": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)