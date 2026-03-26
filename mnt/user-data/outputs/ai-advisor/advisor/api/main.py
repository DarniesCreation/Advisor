
"""FastAPI backend для AI Advisor."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import time

from ..core.recommender import Recommender
from ..data.models import MODELS, get_model, NUM_MODELS


app = FastAPI(
    title="AI Model Advisor API",
    description="Умный подбор AI-моделей по описанию задачи",
    version="3.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный recommender (инициализируется при старте)
recommender: Optional[Recommender] = None


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000, description="Описание задачи")
    top_k: int = Field(3, ge=1, le=10, description="Количество рекомендаций")
    access_filter: Optional[Literal["free", "paid"]] = Field(None, description="Фильтр по цене")
    user_id: Optional[str] = Field(None, description="ID пользователя для персонализации")


class ModelInfo(BaseModel):
    id: int
    name: str
    company: str
    description: str
    price: str
    url: str
    access: str
    tier: str
    tags: List[str]


class RecommendationResponse(BaseModel):
    rank: int
    model: ModelInfo
    confidence: float
    explanation: str


class QueryResponse(BaseModel):
    query: str
    processing_time_ms: float
    detected_tasks: List[int]
    detected_task_names: List[str]
    parameters: dict
    recommendations: List[RecommendationResponse]
    alternatives_free: List[RecommendationResponse]


class CompareRequest(BaseModel):
    model_ids: List[int] = Field(..., min_items=2, max_items=5)


@app.on_event("startup")
async def startup():
    """Инициализация при старте."""
    global recommender
    try:
        from ..core.network import AdvisorNet
        network = AdvisorNet(num_tasks=23, num_models=NUM_MODELS)
        network.load("advisor_weights.pt")
    except:
        from ..core.network import DummyNetwork
        network = DummyNetwork(NUM_MODELS)
    
    recommender = Recommender(network=network)
    print("✅ Recommender initialized")


@app.get("/")
async def root():
    return {
        "service": "AI Model Advisor API",
        "version": "3.1.0",
        "models_count": NUM_MODELS,
        "endpoints": ["/recommend", "/models", "/compare", "/alternatives/{model_id}"]
    }


@app.post("/recommend", response_model=QueryResponse)
async def recommend(req: QueryRequest):
    """Получить рекомендации по описанию задачи."""
    if recommender is None:
        raise HTTPException(503, "Service not ready")
    
    start = time.time()
    
    # Основные рекомендации
    result = recommender.recommend(
        req.text,
        top_k=req.top_k,
        access_filter=req.access_filter
    )
    
    # Бесплатные альтернативы (всегда показываем)
    free_result = recommender.recommend(req.text, top_k=2, access_filter="free")
    
    elapsed = (time.time() - start) * 1000
    
    # Форматируем ответ
    task_names = {
        0: "Тексты", 1: "Код", 2: "Анализ данных", 3: "Документы",
        4: "Картинки", 5: "Видео", 6: "Музыка", 7: "Озвучка",
        8: "Транскрибация", 9: "Перевод", 10: "Поиск", 11: "Обучение",
        12: "Математика", 13: "Чат", 14: "Дизайн", 15: "Маркетинг",
        16: "Big Data", 17: "Локально", 18: "Бесплатно", 19: "SEO",
        20: "Офис", 21: "Бизнес-видео", 22: "Логотипы",
    }
    
    def format_rec(rec):
        m = rec.model
        return RecommendationResponse(
            rank=rec.rank,
            model=ModelInfo(
                id=m.id,
                name=m.name,
                company=m.company,
                description=m.description,
                price=m.price,
                url=m.url,
                access=m.access.value,
                tier=m.tier.value,
                tags=m.tags
            ),
            confidence=rec.confidence,
            explanation=rec.explanation
        )
    
    return QueryResponse(
        query=req.text,
        processing_time_ms=elapsed,
        detected_tasks=result["parsed"].tasks,
        detected_task_names=[task_names.get(t, str(t)) for t in result["parsed"].tasks],
        parameters={
            "volume": result["parsed"].volume,
            "budget": result["parsed"].budget,
            "gpu": result["parsed"].gpu,
            "priorities": {
                "quality": result["parsed"].priorities[0],
                "speed": result["parsed"].priorities[1],
                "cost": result["parsed"].priorities[2],
                "privacy": result["parsed"].priorities[3],
                "ease": result["parsed"].priorities[4],
            }
        },
        recommendations=[format_rec(r) for r in result["recommendations"]],
        alternatives_free=[format_rec(r) for r in free_result["recommendations"]]
    )


@app.get("/models")
async def list_models(
    tag: Optional[str] = None,
    access: Optional[Literal["free", "paid", "freemium"]] = None
):
    """Список всех моделей с фильтрацией."""
    models = []
    for m in MODELS.values():
        if tag and tag not in m.tags:
            continue
        if access and m.access.value != access:
            continue
        models.append(ModelInfo(
            id=m.id,
            name=m.name,
            company=m.company,
            description=m.description,
            price=m.price,
            url=m.url,
            access=m.access.value,
            tier=m.tier.value,
            tags=m.tags
        ))
    return models


@app.get("/models/{model_id}")
async def get_model_info(model_id: int):
    """Информация о конкретной модели."""
    m = get_model(model_id)
    if m is None:
        raise HTTPException(404, "Model not found")
    
    return ModelInfo(
        id=m.id,
        name=m.name,
        company=m.company,
        description=m.description,
        price=m.price,
        url=m.url,
        access=m.access.value,
        tier=m.tier.value,
        tags=m.tags
    )


@app.post("/compare")
async def compare(req: CompareRequest):
    """Сравнить несколько моделей."""
    models = []
    for mid in req.model_ids:
        m = get_model(mid)
        if m is None:
            raise HTTPException(404, f"Model {mid} not found")
        models.append(ModelInfo(
            id=m.id,
            name=m.name,
            company=m.company,
            description=m.description,
            price=m.price,
            url=m.url,
            access=m.access.value,
            tier=m.tier.value,
            tags=m.tags
        ))
    
    # Находим общие теги
    common_tags = set.intersection(*[set(m.tags) for m in models]) if models else set()
    
    return {
        "models": models,
        "common_tags": list(common_tags),
        "differences": {
            "companies": list(set(m.company for m in models)),
            "access_types": list(set(m.access for m in models)),
            "tiers": list(set(m.tier for m in models)),
        }
    }


@app.get("/alternatives/{model_id}")
async def get_alternatives(model_id: int, top_k: int = 3):
    """Найти альтернативы для модели."""
    if recommender is None:
        raise HTTPException(503, "Service not ready")
    
    if get_model(model_id) is None:
        raise HTTPException(404, "Model not found")
    
    alts = recommender.get_alternatives(model_id, top_k=top_k)
    
    return {
        "original_model_id": model_id,
        "alternatives": [
            {
                "rank": rec.rank,
                "model_id": rec.model.id,
                "name": rec.model.name,
                "similarity": rec.confidence,
                "explanation": rec.explanation
            }
            for rec in alts
        ]
    }


# Запуск: uvicorn advisor.api.main:app --reload

