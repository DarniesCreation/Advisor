
"""Основной движок рекомендаций."""


import numpy as np
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass

# Абсолютные импорты для Render
from advisor.data.models import MODELS, AIModel, get_model, NUM_MODELS
from advisor.nlp.parser import ParsedQuery, QueryParser, TaskDefinitions


@dataclass
class Recommendation:
    """Результат рекомендации."""
    rank: int
    model: AIModel
    confidence: float
    explanation: str
    score_breakdown: Dict[str, float]


class ModelScorer:
    """Rule-based скоринг моделей."""
    
    # Матрица: task_id -> {model_id: base_score}
    TASK_SCORES = {
        0: {0: 0.95, 1: 0.93, 20: 0.90, 3: 0.88, 13: 0.80, 18: 0.75, 
            30: 0.72, 31: 0.68, 21: 0.70, 14: 0.60},
        1: {1: 0.96, 20: 0.94, 8: 0.93, 22: 0.91, 0: 0.90, 7: 0.88, 
            33: 0.92, 3: 0.85, 18: 0.82, 13: 0.80, 31: 0.70, 14: 0.65},
        2: {1: 0.90, 20: 0.88, 0: 0.88, 3: 0.90, 2: 0.92, 13: 0.75, 21: 0.72},
        3: {1: 0.95, 20: 0.93, 2: 0.93, 3: 0.88, 16: 0.90, 0: 0.85, 14: 0.70},
        4: {4: 0.95, 23: 0.92, 5: 0.85, 6: 0.85, 24: 0.88, 25: 0.80, 
            19: 0.78, 34: 0.85, 29: 0.75, 3: 0.40},
        5: {11: 0.93, 17: 0.92, 26: 0.94, 27: 0.90, 28: 0.80, 3: 0.30},
        6: {10: 0.95},
        7: {9: 0.95, 16: 0.70},
        8: {15: 0.95, 3: 0.50, 0: 0.40},
        9: {0: 0.90, 1: 0.88, 20: 0.87, 3: 0.88, 13: 0.75, 30: 0.78, 
            31: 0.80, 21: 0.72, 14: 0.60},
        10: {12: 0.95, 3: 0.75, 18: 0.70, 0: 0.65},
        11: {0: 0.90, 1: 0.88, 20: 0.87, 3: 0.88, 12: 0.80, 13: 0.80, 
             16: 0.75, 21: 0.72},
        12: {13: 0.90, 0: 0.88, 1: 0.90, 3: 0.88, 18: 0.82, 2: 0.78},
        13: {0: 0.92, 1: 0.88, 20: 0.87, 18: 0.82, 3: 0.85, 13: 0.80, 
             21: 0.75, 30: 0.72, 14: 0.60},
        14: {4: 0.90, 29: 0.92, 24: 0.88, 5: 0.80, 25: 0.85, 23: 0.82, 
             19: 0.78, 34: 0.80, 6: 0.70, 3: 0.50},
        15: {0: 0.88, 1: 0.82, 20: 0.80, 3: 0.82, 12: 0.78, 18: 0.70, 24: 0.75},
        16: {2: 0.95, 14: 0.88, 1: 0.88, 3: 0.85},
        17: {14: 0.95, 6: 0.92, 15: 0.90, 13: 0.88, 30: 0.85, 31: 0.82},
        18: {13: 0.95, 14: 0.93, 6: 0.92, 15: 0.90, 16: 0.88, 3: 0.85, 21: 0.85, 34: 0.88, 12: 0.78, 30: 0.80, 31: 0.82},
        19: {0: 0.88, 12: 0.82, 1: 0.78, 3: 0.75, 20: 0.75},
        20: {32: 0.95, 0: 0.80, 3: 0.78, 1: 0.70},
        21: {28: 0.95, 11: 0.70, 26: 0.72, 27: 0.70, 17: 0.65},
        22: {29: 0.95, 24: 0.90, 25: 0.82, 4: 0.70},
    }
    
    # Множители по бюджету
    BUDGET_MULTIPLIERS = {
        "free": {
            0: 0.7, 1: 0.7, 2: 0.9, 3: 0.9, 6: 1.0, 12: 0.9, 13: 1.0, 
            14: 1.0, 15: 1.0, 16: 1.0, 20: 0.7, 21: 0.9, 30: 1.0, 
            31: 1.0, 34: 1.0, 23: 0.8
        },
        "cheap": {
            0: 0.9, 1: 0.9, 7: 0.9, 8: 0.8, 20: 0.9, 22: 0.9, 24: 0.9
        },
        "moderate": {
            0: 1.0, 1: 1.0, 4: 0.9, 7: 1.0, 8: 1.0, 9: 0.9, 22: 1.0, 32: 0.9
        },
        "unlimited": {
            0: 1.0, 1: 1.0, 4: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 11: 1.0, 
            17: 1.0, 28: 1.0, 33: 1.0
        },
    }
    
    # Множители по объёму данных
    VOLUME_MULTIPLIERS = {
        "huge": {2: 1.2, 14: 1.15, 1: 1.1, 3: 1.0},
        "large": {2: 1.1, 14: 1.1, 1: 1.1, 3: 1.0, 0: 0.9},
        "medium": {},
        "small": {},
    }
    
    # Бонусы и пенальти
    GPU_BOOST = {6: 1.3, 14: 1.3, 15: 1.2, 13: 1.2, 30: 1.1, 31: 1.1}
    COMPLEXITY_PENALTY = {6: 0.6, 14: 0.6, 15: 0.7}  # если prefer_easy
    PRIVACY_BOOST = {14: 1.4, 6: 1.3, 15: 1.3, 30: 1.3, 13: 1.2}
    
    def score(self, parsed: ParsedQuery) -> np.ndarray:
        """Вычисляет rule-based скоры для всех моделей."""
        scores = np.zeros(NUM_MODELS, dtype=np.float32)
        
        # 1. Базовые скоры по задачам
        for task_id in parsed.tasks:
            if task_id in self.TASK_SCORES:
                for model_id, base_score in self.TASK_SCORES[task_id].items():
                    weight = parsed.task_scores.get(task_id, 1.0)
                    scores[model_id] += base_score * weight
        
        # 2. Множители бюджета
        budget_mult = self.BUDGET_MULTIPLIERS.get(parsed.budget.value, {})
        for model_id, mult in budget_mult.items():
            scores[model_id] *= mult
        
        # Пенальти для платных если хочется бесплатно
        if parsed.budget.value == "free":
            paid_only = [4, 7, 8, 9, 11, 17, 28, 33]
            for m in paid_only:
                scores[m] *= 0.1
        
        # 3. Объём данных
        volume_mult = self.VOLUME_MULTIPLIERS.get(parsed.volume.value, {})
        for model_id, mult in volume_mult.items():
            scores[model_id] *= mult
        
        # 4. GPU
        if parsed.gpu:
            for model_id, boost in self.GPU_BOOST.items():
                scores[model_id] *= boost
        
        # 5. Приоритеты
        if parsed.priorities[4] > 3.5:  # ease
            for model_id, penalty in self.COMPLEXITY_PENALTY.items():
                scores[model_id] *= penalty
        
        if parsed.priorities[3] > 3.5:  # privacy
            for model_id, boost in self.PRIVACY_BOOST.items():
                scores[model_id] *= boost
        
        # Нормализация
        total = scores.sum()
        if total > 0:
            scores = scores / total
        
        return scores


class Recommender:
    """Основной класс для получения рекомендаций."""
    
    def __init__(self, network=None):
        self.parser = QueryParser()
        self.scorer = ModelScorer()
        self.network = network  # Optional: для NN-скоринга
    
    def _explain(self, model_id: int, parsed: ParsedQuery) -> str:
        """Генерирует объяснение выбора."""
        reasons = []
        
        task_names = {
            0: "генерация текстов", 1: "написание и отладка кода",
            2: "анализ данных", 3: "анализ документов",
            4: "генерация изображений", 5: "создание видео",
            6: "генерация музыки", 7: "озвучка и синтез речи",
            8: "транскрибация аудио", 9: "перевод текстов",
            10: "поиск информации", 11: "обучение и подготовка",
            12: "математика и логика", 13: "диалог и ассистент",
            14: "дизайн и визуализация", 15: "маркетинг и SMM",
            16: "большие объёмы данных", 17: "локальный запуск",
            18: "бесплатное использование", 19: "SEO-оптимизация",
            20: "офис и продуктивность", 21: "бизнес-видео с аватарами",
            22: "логотипы и вектор",
        }
        
        for t in parsed.tasks:
            if t in task_names:
                reasons.append(task_names[t])
        
        if parsed.budget.value == "free" and model_id in [13, 14, 6, 15, 30, 31]:
            reasons.append("бесплатная/open-source")
        
        if parsed.gpu and model_id in [6, 14, 15, 13, 30, 31]:
            reasons.append("работает на локальном GPU")
        
        if not reasons:
            reasons.append("лучшее совпадение параметров")
        
        return "; ".join(reasons[:3])
    
    def recommend(
        self,
        query: str,
        top_k: int = 3,
        access_filter: Optional[Literal["free", "paid"]] = None
    ) -> Dict:
        """Главный метод: парсит запрос и возвращает рекомендации."""
        
        # 1. Парсим запрос
        parsed = self.parser.parse(query)
        
        # 2. Получаем скоры
        rule_scores = self.scorer.score(parsed)
        
        # Если есть нейросеть — комбинируем
        if self.network is not None:
            try:
                nn_scores = self.network.predict(parsed)
                # Комбинация: 55% NN + 45% rules
                final_scores = 0.55 * nn_scores + 0.45 * rule_scores
            except Exception as e:
                print(f"⚠️ NN failed: {e}, using rule-based only")
                final_scores = rule_scores
        else:
            final_scores = rule_scores
        
        # 3. Фильтрация по доступности
        candidates = []
        for model_id, score in enumerate(final_scores):
            model = get_model(model_id)
            if model is None:
                continue
            
            if access_filter == "free" and not model.is_free_capable:
                continue
            if access_filter == "paid" and not model.is_paid_capable:
                continue
            
            candidates.append((model_id, score))
        
        # 4. Сортируем и формируем результаты
        candidates.sort(key=lambda x: -x[1])
        top_candidates = candidates[:top_k]
        
        # Нормализуем confidence
        total_score = sum(s for _, s in top_candidates) or 1.0
        
        recommendations = []
        for rank, (model_id, raw_score) in enumerate(top_candidates, 1):
            model = get_model(model_id)
            confidence = raw_score / total_score
            
            rec = Recommendation(
                rank=rank,
                model=model,
                confidence=confidence,
                explanation=self._explain(model_id, parsed),
                score_breakdown={
                    "raw_score": float(raw_score),
                    "rule_component": float(rule_scores[model_id]),
                    "nn_component": float(nn_scores[model_id]) if self.network else 0.0
                }
            )
            recommendations.append(rec)
        
        return {
            "query": query,
            "parsed": parsed,
            "recommendations": recommendations,
            "all_scores": final_scores,
            "rule_scores": rule_scores,
            "nn_scores": nn_scores if self.network else None
        }
    
    def get_alternatives(self, model_id: int, top_k: int = 3) -> List[Recommendation]:
        """Находит похожие модели (если выбранная недоступна)."""
        target = get_model(model_id)
        if target is None:
            return []
        
        # Считаем схожесть по тегам
        similarities = []
        for mid, model in MODELS.items():
            if mid == model_id:
                continue
            
            # Jaccard similarity по тегам
            tags_target = set(target.tags)
            tags_model = set(model.tags)
            if tags_target and tags_model:
                intersection = len(tags_target & tags_model)
                union = len(tags_target | tags_model)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            # Бонус за ту же компанию
            if target.company == model.company:
                similarity += 0.3
            
            # Бонус за тот же tier
            if target.tier == model.tier:
                similarity += 0.1
            
            similarities.append((mid, similarity))
        
        similarities.sort(key=lambda x: -x[1])
        
        result = []
        for rank, (mid, sim) in enumerate(similarities[:top_k], 1):
            model = get_model(mid)
            common_tags = set(target.tags) & set(model.tags)
            explanation = f"Похожие теги: {', '.join(common_tags)}" if common_tags else "Схожие характеристики"
            
            result.append(Recommendation(
                rank=rank,
                model=model,
                confidence=min(sim, 1.0),
                explanation=explanation,
                score_breakdown={"similarity": sim}
            ))
        
        return result


