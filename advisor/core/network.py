
"""Нейросеть рекомендаций на PyTorch с улучшенной архитектурой."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class TaskEncoder(nn.Module):
    """Кодирует задачи с учётом их взаимосвязей."""
    
    def __init__(self, num_tasks: int, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, task_indices: torch.Tensor, task_scores: torch.Tensor):
        """
        task_indices: [batch, num_tasks] - бинарная маска активных задач
        task_scores: [batch, num_tasks] - веса задач
        """
        # Получаем эмбеддинги всех задач
        all_indices = torch.arange(self.embedding.num_embeddings, device=task_indices.device)
        all_emb = self.embedding(all_indices)  # [num_tasks, hidden]
        
        # Умножаем на scores для взвешивания
        batch_size = task_indices.size(0)
        weighted = all_emb.unsqueeze(0) * task_scores.unsqueeze(-1)  # [batch, num_tasks, hidden]
        
        # Self-attention между задачами
        attn_out, _ = self.attention(weighted, weighted, weighted)
        attn_out = self.norm(attn_out + weighted)
        
        # Суммируем по задачам (только активные)
        masked = attn_out * task_indices.unsqueeze(-1)
        return masked.sum(dim=1)  # [batch, hidden]


class AdvisorNet(nn.Module):
    """Улучшенная нейросеть для рекомендаций моделей."""
    
    def __init__(
        self,
        num_tasks: int = 23,
        num_models: int = 35,
        task_hidden: int = 64,
        context_dim: int = 16,
        hidden_dims: list = [256, 128, 64]
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.num_models = num_models
        
        # Кодировщик задач
        self.task_encoder = TaskEncoder(num_tasks, task_hidden)
        
        # Контекстные признаки (volume, budget, gpu, priorities)
        self.context_dim = context_dim
        self.context_proj = nn.Linear(5 + 3, context_dim)  # 5 priorities + volume/budget/gpu
        
        # Общее представление
        combined_dim = task_hidden + context_dim
        
        # MLP для скоринга моделей
        layers = []
        prev_dim = combined_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            ])
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        
        # Выходной слой: по одному скору на модель
        self.scorer = nn.Linear(prev_dim, num_models)
        
        # Skip connection: прямая связь от задач к моделям (как в твоём rule-based)
        self.task_to_model = nn.Linear(task_hidden, num_models, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        task_indices: torch.Tensor,
        task_scores: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            task_indices: [batch, num_tasks] - бинарная маска
            task_scores: [batch, num_tasks] - веса задач
            context: [batch, 8] - volume, budget, gpu, 5 priorities
        
        Returns:
            logits: [batch, num_models] - скоры для каждой модели
        """
        # Кодируем задачи
        task_repr = self.task_encoder(task_indices, task_scores)  # [batch, task_hidden]
        
        # Кодируем контекст
        context_repr = self.context_proj(context)  # [batch, context_dim]
        
        # Объединяем
        combined = torch.cat([task_repr, context_repr], dim=-1)
        
        # Пропускаем через MLP
        features = self.backbone(combined)
        
        # Основной скор
        main_scores = self.scorer(features)
        
        # Skip connection от задач
        skip_scores = self.task_to_model(task_repr)
        
        # Итоговый скор (комбинация как в твоём коде: 0.55 * nn + 0.45 * rules)
        # Здесь skip играет роль rule-based компонента
        final_scores = 0.6 * main_scores + 0.4 * skip_scores
        
        return F.softmax(final_scores, dim=-1)
    
    def predict(self, parsed_query) -> np.ndarray:
        """Удобный метод для инференса из parsed query."""
        # Конвертируем parsed query в тензоры
        task_indices = torch.zeros(1, self.num_tasks)
        task_scores = torch.zeros(1, self.num_tasks)
        
        for t in parsed_query.tasks:
            task_indices[0, t] = 1.0
            task_scores[0, t] = parsed_query.task_scores.get(t, 1.0)
        
        # Нормализуем scores
        if task_scores.sum() > 0:
            task_scores = task_scores / task_scores.sum()
        
        # Контекст
        volume_map = {"small": 0.0, "medium": 0.33, "large": 0.66, "huge": 1.0}
        budget_map = {"free": 0.0, "cheap": 0.33, "moderate": 0.66, "unlimited": 1.0}
        
        context = torch.tensor([[
            *parsed_query.priorities,  # 5 значений
            volume_map.get(parsed_query.volume, 0.33),
            budget_map.get(parsed_query.budget, 0.66),
            1.0 if parsed_query.gpu else 0.0
        ]])
        
        with torch.no_grad():
            probs = self.forward(task_indices, task_scores, context)
        
        return probs.numpy()[0]
    
    def save(self, path: str):
        """Сохраняет модель и конфиг."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'num_tasks': self.num_tasks,
                'num_models': self.num_models,
                'task_hidden': self.task_encoder.embedding.embedding_dim,
                'context_dim': self.context_dim,
            }
        }, path)
        print(f"💾 Модель сохранена: {path}")
    
    def load(self, path: str) -> bool:
        """Загружает модель."""
        path = Path(path)
        if not path.exists():
            return False
        
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        print(f"📂 Модель загружена: {path}")
        return True


class DummyNetwork:
    """Fallback если PyTorch недоступен — использует rule-based скоринг."""
    
    def __init__(self, num_models: int = 35):
        self.num_models = num_models
    
    def predict(self, parsed_query) -> np.ndarray:
        # Возвращает равномерное распределение (будет переопределено скорером)
        return np.ones(self.num_models) / self.num_models
    
    def save(self, path: str):
        pass
    
    def load(self, path: str) -> bool:
        return False

