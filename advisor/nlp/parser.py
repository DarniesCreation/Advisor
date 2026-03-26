
"""Умный парсер запросов: keywords + sentence embeddings."""

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np

# Пробуем импортировать sentence-transformers, но не падаем если нет
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class Volume(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"


class Budget(str, Enum):
    FREE = "free"
    CHEAP = "cheap"
    MODERATE = "moderate"
    UNLIMITED = "unlimited"


@dataclass
class ParsedQuery:
    """Результат парсинга запроса."""
    tasks: List[int]
    task_scores: Dict[int, float]
    volume: Volume
    budget: Budget
    gpu: bool
    priorities: List[float]  # [quality, speed, cost, privacy, ease]
    prefer_easy: bool
    negations: Dict
    original_text: str
    semantic_similarities: Optional[Dict[int, float]] = None


class TaskDefinitions:
    """Определения задач для семантического поиска."""
    
    TASK_DESCRIPTIONS = {
        0: "Написание текстов, статей, постов, копирайтинг, storytelling, блогинг",
        1: "Программирование, написание кода, отладка, разработка ПО, скрипты, Python, JavaScript",
        2: "Анализ данных, статистика, визуализация, работа с таблицами Excel, pandas, big data",
        3: "Работа с документами, PDF, контракты, юридические тексты, резюме, договоры",
        4: "Генерация изображений, арт, иллюстрации, логотипы, дизайн, картинки, рисование",
        5: "Создание видео, анимация, монтаж, генерация клипов, видеоролики",
        6: "Музыка, песни, саундтреки, генерация аудио, биты, вокал",
        7: "Озвучка текста, TTS, синтез речи, клонирование голоса, диктор",
        8: "Транскрибация, расшифровка аудио в текст, субтитры, speech to text",
        9: "Перевод текстов, локализация, мультиязычность, с английского, на русский",
        10: "Поиск информации, факты, исследования, источники, deep research",
        11: "Обучение, подготовка к экзаменам, репетиторство, курсы, объяснение",
        12: "Математика, формулы, решение уравнений, логика, физика, доказательства",
        13: "Диалог, чат, общий ассистент, советы, разговор, вопросы и ответы",
        14: "Дизайн интерфейсов, UI/UX, макеты, Figma, прототипирование",
        15: "Маркетинг, SMM, реклама, продвижение, соцсети, таргет, бренд",
        16: "Большие данные, Big Data, миллионы записей, терабайты, huge datasets",
        17: "Локальный запуск, приватность, offline, без интернета, GDPR, свой сервер",
        18: "Бесплатные инструменты, open-source, без денег, free software",
        19: "SEO, ключевые слова, поисковая оптимизация, ранжирование",
        20: "Офис, продуктивность, Word, Excel, PowerPoint, документы Microsoft",
        21: "Бизнес-видео, аватары, корпоративное обучение, презентации с аватаром",
        22: "Векторная графика, SVG, логотипы для печати, масштабируемые иконки",
    }
    
    NUM_TASKS = 23


class KeywordParser:
    """Классический keyword-based парсинг."""
    
    KEYWORD_MAP = {
        # Тексты (0)
        "текст": (0, 1.0), "статья": (0, 1.0), "пост": (0, 1.0),
        "написать текст": (0, 1.2), "напиши текст": (0, 1.2),
        "эссе": (0, 1.0), "сочинение": (0, 1.0), "блог": (0, 1.0),
        "письмо": (0, 0.8), "рассказ": (0, 1.0), "история": (0, 0.7),
        "контент": (0, 1.0), "копирайт": (0, 1.0), "рерайт": (0, 1.0),
        "write": (0, 1.0), "text": (0, 1.0), "article": (0, 1.0),
        "описание": (0, 0.8), "резюме": (0, 0.9), "отзыв": (0, 0.8),
        "сценарий": (0, 1.0), "книга": (0, 1.0), "роман": (0, 1.0),
        "стихи": (0, 1.0), "эссе": (0, 1.0),
        
        # Код (1)
        "код": (1, 1.2), "программ": (1, 1.2), "скрипт": (1, 1.0),
        "python": (1, 1.2), "javascript": (1, 1.2), "java": (1, 1.0),
        "code": (1, 1.2), "coding": (1, 1.2), "баг": (1, 1.0),
        "debug": (1, 1.0), "отладк": (1, 1.0), "функци": (1, 0.8),
        "api": (1, 1.0), "бэкенд": (1, 1.0), "фронтенд": (1, 1.0),
        "backend": (1, 1.0), "frontend": (1, 1.0), "сайт": (1, 0.9),
        "приложени": (1, 1.0), "бот": (1, 1.0), "автоматизац": (1, 0.9),
        "html": (1, 1.0), "css": (1, 1.0), "sql": (1, 1.0),
        "react": (1, 1.0), "django": (1, 1.0), "flask": (1, 1.0),
        "алгоритм": (1, 0.9), "рефакторинг": (1, 1.0),
        "typescript": (1, 1.2), "c++": (1, 1.2), "rust": (1, 1.2),
        "docker": (1, 1.0), "deploy": (1, 0.9), "devops": (1, 1.0),
        "vibe coding": (1, 1.2), "вайб кодинг": (1, 1.2),
        
        # Анализ данных (2)
        "анализ данных": (2, 1.3), "данные": (2, 0.9), "таблиц": (2, 1.2),
        "excel": (2, 1.2), "csv": (2, 1.2), "database": (2, 1.2),
        "база данных": (2, 1.3), "датасет": (2, 1.2),
        "аналитик": (2, 1.0), "статистик": (2, 1.0),
        "график": (2, 0.9), "визуализац": (2, 0.9), "дашборд": (2, 1.0),
        "data": (2, 0.8), "analysis": (2, 1.0), "big data": (2, 1.3),
        "парс": (2, 0.9), "pandas": (2, 1.1),
        
        # Документы (3)
        "документ": (3, 1.2), "pdf": (3, 1.3), "книг": (3, 1.0),
        "файл": (3, 0.9), "отчёт": (3, 1.0), "отчет": (3, 1.0),
        "договор": (3, 1.1), "контракт": (3, 1.1), "юридич": (3, 1.0),
        "конспект": (3, 1.0), "реферат": (3, 1.0), "диплом": (3, 1.0),
        "резюмир": (3, 1.0), "summarize": (3, 1.0), "пересказ": (3, 1.0),
        
        # Картинки (4)
        "картинк": (4, 1.3), "изображени": (4, 1.3), "фото": (4, 1.0),
        "нарисо": (4, 1.3), "нарисуй": (4, 1.3), "рисунок": (4, 1.2),
        "арт": (4, 1.2), "иллюстрац": (4, 1.2), "лого": (4, 1.1),
        "логотип": (4, 1.1), "аватар": (4, 1.0), "обложк": (4, 1.0),
        "image": (4, 1.2), "picture": (4, 1.2), "draw": (4, 1.2),
        "баннер": (4, 1.0), "постер": (4, 1.0), "стикер": (4, 1.0),
        "иконк": (4, 1.0), "мем": (4, 0.9), "аниме": (4, 1.1),
        "фотореализм": (4, 1.2), "концепт-арт": (4, 1.2),
        
        # Видео (5)
        "видео": (5, 1.3), "видеоролик": (5, 1.3), "клип": (5, 1.2),
        "анимац": (5, 1.2), "монтаж": (5, 1.0), "ролик": (5, 1.2),
        "video": (5, 1.2), "рилс": (5, 1.0), "reels": (5, 1.0),
        "тикток": (5, 1.0), "shorts": (5, 1.0),
        
        # Музыка (6)
        "музык": (6, 1.3), "песн": (6, 1.3), "трек": (6, 1.2),
        "мелоди": (6, 1.2), "бит": (6, 1.0), "минус": (6, 1.0),
        "music": (6, 1.2), "song": (6, 1.2), "рэп": (6, 1.0),
        "джингл": (6, 1.0), "саундтрек": (6, 1.0), "лофи": (6, 1.0),
        
        # Озвучка (7)
        "озвучк": (7, 1.3), "озвучить": (7, 1.3), "голос": (7, 1.2),
        "дикторск": (7, 1.2), "речь": (7, 1.0), "tts": (7, 1.3),
        "text to speech": (7, 1.3), "клонирование голос": (7, 1.3),
        "подкаст": (7, 1.0), "аудиокниг": (7, 1.1), "начитать": (7, 1.2),
        
        # Транскрибация (8)
        "транскриб": (8, 1.3), "расшифро": (8, 1.3), "субтитр": (8, 1.2),
        "аудио в текст": (8, 1.3), "речь в текст": (8, 1.3),
        "speech to text": (8, 1.3), "transcrib": (8, 1.3),
        "запись": (8, 0.7), "интервью": (8, 0.9), "лекци": (8, 0.9),
        
        # Перевод (9)
        "перевод": (9, 1.3), "перевести": (9, 1.3), "translat": (9, 1.3),
        "на английский": (9, 1.2), "на русский": (9, 1.2),
        "с английского": (9, 1.2), "локализац": (9, 1.0),
        
        # Поиск (10)
        "поиск": (10, 1.2), "найти информац": (10, 1.3), "загугли": (10, 1.2),
        "факт": (10, 1.0), "актуальн": (10, 1.0), "новост": (10, 1.0),
        "search": (10, 1.2), "research": (10, 1.0), "источник": (10, 1.1),
        "проверить": (10, 0.9), "правда ли": (10, 1.0),
        
        # Обучение (11)
        "учёб": (11, 1.0), "учеб": (11, 1.0), "обучени": (11, 1.0),
        "экзамен": (11, 1.2), "подготов": (11, 1.0), "объясн": (11, 1.0),
        "понять": (11, 0.9), "разобраться": (11, 0.9), "курс": (11, 0.9),
        "learn": (11, 1.0), "study": (11, 1.0), "репетитор": (11, 1.1),
        
        # Математика (12)
        "математик": (12, 1.3), "формул": (12, 1.2), "уравнени": (12, 1.2),
        "вычисл": (12, 1.0), "интеграл": (12, 1.2), "физик": (12, 1.0),
        "math": (12, 1.2), "доказ": (12, 1.0), "логик": (12, 1.0),
        "reasoning": (12, 1.2), "олимпиад": (12, 1.1),
        
        # Чат (13)
        "чат": (13, 1.0), "поговор": (13, 1.0), "пообщаться": (13, 1.0),
        "ассистент": (13, 1.0), "помощник": (13, 0.9), "совет": (13, 0.8),
        "chat": (13, 1.0), "спросить": (13, 0.8), "вопрос": (13, 0.7),
        
        # Дизайн (14)
        "дизайн": (14, 1.2), "ui": (14, 1.2), "ux": (14, 1.2),
        "макет": (14, 1.1), "презентац": (14, 1.0), "слайд": (14, 1.0),
        "design": (14, 1.2), "figma": (14, 1.0), "прототип": (14, 1.0),
        "вектор": (14, 1.1), "svg": (14, 1.2),
        
        # Маркетинг (15)
        "маркетинг": (15, 1.2), "smm": (15, 1.2), "реклам": (15, 1.0),
        "продвижени": (15, 1.0), "соцсет": (15, 1.0),
        "таргет": (15, 1.0), "воронк": (15, 1.0), "бренд": (15, 0.9),
        
        # Big Data (16)
        "большая база": (16, 1.3), "миллион": (16, 1.2), "100 тысяч": (16, 1.2),
        "огромн": (16, 1.0), "100k": (16, 1.2), "терабайт": (16, 1.3),
        "много данных": (16, 1.2), "big data": (16, 1.3),
        
        # Локально (17)
        "локальн": (17, 1.3), "приватн": (17, 1.3), "конфиденциальн": (17, 1.2),
        "без интернет": (17, 1.3), "оффлайн": (17, 1.3), "offline": (17, 1.3),
        "local": (17, 1.2), "privacy": (17, 1.2), "свой сервер": (17, 1.3),
        "gdpr": (17, 1.2),
        
        # Бесплатно (18)
        "бесплатн": (18, 1.3), "free": (18, 1.2), "даром": (18, 1.0),
        "без денег": (18, 1.2), "не платить": (18, 1.2),
        "дёшев": (18, 0.9), "дешев": (18, 0.9), "недорог": (18, 0.9),
        "open source": (18, 1.2), "open-source": (18, 1.2),
        
        # SEO (19)
        "seo": (19, 1.3), "ключевые слова": (19, 1.2),
        "поисковая выдача": (19, 1.2), "ранжирован": (19, 1.1),
        
        # Офис (20)
        "офис": (20, 1.2), "office": (20, 1.2), "word": (20, 1.0),
        "powerpoint": (20, 1.0), "teams": (20, 1.0), "outlook": (20, 1.0),
        "excel отчёт": (20, 1.0), "автоматизация офис": (20, 1.2),
        
        # Бизнес-видео (21)
        "аватар видео": (21, 1.3), "обучающее видео": (21, 1.2),
        "корпоративн видео": (21, 1.2), "lip sync": (21, 1.0),
        
        # Логотипы/вектор (22)
        "логотип вектор": (22, 1.4), "svg логотип": (22, 1.4),
        "вектор логотип": (22, 1.3), "svg": (22, 1.2),
    }
    
    NGRAM_MAP = {
        "написать код": (1, 1.4), "помоги с кодом": (1, 1.4),
        "создать сайт": (1, 1.3), "сделать бота": (1, 1.3),
        "нарисовать картинку": (4, 1.5), "создать картинку": (4, 1.5),
        "сделать видео": (5, 1.5), "создать видео": (5, 1.5),
        "написать музыку": (6, 1.5), "создать песню": (6, 1.5),
        "озвучить текст": (7, 1.5), "перевести текст": (9, 1.4),
        "найти информацию": (10, 1.4), "проанализировать данные": (2, 1.5),
        "расшифровать аудио": (8, 1.5), "расшифровать запись": (8, 1.5),
        "запустить локально": (17, 1.5), "написать статью": (0, 1.4),
        "решить уравнение": (12, 1.5), "анализ документов": (3, 1.5),
        "прочитать pdf": (3, 1.5), "создать логотип": (4, 1.4),
        "создать презентацию": (20, 1.4), "обучающий ролик": (21, 1.4),
        "видео с аватаром": (21, 1.5), "корпоративное обучение": (21, 1.4),
        "сделать логотип svg": (22, 1.5), "вектор логотип": (22, 1.4),
        "агентный код": (1, 1.3), "мульти агент": (1, 1.2),
    }
    
    NEGATION_PATTERNS = [
        (r"не хочу платить", {"budget": "free"}),
        (r"без оплат", {"budget": "free"}),
        (r"не разбираюсь", {"prefer_easy": True}),
        (r"без настройк", {"prefer_easy": True}),
        (r"не для кода", {"exclude_task": 1}),
        (r"не код", {"exclude_task": 1}),
    ]
    
    VOLUME_KEYWORDS = {
        Volume.SMALL: ["маленьк", "немного", "пару", "один", "мало", "чуть-чуть"],
        Volume.MEDIUM: ["средн", "несколько", "пара тысяч", "нормально"],
        Volume.LARGE: ["большо", "большая", "много", "тысяч", "50k", "100k",
                       "десятки тысяч", "крупн", "солидн"],
        Volume.HUGE: ["миллион", "1m", "10m", "терабайт", "огромн", "гигантск",
                      "big data", "сотни тысяч", "500k", "масштаб"],
    }
    
    BUDGET_KEYWORDS = {
        Budget.FREE: ["бесплатн", "free", "даром", "без денег", "не платить", 
                      "халяв", "open source", "open-source"],
        Budget.CHEAP: ["дёшев", "дешев", "недорог", "до 10", "бюджетн", "эконом"],
        Budget.MODERATE: ["средн бюджет", "до 50", "нормальн цен"],
        Budget.UNLIMITED: ["любой бюджет", "без ограничен", "деньги не важн", 
                           "премиум", "лучшее"],
    }


class SemanticParser:
    """Sentence-transformers для семантического сходства."""
    
    def __init__(self):
        self.model = None
        self.task_embeddings = {}
        
        if EMBEDDINGS_AVAILABLE:
            try:
                # Лёгкая мультиязычная модель
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self._precompute_embeddings()
                print("✅ Semantic embeddings loaded")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить embeddings: {e}")
    
    def _precompute_embeddings(self):
        """Предвычисляем эмбеддинги для всех задач."""
        for task_id, desc in TaskDefinitions.TASK_DESCRIPTIONS.items():
            self.task_embeddings[task_id] = self.model.encode(desc, convert_to_tensor=False)
    
    def parse(self, text: str) -> Dict[int, float]:
        """Возвращает семантическое сходство для каждой задачи."""
        if self.model is None or not self.task_embeddings:
            return {}
        
        query_vec = self.model.encode(text, convert_to_tensor=False)
        
        similarities = {}
        for task_id, task_vec in self.task_embeddings.items():
            # Косинусное сходство
            sim = np.dot(query_vec, task_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(task_vec) + 1e-8
            )
            similarities[task_id] = float(sim)
        
        return similarities


class QueryParser:
    """Комбинированный парсер: keywords + semantics."""
    
    def __init__(self):
        self.keyword_parser = KeywordParser()
        self.semantic_parser = SemanticParser()
    
    def _extract_ngrams(self, text: str, n_range=(2, 4)) -> List[str]:
        """Извлекает n-граммы из текста."""
        words = re.findall(r'[a-zа-яё0-9+#-]+', text.lower())
        ngrams = []
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i:i + n]))
        return ngrams
    
    def _detect_negations(self, text: str) -> Tuple[Dict, List[int]]:
        """Обнаруживает негации и исключения."""
        mods = {}
        text_lower = text.lower()
        exclude_cats = []
        
        for pattern, effect in KeywordParser.NEGATION_PATTERNS:
            if re.search(pattern, text_lower):
                mods.update(effect)
                if "exclude_task" in effect:
                    exclude_cats.append(effect["exclude_task"])
        
        return mods, exclude_cats
    
    def _detect_volume(self, text: str) -> Volume:
        """Определяет объём данных."""
        text_lower = text.lower()
        for vol, keywords in KeywordParser.VOLUME_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return vol
        return Volume.MEDIUM
    
    def _detect_budget(self, text: str, negations: Dict) -> Budget:
        """Определяет бюджет."""
        # Сначала проверяем негации
        if negations.get("budget") == "free":
            return Budget.FREE
        
        text_lower = text.lower()
        for bud, keywords in KeywordParser.BUDGET_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return bud
        return Budget.MODERATE
    
    def _detect_gpu(self, text: str) -> bool:
        """Определяет наличие GPU."""
        gpu_keywords = ["gpu", "видеокарт", "rtx", "nvidia", "cuda", "локальн",
                       "свой сервер", "мощный компьютер", "3090", "4090"]
        text_lower = text.lower()
        return any(kw in text_lower for kw in gpu_keywords)
    
    def _compute_priorities(self, text: str, budget: Budget, 
                           negations: Dict) -> List[float]:
        """Вычисляет приоритеты [quality, speed, cost, privacy, ease]."""
        text_lower = text.lower()
        priorities = [3.0, 3.0, 3.0, 2.0, 3.0]
        
        # Quality
        quality_kw = ["качествен", "лучш", "идеальн", "точн", "максимальн",
                      "профессиональн", "студийн", "высокий класс"]
        if any(kw in text_lower for kw in quality_kw):
            priorities[0] = 5.0
        
        # Speed
        speed_kw = ["быстр", "срочн", "скорее", "немедленн", "побыстрее",
                    "моментально", "сейчас"]
        if any(kw in text_lower for kw in speed_kw):
            priorities[1] = 5.0
        
        # Cost (автоматически высокий если free)
        if budget == Budget.FREE:
            priorities[2] = 5.0
        
        # Privacy
        privacy_kw = ["приватн", "конфиденциальн", "секретн", "privacy",
                     "без интернет", "gdpr", "локально", "не в облак"]
        if any(kw in text_lower for kw in privacy_kw):
            priorities[3] = 5.0
        
        # Ease
        if negations.get("prefer_easy", False):
            priorities[4] = 5.0
        else:
            ease_kw = ["прост", "легк", "удобн", "для новичк", "не разбираюсь",
                      "без настройк", "интуитивн", "под ключ"]
            if any(kw in text_lower for kw in ease_kw):
                priorities[4] = 5.0
        
        return priorities
    
    def parse(self, text: str) -> ParsedQuery:
        """Основной метод парсинга."""
        text_lower = text.lower().strip()
        
        # 1. Keyword-based scoring
        task_scores = {}
        
        # N-gram matching (более точный)
        for ngram in self._extract_ngrams(text):
            if ngram in KeywordParser.NGRAM_MAP:
                cat_id, weight = KeywordParser.NGRAM_MAP[ngram]
                task_scores[cat_id] = task_scores.get(cat_id, 0) + weight
        
        # Keyword matching
        for keyword, (cat_id, weight) in KeywordParser.KEYWORD_MAP.items():
            if keyword in text_lower:
                # Бонус если слово в начале фразы
                bonus = 1.2 if text_lower.startswith(keyword) else 1.0
                task_scores[cat_id] = task_scores.get(cat_id, 0) + weight * bonus
        
        # 2. Semantic scoring (если доступно)
        semantic_scores = self.semantic_parser.parse(text)
        
        # 3. Комбинируем scores
        combined_scores = task_scores.copy()
        for task_id, sem_score in semantic_scores.items():
            # Нормализуем семантический score к шкале keywords
            # Косинус ~0.3-0.9 → умножаем на 3 для сопоставимости
            normalized_sem = max(0, (sem_score - 0.3) * 5)
            
            if task_id in combined_scores:
                # Берём максимум или усредняем
                combined_scores[task_id] = max(combined_scores[task_id], normalized_sem)
            else:
                combined_scores[task_id] = normalized_sem
        
        # 4. Определяем топ задачи
        if not combined_scores:
            tasks = [13]  # default: chat
            combined_scores = {13: 1.0}
        else:
            # Фильтруем по порогу и берём топ-3
            sorted_tasks = sorted(combined_scores.items(), key=lambda x: -x[1])
            tasks = [t for t, s in sorted_tasks if s >= 0.5][:3]
            if not tasks:
                tasks = [sorted_tasks[0][0]]
        
        # 5. Негации и уточнения
        negations, exclude_cats = self._detect_negations(text)
        tasks = [t for t in tasks if t not in exclude_cats]
        if not tasks:
            tasks = [13]
        
        # 6. Остальные параметры
        volume = self._detect_volume(text)
        budget = self._detect_budget(text, negations)
        gpu = self._detect_gpu(text)
        priorities = self._compute_priorities(text, budget, negations)
        
        return ParsedQuery(
            tasks=tasks,
            task_scores={int(k): float(v) for k, v in combined_scores.items()},
            volume=volume,
            budget=budget,
            gpu=gpu,
            priorities=priorities,
            prefer_easy=negations.get("prefer_easy", False),
            negations=negations,
            original_text=text,
            semantic_similarities=semantic_scores if semantic_scores else None
        )

