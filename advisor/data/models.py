"""Метаданные AI-моделей — ИСПРАВЛЕННЫЕ ЦЕНЫ И ACCESS (март 2026)."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class AccessType(str, Enum):
    FREE = "free"          # Полностью бесплатно, open-source
    PAID = "paid"          # Только платно
    FREEMIUM = "freemium"  # Бесплатный лимит + платная версия


class Tier(str, Enum):
    FREE = "free"
    STANDARD = "standard"
    BALANCED = "balanced"
    PRO = "pro"
    PREMIUM = "premium"


@dataclass(frozen=True)
class AIModel:
    id: int
    name: str
    company: str
    description: str
    price: str
    url: str
    plus: str
    minus: str
    tags: List[str] = field(default_factory=list)
    access: AccessType = AccessType.PAID
    tier: Tier = Tier.STANDARD

    @property
    def is_free_capable(self) -> bool:
        return self.access in (AccessType.FREE, AccessType.FREEMIUM)

    @property
    def is_paid_capable(self) -> bool:
        return self.access in (AccessType.PAID, AccessType.FREEMIUM)

    @property
    def access_label(self) -> str:
        labels = {
            AccessType.FREE: "🆓 Бесплатная",
            AccessType.PAID: "💳 Платная",
            AccessType.FREEMIUM: "🆓/💳 Freemium",
        }
        return labels.get(self.access, "💳 Платная")


# ═══════════════════════════════════════════════════════════════
#  МОДЕЛИ — актуальные на март 2026
# ═══════════════════════════════════════════════════════════════

MODELS: Dict[int, AIModel] = {

    # ── 0: ChatGPT ──────────────────────────────────────────────
    # ✅ GPT-5 / 5.4 — актуально
    # ✅ Добавлен план Go $8/мес
    # ✅ Access: FREEMIUM — корректно (есть бесплатный лимит)
    0: AIModel(
        id=0,
        name="ChatGPT (GPT-5 / 5.4)",
        company="OpenAI",
        description="Флагман OpenAI. GPT-5.4, reasoning, DALL-E, Sora, Codex.",
        price="Бесплатный лимит / $8 Go / $20 Plus / $200 Pro",
        url="https://chat.openai.com",
        plus="Универсальность, GPT-5.4 reasoning, GPTs, DALL-E, Sora",
        minus="Цензура, лимиты бесплатной версии, Go с рекламой",
        tags=["текст", "код", "перевод", "анализ", "диалог"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 1: Claude Opus 4.6 ──────────────────────────────────────
    # 🔴 ИСПРАВЛЕНО: Claude 3.5 Opus НИКОГДА НЕ СУЩЕСТВОВАЛ
    #    Линейка: 3 Opus → 4 Opus → 4.1 → 4.5 → 4.6 (5 фев 2026)
    # 🔴 ИСПРАВЛЕНО access: FREEMIUM → PAID
    #    Opus 4.6 НЕ доступен на бесплатном тире!
    #    Бесплатный тир даёт только Sonnet, а Opus — только Pro/Max
    # ✅ Контекст 1M, API $5/$25 per MTok
    1: AIModel(
        id=1,
        name="Claude Opus 4.6",
        company="Anthropic",
        description="Флагман Anthropic (5 фев 2026). 1M контекст, agent teams.",
        price="$20/мес Pro / $100–$200 Max / API: $5/$25 per MTok",
        url="https://claude.ai",
        plus="Лучший в коде и reasoning, 1M контекст, agent teams",
        minus="Нет на бесплатном тире, дорого, нет генерации картинок",
        tags=["код", "документы", "контекст", "анализ", "агент"],
        access=AccessType.PAID,
        tier=Tier.PREMIUM,
    ),

    # ── 2: Kimi ─────────────────────────────────────────────────
    2: AIModel(
        id=2,
        name="Kimi K1.5",
        company="Moonshot AI",
        description="Контекст до 2M токенов. Сильный в рассуждениях.",
        price="Бесплатная / платная",
        url="https://kimi.moonshot.cn",
        plus="Гигантский контекст, reasoning",
        minus="Слабее в креативе",
        tags=["контекст", "анализ", "reasoning"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 3: Gemini Pro ───────────────────────────────────────────
    # ✅ Обновлено: Gemini 2.5 Pro (1.5 устарел)
    3: AIModel(
        id=3,
        name="Gemini 2.5 Pro",
        company="Google",
        description="Мультимодальная: текст, картинки, видео, код. 1M+ контекст.",
        price="Бесплатный лимит / $19.99/мес Pro / $124.99/3 мес Ultra",
        url="https://gemini.google.com",
        plus="Мультимодальность, 1M+ контекст, thinking-native",
        minus="Галлюцинации, нестабильность",
        tags=["мультимодальная", "текст", "код", "видео"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 4: Midjourney ───────────────────────────────────────────
    # ✅ Обновлено v6 → v7
    4: AIModel(
        id=4,
        name="Midjourney v7",
        company="Midjourney",
        description="Король эстетики. Арт, фотореализм, Draft Mode.",
        price="$10–$120/мес (только платно!)",
        url="https://midjourney.com",
        plus="Лучшее арт-качество, Draft Mode 10x быстрее",
        minus="Только картинки, только платная, нет бесплатного пробного",
        tags=["картинки", "арт", "дизайн", "фотореализм"],
        access=AccessType.PAID,
        tier=Tier.PREMIUM,
    ),

    # ── 5: DALL-E / GPT Image ───────────────────────────────────
    # ✅ Access: FREEMIUM — бесплатные пользователи ChatGPT
    #    имеют ограниченную генерацию картинок
    5: AIModel(
        id=5,
        name="DALL-E 4 / GPT Image",
        company="OpenAI",
        description="Генерация картинок нативно в ChatGPT через GPT-5.",
        price="Лимит бесплатно / полный доступ в Plus $20/мес",
        url="https://chat.openai.com",
        plus="Понимание промптов, текст на картинках, нативная генерация",
        minus="Лимиты на бесплатном, цензура",
        tags=["картинки", "генерация", "редактирование"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 6: Stable Diffusion ─────────────────────────────────────
    6: AIModel(
        id=6,
        name="Stable Diffusion XL/3.5",
        company="Stability AI",
        description="Открытая модель. Запуск локально.",
        price="Бесплатная (open-source)",
        url="https://stability.ai",
        plus="Бесплатная, локально, LoRA, безлимит",
        minus="Нужна GPU, сложная настройка",
        tags=["картинки", "бесплатная", "локально", "open-source"],
        access=AccessType.FREE,
        tier=Tier.FREE,
    ),

    # ── 7: GitHub Copilot ───────────────────────────────────────
    # 🔴 ИСПРАВЛЕНО access: PAID → FREEMIUM
    #    GitHub Copilot Free запущен (ограниченный бесплатный тир)
    7: AIModel(
        id=7,
        name="GitHub Copilot",
        company="GitHub/Microsoft",
        description="AI-ассистент для кода в IDE. Мультимодель, agent mode.",
        price="Бесплатный тир / $10/мес Pro / $39/мес Pro+",
        url="https://github.com/features/copilot",
        plus="Бесплатный тир, мультимодель (GPT-5, Claude, Gemini)",
        minus="Слабее Cursor в агентных задачах",
        tags=["код", "IDE", "автодополнение", "агент"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 8: Cursor ───────────────────────────────────────────────
    8: AIModel(
        id=8,
        name="Cursor",
        company="Anysphere",
        description="AI-IDE на базе VS Code.",
        price="Бесплатный лимит / $20/мес Pro",
        url="https://cursor.com",
        plus="AI-IDE, агент Composer, понимание проекта",
        minus="Дорого для Ultra, привязка к IDE",
        tags=["код", "IDE", "редактор", "агент"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 9: ElevenLabs ───────────────────────────────────────────
    9: AIModel(
        id=9,
        name="ElevenLabs",
        company="ElevenLabs",
        description="Генерация речи, клонирование голоса.",
        price="Бесплатный лимит / от $5/мес",
        url="https://elevenlabs.io",
        plus="Самые реалистичные голоса",
        minus="Лимиты бесплатного",
        tags=["озвучка", "голос", "tts"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 10: Suno ────────────────────────────────────────────────
    10: AIModel(
        id=10,
        name="Suno AI",
        company="Suno",
        description="Генерация музыки и песен с вокалом.",
        price="Бесплатный лимит / от $10/мес",
        url="https://suno.com",
        plus="Полные песни с вокалом",
        minus="Не студийное качество",
        tags=["музыка", "песни", "вокал"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 11: Runway ──────────────────────────────────────────────
    # 🔴 ИСПРАВЛЕНО access: PAID → FREEMIUM
    #    Runway имеет Free план (125 одноразовых кредитов),
    #    а также обновлена до Gen-4/4.5
    # 🔴 ИСПРАВЛЕНО: цена $15/мес Standard (не $12)
    11: AIModel(
        id=11,
        name="Runway Gen-4 / Gen-4.5",
        company="Runway",
        description="Генерация и редактирование видео. Gen-4, Gen-4.5, Aleph.",
        price="Free (125 кредитов) / $15/мес Standard / $35/мес Pro",
        url="https://runwayml.com",
        plus="Видео из текста/картинки, профессиональные эффекты, Aleph",
        minus="Кредитная система, короткие клипы (до 16 сек)",
        tags=["видео", "анимация", "эффекты"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 12: Perplexity ──────────────────────────────────────────
    12: AIModel(
        id=12,
        name="Perplexity AI",
        company="Perplexity",
        description="AI-поисковик с источниками.",
        price="Бесплатная / $20/мес Pro",
        url="https://perplexity.ai",
        plus="Актуальная инфа, источники",
        minus="Не для генерации контента",
        tags=["поиск", "факты", "источники"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 13: DeepSeek ────────────────────────────────────────────
    13: AIModel(
        id=13,
        name="DeepSeek V3/R1",
        company="DeepSeek",
        description="Мощная open-source модель. Код, математика, reasoning.",
        price="Бесплатная (open-source + бесплатный API)",
        url="https://chat.deepseek.com",
        plus="Бесплатная, MIT лицензия, self-host",
        minus="Серверы нестабильны, Китай",
        tags=["бесплатная", "код", "математика", "open-source"],
        access=AccessType.FREE,
        tier=Tier.FREE,
    ),

    # ── 14: Llama ───────────────────────────────────────────────
    14: AIModel(
        id=14,
        name="Llama 3.3/4",
        company="Meta",
        description="Открытая модель. Локальный запуск, дообучение.",
        price="Бесплатная (open-source)",
        url="https://llama.meta.com",
        plus="Бесплатная, огромный контекст, приватность",
        minus="Нужно мощное железо",
        tags=["бесплатная", "локально", "open-source", "контекст"],
        access=AccessType.FREE,
        tier=Tier.FREE,
    ),

    # ── 15: Whisper ─────────────────────────────────────────────
    15: AIModel(
        id=15,
        name="Whisper",
        company="OpenAI",
        description="Расшифровка аудио в текст. Локально.",
        price="Бесплатная (open-source)",
        url="https://github.com/openai/whisper",
        plus="Точная транскрибация, бесплатная, 90+ языков",
        minus="Только speech-to-text, нужна GPU",
        tags=["транскрибация", "аудио", "бесплатная"],
        access=AccessType.FREE,
        tier=Tier.FREE,
    ),

    # ── 16: NotebookLM ──────────────────────────────────────────
    16: AIModel(
        id=16,
        name="NotebookLM",
        company="Google",
        description="Анализ документов + подкасты.",
        price="Бесплатная",
        url="https://notebooklm.google.com",
        plus="Анализ документов, подкасты, бесплатная",
        minus="Только загруженные документы",
        tags=["документы", "подкаст", "анализ"],
        access=AccessType.FREE,
        tier=Tier.FREE,
    ),

    # ── 17: Sora ────────────────────────────────────────────────
    # ✅ Access: PAID — подтверждено
    #    С 10 января 2026 бесплатный доступ полностью убран.
    #    Только Plus ($20) / Pro ($200). Sora 1 сворачивается.
    # ⚠️  24 марта 2026 OpenAI объявила о закрытии Sora app/API
    17: AIModel(
        id=17,
        name="Sora 2",
        company="OpenAI",
        description="Генерация видео из текста. Закрывается (март 2026).",
        price="Только ChatGPT Plus $20/мес (1000 кр.) / Pro $200/мес",
        url="https://sora.com",
        plus="Реалистичное видео, до 20 сек 1080p (Pro), физика",
        minus="Закрывается, дорого, лимиты, нет бесплатного доступа",
        tags=["видео", "генерация"],
        access=AccessType.PAID,
        tier=Tier.PREMIUM,
    ),

    # ── 18: Grok ────────────────────────────────────────────────
    # 🔴 ИСПРАВЛЕНО access: PAID → FREEMIUM
    #    Grok имеет бесплатный тир (10 промптов / 2 часа),
    #    а также standalone SuperGrok $30/мес
    # 🔴 ИСПРАВЛЕНО: Grok 4 (вышел июль 2025), не 2/3
    #    Grok 4 Heavy для Pro, standalone SuperGrok
    18: AIModel(
        id=18,
        name="Grok 4",
        company="xAI (SpaceX)",
        description="Модель от xAI. Данные из X/Twitter. Standalone + X.",
        price="Бесплатный лимит (10/2ч) / SuperGrok $30/мес / X Premium+ $8/мес",
        url="https://grok.com",
        plus="Бесплатный тир, мало цензуры, актуальные данные X, standalone",
        minus="Лимиты бесплатного, Imagine теперь только платно",
        tags=["чат", "без цензуры", "twitter", "код"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 19: Leonardo ────────────────────────────────────────────
    19: AIModel(
        id=19,
        name="Leonardo AI",
        company="Leonardo",
        description="Генерация картинок, обучение моделей.",
        price="150 бесплатных/день / платная",
        url="https://leonardo.ai",
        plus="Удобный UI, бесплатные кредиты",
        minus="Качество ниже Midjourney",
        tags=["картинки", "генерация", "UI"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 20: Claude Sonnet 4.6 ───────────────────────────────────
    # 🔴 ИСПРАВЛЕНО: Claude 3.5 Sonnet → Sonnet 4.6 (17 фев 2026)
    # ✅ Access: FREEMIUM — корректно
    #    Бесплатный тир Claude даёт доступ к Sonnet (не Opus!)
    20: AIModel(
        id=20,
        name="Claude Sonnet 4.6",
        company="Anthropic",
        description="Opus-уровень по цене Sonnet. 1M контекст (17 фев 2026).",
        price="Бесплатный лимит (Sonnet) / $20/мес Pro / API: $3/$15 MTok",
        url="https://claude.ai",
        plus="70% предпочтений над Sonnet 4.5, дешевле Opus, 1M контекст",
        minus="Нет генерации картинок",
        tags=["код", "текст", "документы", "агент"],
        access=AccessType.FREEMIUM,
        tier=Tier.BALANCED,
    ),

    # ── 21: Gemini Flash ────────────────────────────────────────
    # ✅ Обновлено: 2.5 Flash (1.5 устарел)
    21: AIModel(
        id=21,
        name="Gemini 2.5 Flash",
        company="Google",
        description="Быстрая и дёшевая модель для продакшна.",
        price="Бесплатная в AI Studio / API: $0.30/$2.50 per MTok",
        url="https://gemini.google.com",
        plus="Очень быстрая, дешёвая, free tier, хорошее качество",
        minus="Слабее Pro в сложных задачах",
        tags=["текст", "код", "быстрая", "бесплатная"],
        access=AccessType.FREE,
        tier=Tier.BALANCED,
    ),

    # ── 22: Windsurf ────────────────────────────────────────────
    22: AIModel(
        id=22,
        name="Windsurf",
        company="Codeium",
        description="AI-IDE с агентом Cascade.",
        price="Бесплатный лимит / $15–$20/мес",
        url="https://windsurf.com",
        plus="Cascade агент, Memories, дешевле Cursor",
        minus="Кредитная система",
        tags=["код", "IDE", "агент"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 23: Flux ────────────────────────────────────────────────
    23: AIModel(
        id=23,
        name="Flux.1 [pro/dev]",
        company="Black Forest Labs",
        description="Лидер фотореализма. Открытые веса.",
        price="5 бесплатных/день / API ~$0.03/img",
        url="https://blackforestlabs.ai",
        plus="Лучший фотореализм, open-weight",
        minus="Нужен точный промпт",
        tags=["картинки", "фотореализм", "open-source"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 24: Ideogram ────────────────────────────────────────────
    24: AIModel(
        id=24,
        name="Ideogram 2.0",
        company="Ideogram",
        description="Лучший текст на картинках.",
        price="Бесплатный лимит / от $7/мес",
        url="https://ideogram.ai",
        plus="90%+ точность текста, логотипы",
        minus="Арт-качество ниже Midjourney",
        tags=["картинки", "логотипы", "текст"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 25: Adobe Firefly ───────────────────────────────────────
    # 🔴 ИСПРАВЛЕНО access: PAID → FREEMIUM
    #    Adobe Firefly имеет бесплатный план (25 кредитов/мес)!
    #    Также есть standalone Firefly планы от $4.99/мес
    25: AIModel(
        id=25,
        name="Adobe Firefly",
        company="Adobe",
        description="AI-генератор на лицензированных данных. All-in-one AI studio.",
        price="Бесплатно (25 кредитов/мес) / от $4.99/мес Starter / $9.99 Pro",
        url="https://firefly.adobe.com",
        plus="Коммерчески безопасно, интеграция Photoshop, бесплатный тир",
        minus="Кредитная система, Fast mode 2x кредитов",
        tags=["картинки", "дизайн", "коммерческий", "видео"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 26: Veo ─────────────────────────────────────────────────
    # ✅ Обновлено: Veo 3.1
    26: AIModel(
        id=26,
        name="Veo 3.1",
        company="Google",
        description="Генерация видео от Google. До 60 сек.",
        price="В Gemini Pro/Ultra подписке / API",
        url="https://deepmind.google/technologies/veo",
        plus="Качественное видео до 60 сек, хорошая физика",
        minus="Нужна подписка Gemini",
        tags=["видео", "генерация"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 27: Kling ───────────────────────────────────────────────
    27: AIModel(
        id=27,
        name="Kling AI",
        company="Kuaishou",
        description="Генерация видео из текста/картинки.",
        price="Бесплатные кредиты / платная",
        url="https://klingai.com",
        plus="Длинные клипы до 2 мин, реалистичное движение",
        minus="Серверы в Китае",
        tags=["видео", "анимация", "генерация"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 28: Synthesia ───────────────────────────────────────────
    28: AIModel(
        id=28,
        name="Synthesia",
        company="Synthesia",
        description="AI-видео с аватарами для бизнеса.",
        price="От $22/мес (только платно)",
        url="https://synthesia.io",
        plus="240+ аватаров, 160+ языков",
        minus="Дорого, только для бизнеса",
        tags=["видео", "аватары", "бизнес"],
        access=AccessType.PAID,
        tier=Tier.PRO,
    ),

    # ── 29: Recraft ─────────────────────────────────────────────
    29: AIModel(
        id=29,
        name="Recraft V3",
        company="Recraft",
        description="AI-генератор с SVG-экспортом.",
        price="Бесплатный лимит / платная",
        url="https://recraft.ai",
        plus="SVG экспорт, brand styling",
        minus="Узкая ниша",
        tags=["логотипы", "вектор", "svg"],
        access=AccessType.FREEMIUM,
        tier=Tier.STANDARD,
    ),

    # ── 30: Mistral ─────────────────────────────────────────────
    30: AIModel(
        id=30,
        name="Mistral Large",
        company="Mistral AI",
        description="Европейская open-source модель.",
        price="Бесплатная (open-source) / API",
        url="https://mistral.ai",
        plus="Европейская, GDPR-compliant, open-source",
        minus="Слабее GPT-5/Claude 4.6",
        tags=["текст", "open-source", "мультиязычная"],
        access=AccessType.FREE,
        tier=Tier.STANDARD,
    ),

    # ── 31: Qwen ────────────────────────────────────────────────
    31: AIModel(
        id=31,
        name="Qwen 2.5",
        company="Alibaba",
        description="Китайская open-source модель.",
        price="Бесплатная (open-source)",
        url="https://qwen.ai",
        plus="Бесплатная, сильная в азиатских языках",
        minus="Китайская компания",
        tags=["бесплатная", "open-source", "мультиязычная"],
        access=AccessType.FREE,
        tier=Tier.STANDARD,
    ),

    # ── 32: Microsoft Copilot ───────────────────────────────────
    32: AIModel(
        id=32,
        name="Microsoft Copilot",
        company="Microsoft",
        description="AI в Windows, Office, Edge.",
        price="Бесплатная / $30/мес Pro",
        url="https://copilot.microsoft.com",
        plus="Интеграция в Office/Windows",
        minus="Дорого для Pro",
        tags=["офис", "продуктивность", "windows"],
        access=AccessType.FREEMIUM,
        tier=Tier.PRO,
    ),

    # ── 33: OpenAI Codex ────────────────────────────────────────
    33: AIModel(
        id=33,
        name="OpenAI Codex",
        company="OpenAI",
        description="Облачный агент-кодер.",
        price="Только ChatGPT Pro/Team ($200/мес)",
        url="https://openai.com/codex",
        plus="Автономное написание кода",
        minus="Очень дорого",
        tags=["код", "агент", "автоматизация"],
        access=AccessType.PAID,
        tier=Tier.PREMIUM,
    ),

    # ── 34: Imagen 4 ────────────────────────────────────────────
    34: AIModel(
        id=34,
        name="Imagen 4",
        company="Google",
        description="AI-генератор картинок Google. Интеграция в Gemini.",
        price="Бесплатная в AI Studio / API",
        url="https://deepmind.google/technologies/imagen",
        plus="Лучшая композиция сцен, бесплатная в AI Studio",
        minus="Строгая модерация",
        tags=["картинки", "фотореализм", "бесплатная"],
        access=AccessType.FREE,
        tier=Tier.PRO,
    ),
}

NUM_MODELS = len(MODELS)


def get_model(model_id: int) -> Optional[AIModel]:
    return MODELS.get(model_id)


def get_models_by_tag(tag: str) -> List[AIModel]:
    return [m for m in MODELS.values() if tag in m.tags]


def get_free_models() -> List[AIModel]:
    """Только полностью бесплатные (не freemium)."""
    return [m for m in MODELS.values() if m.access == AccessType.FREE]


def get_freemium_models() -> List[AIModel]:
    """Есть бесплатный лимит + платная версия."""
    return [m for m in MODELS.values() if m.access == AccessType.FREEMIUM]


def get_paid_models() -> List[AIModel]:
    """Только платные."""
    return [m for m in MODELS.values() if m.access == AccessType.PAID]