
"""Интерактивный CLI с Rich."""

import time
import re
from typing import Optional
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Установи rich: pip install rich")

from ..core.recommender import Recommender
from ..data.models import MODELS, get_model, AccessType


class AdvisorUI:
    """Красивый интерфейс для AI Advisor."""
    
    def __init__(self, recommender: Optional[Recommender] = None):
        self.recommender = recommender or Recommender()
        self.console = Console() if RICH_AVAILABLE else None
        self.history = []
        self.last_result = None
    
    def _print(self, text: str, **kwargs):
        """Универсальный print (Rich или обычный)."""
        if self.console:
            self.console.print(text, **kwargs)
        else:
            print(text)
    
    def _format_confidence_bar(self, confidence: float, width: int = 30) -> str:
        """Создаёт визуальный бар уверенности."""
        filled = int(confidence * width)
        if self.console:
            bar = "█" * filled + "░" * (width - filled)
            return f"[{bar}] {confidence:.1%}"
        else:
            bar = "=" * filled + "-" * (width - filled)
            return f"[{bar}] {confidence:.1%}"
    
    def _render_recommendation(self, rec, detailed: bool = False):
        """Рендерит одну рекомендацию."""
        m = rec.model
        
        # Цвет по уверенности
        if rec.confidence > 0.5:
            color = "green"
            prefix = "🥇" if rec.rank == 1 else "🥈" if rec.rank == 2 else "🥉"
        elif rec.confidence > 0.3:
            color = "yellow"
            prefix = "⚡"
        else:
            color = "red"
            prefix = "•"
        
        if not self.console:
            # Простой текстовый вывод
            lines = [
                f"\n{prefix} #{rec.rank} — {m.name} ({m.company})",
                f"   Уверенность: {rec.confidence:.1%}",
                f"   Доступ: {m.access_label} | Tier: {m.tier.value}",
                f"   {m.description}",
                f"   💰 {m.price}",
                f"   ✅ {m.plus}",
                f"   ❌ {m.minus}",
                f"   💡 Почему: {rec.explanation}",
            ]
            if detailed:
                lines.append(f"   📊 Scores: {rec.score_breakdown}")
            return "\n".join(lines)
        
        # Rich вывод
        header = f"[bold {color}]{prefix} #{rec.rank}. {m.name}[/] [dim]({m.company})[/]"
        
        content = [
            f"[cyan]Уверенность:[/] {self._format_confidence_bar(rec.confidence)}",
            f"[cyan]Доступ:[/] {m.access_label} | Tier: {m.tier.value}",
            f"[cyan]Цена:[/] {m.price}",
            "",
            f"[dim]{m.description}[/]",
            "",
            f"[green]✓ {m.plus}[/]",
            f"[red]✗ {m.minus}[/]",
            "",
            f"[yellow]💡 Почему:[/] {rec.explanation}",
        ]
        
        if detailed:
            content.extend([
                "",
                "[dim]Score breakdown:[/]",
                f"  Raw: {rec.score_breakdown.get('raw_score', 0):.3f}",
                f"  Rule: {rec.score_breakdown.get('rule_component', 0):.3f}",
                f"  NN: {rec.score_breakdown.get('nn_component', 0):.3f}",
            ])
        
        return Panel(
            "\n".join(content),
            title=header,
            border_style=color,
            box=box.ROUNDED
        )
    
    def _render_comparison(self, model_ids: list[int]):
        """Сравнение моделей в таблице."""
        models = [get_model(mid) for mid in model_ids if get_model(mid)]
        if len(models) < 2:
            self._print("[red]Нужно минимум 2 модели[/]")
            return
        
        if not self.console:
            # Простой текст
            for m in models:
                self._print(f"\n[{m.id}] {m.name}")
                self._print(f"  Компания: {m.company}")
                self._print(f"  Цена: {m.price}")
                self._print(f"  Плюсы: {m.plus}")
                self._print(f"  Минусы: {m.minus}")
            return
        
        # Rich таблица
        table = Table(title="⚖️ Сравнение моделей", box=box.DOUBLE_EDGE)
        
        table.add_column("Параметр", style="cyan", no_wrap=True)
        for m in models:
            table.add_column(m.name[:18], justify="center")
        
        rows = [
            ("Компания", lambda m: m.company),
            ("Доступ", lambda m: m.access_label),
            ("Tier", lambda m: m.tier.value),
            ("Цена", lambda m: m.price[:30]),
            ("Плюсы", lambda m: m.plus[:40]),
            ("Минусы", lambda m: m.minus[:40]),
            ("Теги", lambda m: ", ".join(m.tags[:3])),
        ]
        
        for label, getter in rows:
            row = [label]
            for m in models:
                val = getter(m)
                row.append(str(val))
            table.add_row(*row)
        
        self.console.print(table)
    
    def _show_catalog(self):
        """Показывает каталог моделей по категориям."""
        categories = {
            "🧠 LLM / Чатботы": [0, 1, 20, 3, 21, 18, 13, 14, 30, 31, 2],
            "💻 Код / IDE": [7, 8, 22, 33],
            "🎨 Картинки": [4, 5, 23, 24, 25, 6, 19, 29, 34],
            "🎬 Видео": [11, 17, 26, 27, 28],
            "🎵 Аудио": [9, 10, 15],
            "🔍 Поиск / Документы": [12, 16],
            "🏢 Продуктивность": [32],
        }
        
        for cat_name, ids in categories.items():
            self._print(f"\n[bold blue]{cat_name}[/]" if self.console else f"\n=== {cat_name} ===")
            
            for mid in ids:
                m = get_model(mid)
                if m:
                    free_badge = "[green]FREE[/]" if self.console and m.is_free_capable else "FREE" if m.is_free_capable else "PAID"
                    if self.console:
                        self._print(f"  [{mid:2d}] [bold]{m.name}[/] {free_badge} [dim]- {m.description[:50]}...[/]")
                    else:
                        self._print(f"  [{mid:2d}] {m.name} ({free_badge}) - {m.description[:50]}...")
    
    def _show_history(self):
        """История запросов."""
        if not self.history:
            self._print("[dim]История пуста[/]" if self.console else "История пуста")
            return
        
        if not self.console:
            for e in self.history[-10:]:
                self._print(f"  [{e['time']}] {e['query'][:40]} → {e['top_model']} ({e['confidence']:.0%})")
            return
        
        table = Table(title="📜 История запросов")
        table.add_column("Время", style="dim")
        table.add_column("Запрос")
        table.add_column("Рекомендация", style="green")
        table.add_column("Уверенность")
        
        for entry in self.history[-10:]:
            table.add_row(
                entry["time"],
                entry["query"][:40],
                entry["top_model"],
                f"{entry['confidence']:.0%}"
            )
        
        self.console.print(table)
    
    def _show_probs_chart(self, result: dict):
        """График вероятностей."""
        probs = result["all_scores"]
        top_indices = sorted(range(len(probs)), key=lambda i: -probs[i])[:10]
        
        max_prob = max(probs[top_indices[0]], 1e-8)
        
        if not self.console:
            self._print("\nТоп-10 моделей:")
            for i in top_indices:
                m = get_model(i)
                p = probs[i]
                bar = "=" * int((p / max_prob) * 20) + "-" * (20 - int((p / max_prob) * 20))
                marker = " <- BEST" if i == top_indices[0] else ""
                self._print(f"  {m.name[:25]:25} [{bar}] {p:.1%}{marker}")
            return
        
        # Rich таблица
        table = Table(title="📊 Распределение вероятностей", box=box.SIMPLE)
        table.add_column("Модель", style="cyan")
        table.add_column("Вероятность", justify="right")
        table.add_column("Визуализация")
        
        for i in top_indices:
            m = get_model(i)
            if not m:
                continue
            p = probs[i]
            bar_width = int((p / max_prob) * 20)
            bar = "█" * bar_width + "░" * (20 - bar_width)
            
            style = "bold green" if i == top_indices[0] else ""
            
            table.add_row(
                m.name[:25],
                f"{p:.1%}",
                f"[{bar}]",
                style=style
            )
        
        self.console.print(table)
    
    def run_query(self, query: str, detailed: bool = False) -> dict:
        """Выполняет один запрос и показывает результат."""
        start_time = time.time()
        
        # Показываем спиннер если есть Rich
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task(description="Анализирую запрос...", total=None)
                result = self.recommender.recommend(query, top_k=3)
        else:
            self._print("Анализирую...")
            result = self.recommender.recommend(query, top_k=3)
        
        elapsed = (time.time() - start_time) * 1000
        self.last_result = result
        
        # Показываем детали запроса
        parsed = result["parsed"]
        self._print(f"\n[bold]📝 Запрос:[/] {query}" if self.console else f"\nЗапрос: {query}")
        
        # Детекция задач
        task_names = {
            0: "✍️ Тексты", 1: "💻 Код", 2: "📊 Анализ данных",
            3: "📄 Документы", 4: "🎨 Картинки", 5: "🎬 Видео",
            6: "🎵 Музыка", 7: "🔊 Озвучка", 8: "🎙 Транскрибация",
            9: "🌐 Перевод", 10: "🔍 Поиск", 11: "📚 Обучение",
            12: "🔢 Математика", 13: "💬 Чат", 14: "🎨 Дизайн",
            15: "📣 Маркетинг", 16: "📦 Big Data", 17: "🔒 Локально",
            18: "🆓 Бесплатно", 19: "📈 SEO", 20: "🏢 Офис",
            21: "🎥 Бизнес-видео", 22: "✏️ Логотипы/Вектор",
        }
        
        detected = [task_names.get(t, f"#{t}") for t in parsed.tasks]
        self._print(f"[bold]🎯 Задачи:[/] {', '.join(detected)}" if self.console else f"Задачи: {', '.join(detected)}")
        
        # Параметры
        params = []
        params.append(f"💰 {parsed.budget.value}" if self.console else f"Бюджет: {parsed.budget.value}")
        params.append(f"📦 {parsed.volume.value}" if self.console else f"Объём: {parsed.volume.value}")
        params.append(f"🖥 GPU: {'да' if parsed.gpu else 'нет'}")
        
        # Приоритеты
        priority_names = ["качество", "скорость", "цена", "приватность", "простота"]
        high_priority = [(n, v) for n, v in zip(priority_names, parsed.priorities) if v > 4]
        if high_priority:
            params.append(f"⭐ Приоритет: {', '.join(n for n, _ in high_priority)}")
        
        self._print(f"[bold]⚙️ Параметры:[/] {' | '.join(params)}" if self.console else f"Параметры: {' | '.join(params)}")
        
        # Рекомендации
        self._print(f"\n[bold green]🏆 Топ рекомендации:[/]" if self.console else "\nТоп рекомендации:")
        for rec in result["recommendations"]:
            rendered = self._render_recommendation(rec, detailed=detailed)
            if isinstance(rendered, str):
                self._print(rendered)
            else:
                self.console.print(rendered)
        
        # Бесплатные альтернативы
        self._print(f"\n[bold blue]🆓 Бесплатные альтернативы:[/]" if self.console else "\nБесплатные альтернативы:")
        free_result = self.recommender.recommend(query, top_k=2, access_filter="free")
        for rec in free_result["recommendations"]:
            rendered = self._render_recommendation(rec)
            if isinstance(rendered, str):
                self._print(rendered)
            else:
                self.console.print(rendered)
        
        self._print(f"\n[dim]⏱ Обработано за {elapsed:.0f} мс[/]" if self.console else f"\nОбработано за {elapsed:.0f} мс")
        
        # Сохраняем в историю
        if result["recommendations"]:
            top = result["recommendations"][0]
            self.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "query": query,
                "top_model": top.model.name,
                "confidence": top.confidence
            })
        
        return result
    
    def interactive(self):
        """Интерактивный режим."""
        if self.console:
            self.console.print(Panel.fit(
                "[bold cyan]🤖 AI Model Advisor v3.1[/]\n"
                "[dim]Умный подбор AI-моделей по описанию задачи[/]\n"
                "[green]Обновлено: март 2026[/]",
                border_style="cyan"
            ))
        else:
            self._print("=" * 50)
            self._print("🤖 AI Model Advisor v3.1")
            self._print("Умный подбор AI-моделей по описанию задачи")
            self._print("=" * 50)
        
        self._print("\nКоманды: каталог | история | граф | сравни 0 1 | выход | help")
        
        examples = [
            "Нарисуй логотип для кофейни в SVG",
            "Проанализировать 100к строк Excel бесплатно",
            "Сделать музыку для YouTube",
            "Написать код на Python, я новичок",
        ]
        
        self._print("\nПримеры:")
        for e in examples:
            self._print(f"  • {e}")
        
        while True:
            try:
                query = input("\n📝 > " if self.console else "\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("\n👋 Пока!")
                break
            
            if not query:
                continue
            
            cmd = query.lower()
            
            if cmd in ("выход", "exit", "quit", "q"):
                self._print("👋 Пока!")
                break
            
            elif cmd in ("помощь", "help", "?"):
                help_text = """
Команды:
  <текст>         - описать задачу для рекомендации
  каталог         - список всех моделей по категориям
  история         - последние запросы
  граф            - показать распределение вероятностей
  сравни 0 1      - сравнить модели по ID
  детально        - переключить детальный вывод
  выход           - завершить работу
                """
                self._print(help_text)
            
            elif cmd == "каталог":
                self._show_catalog()
            
            elif cmd == "история":
                self._show_history()
            
            elif cmd == "граф":
                if self.last_result:
                    self._show_probs_chart(self.last_result)
                else:
                    self._print("Сначала сделайте запрос")
            
            elif cmd.startswith(("сравни", "compare")):
                nums = re.findall(r'\d+', cmd)
                if len(nums) >= 2:
                    self._render_comparison([int(n) for n in nums[:4]])
                else:
                    self._print("Формат: сравни 0 1 20")
            
            else:
                # Обычный запрос
                try:
                    detailed = "детально" in cmd
                    query_clean = query.replace("детально", "").strip() if detailed else query
                    self.run_query(query_clean, detailed=detailed)
                except Exception as e:
                    self._print(f"Ошибка: {e}")
                    import traceback
                    self._print(traceback.format_exc())


def main():
    """Точка входа CLI."""
    # Создаём рекомендер без нейросети (только rules)
    # Можно добавить загрузку NN если есть файл
    recommender = Recommender(network=None)
    
    ui = AdvisorUI(recommender)
    ui.interactive()


if __name__ == "__main__":
    main()

