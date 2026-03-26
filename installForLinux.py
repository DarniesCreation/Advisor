#!/usr/bin/env python3
"""Автоустановка зависимостей для AI Model Advisor (Arch Linux / любой ОС)."""

import subprocess
import sys
import os
from pathlib import Path

VENV_DIR = Path(__file__).parent / "venv"


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def create_venv():
    if not VENV_DIR.exists():
        print("🔧 Создаю виртуальное окружение (venv)...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("  ✅ venv создан")
    else:
        print("  ✅ venv уже существует")


def get_pip():
    if os.name == "nt":
        return str(VENV_DIR / "Scripts" / "pip")
    return str(VENV_DIR / "bin" / "pip")


def get_python():
    if os.name == "nt":
        return str(VENV_DIR / "Scripts" / "python")
    return str(VENV_DIR / "bin" / "python")


def install(packages):
    run([get_pip(), "install", "--upgrade"] + packages)


def main():
    print("=" * 50)
    print("📦 AI Model Advisor — установка зависимостей")
    print("=" * 50)

    create_venv()

    print("\n🔄 Обновляю pip...")
    run([get_python(), "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL)

    core = ["numpy", "rich", "pydantic", "fastapi", "uvicorn"]
    print("\n📥 Устанавливаю core-зависимости...")
    try:
        install(core)
        print("  ✅ numpy, rich, pydantic, fastapi, uvicorn")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Ошибка: {e}")
        sys.exit(1)

    print("\nУстанавливать torch + sentence-transformers?")
    print("(нужны для нейросети и семантического поиска, ~2-3 ГБ)")
    ans = input("Установить? [y/N]: ").strip().lower()

    if ans == "y":
        for pkg in ["torch", "sentence-transformers"]:
            print(f"📥 Устанавливаю {pkg}...")
            try:
                install([pkg])
                print(f"  ✅ {pkg}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ {pkg}: {e}")
    else:
        print("  ⏭ Пропущено — будет работать в rule-only режиме")

    # Создаём удобный скрипт запуска
    if os.name == "nt":
        script = Path(__file__).parent / "start.bat"
        script.write_text(f'@echo off\n"{get_python()}" run.py %*\n')
        print(f"\n✅ Готово! Запуск: start.bat cli")
    else:
        script = Path(__file__).parent / "start.sh"
        script.write_text(f'#!/bin/bash\n"{get_python()}" run.py "$@"\n')
        script.chmod(0o755)
        print(f"\n✅ Готово!")
        print(f"   ./start.sh cli   — интерактивный режим")
        print(f"   ./start.sh api   — запустить API")
        print(f"\n   Или напрямую: {get_python()} run.py cli")


if __name__ == "__main__":
    main()
