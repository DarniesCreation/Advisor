#!/usr/bin/env python3
"""Автоустановка зависимостей для AI Model Advisor."""

import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main():
    print("📦 Установка зависимостей AI Model Advisor...\n")

    # Минимальные (всегда нужны)
    core = ["numpy", "rich", "pydantic", "fastapi", "uvicorn"]

    # Опциональные (тяжёлые)
    optional = ["torch", "sentence-transformers"]

    print("Устанавливаю core-зависимости...")
    for pkg in core:
        try:
            install(pkg)
            print(f"  ✅ {pkg}")
        except Exception as e:
            print(f"  ❌ {pkg}: {e}")

    print("\nУстанавливать torch + sentence-transformers?")
    print("(нужны для нейросети и семантического поиска, ~2-3 ГБ)")
    ans = input("Установить? [y/N]: ").strip().lower()

    if ans == "y":
        for pkg in optional:
            try:
                install(pkg)
                print(f"  ✅ {pkg}")
            except Exception as e:
                print(f"  ❌ {pkg}: {e}")
    else:
        print("  ⏭ Пропущено — будет работать в rule-only режиме")

    print("\n✅ Готово! Запуск: python run.py cli")


if __name__ == "__main__":
    main()
