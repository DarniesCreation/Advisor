#!/usr/bin/env python3
"""Точка входа для AI Model Advisor."""

import sys
import os

# Добавляем текущую папку в путь (важно!)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import argparse


def main():
    parser = argparse.ArgumentParser(description="AI Model Advisor v3.1")
    parser.add_argument("command", choices=["cli", "test"], 
                       help="Режим работы")
    
    args = parser.parse_args()
    
    if args.command == "cli":
        from advisor.cli.main import main as cli_main
        cli_main()
    
    elif args.command == "test":
        print("🧪 Запуск тестов...")
        from advisor.core.recommender import Recommender
        
        rec = Recommender()
        
        test_cases = [
            ("Нарисуй картинку кота", [4, 5, 6, 23, 34]),
            ("Напиши код на Python", [1, 8, 20, 13]),
            ("Расшифровать аудио лекции", [15]),
            ("Бесплатный ИИ для кода", [13, 14, 6]),
            ("Проанализировать миллион строк данных", [2, 1, 14]),
            ("Сделать SVG логотип", [29, 24, 4]),
        ]
        
        passed = 0
        for query, expected in test_cases:
            result = rec.recommend(query, top_k=1)
            top_id = result["recommendations"][0].model.id
            ok = top_id in expected
            status = "✅" if ok else "❌"
            print(f"  {status} '{query[:35]}...' → {result['recommendations'][0].model.name}")
            if ok:
                passed += 1
        
        print(f"\nИтог: {passed}/{len(test_cases)} ({passed/len(test_cases):.0%})")


if __name__ == "__main__":
    main()