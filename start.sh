#!/bin/bash

# Переходим в папку со скриптом
cd "$(dirname "$0")"

# Активируем venv если есть
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Запускаем
python3 run.py "$@"

#!/bin/bash
uvicorn advisor.api.main:app --host 0.0.0.0 --port ${PORT:-8000}