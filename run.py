#!/usr/bin/env python
"""
Скрипт для запуска приложения.

Добавляет текущую директорию в PYTHONPATH и запускает приложение.
"""

import os
import sys
import subprocess

# Добавляем текущую директорию в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Запускаем приложение
if __name__ == "__main__":
    from src.main import main
    main() 