"""
Модуль конфигурации приложения.

Предоставляет классы и функции для управления настройками приложения 
и LLM провайдеров.
"""

from .app_config import AppConfig
from .llm_config import LLMConfig

# Экземпляры для удобного импорта
app_config = AppConfig.get_instance()
llm_config = LLMConfig.get_instance()

__all__ = ['AppConfig', 'LLMConfig', 'app_config', 'llm_config']
