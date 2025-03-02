"""
Модуль для управления конфигурацией LLM провайдеров.

Загружает и предоставляет доступ к настройкам различных LLM провайдеров,
таким как Claude, GPT, Gemini, DeepSeek и Ollama, включая API ключи,
названия моделей, параметры генерации и т.д.
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Исправляем импорт для работы при запуске из командной строки
try:
    from ..utils.logger import get_logger
except ImportError:
    from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMConfig:
    """
    Класс для управления конфигурацией LLM провайдеров.
    
    Загружает настройки из файла .env и предоставляет к ним доступ.
    """
    
    _instance = None
    
    # Поддерживаемые провайдеры
    SUPPORTED_PROVIDERS = ["claude", "gpt", "gemini", "deepseek", "ollama"]
    
    def __new__(cls):
        """Реализация паттерна Singleton для обеспечения единственного экземпляра конфигурации."""
        if cls._instance is None:
            cls._instance = super(LLMConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Инициализация объекта конфигурации. Загружает настройки из .env файла."""
        if self._initialized:
            return
            
        # Загрузка переменных окружения из .env файла
        load_dotenv()
        
        # Функция для очистки значений переменных окружения от комментариев
        def clean_env_value(key, default):
            value = os.getenv(key, default)
            # Если значение содержит пробел или табуляцию, обрезаем по первому пробелу или табуляции
            if " " in value or "\t" in value:
                value = value.split(" ")[0].split("\t")[0]
            return value
        
        # Основные настройки LLM
        self.provider = clean_env_value("LLM_PROVIDER", "claude")
        self.model = clean_env_value("LLM_MODEL", "claude-3-7-sonnet-latest")
        self.max_tokens = int(clean_env_value("LLM_MAX_TOKENS", "4000"))
        self.temperature = float(clean_env_value("LLM_TEMPERATURE", "0.3"))
        
        # API ключи для разных провайдеров
        self.api_keys = {
            "claude": clean_env_value("CLAUDE_API_KEY", ""),
            "gpt": clean_env_value("GPT_API_KEY", ""),
            "gemini": clean_env_value("GEMINI_API_KEY", ""),
            "deepseek": clean_env_value("DEEPSEEK_API_KEY", "")
            # Ollama не требует API ключа
        }
        
        # Доступные модели для каждого провайдера
        self.available_models = {
            "claude": ["claude-3-7-sonnet-latest", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "gpt": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
            "ollama": ["llama2", "mistral", "vicuna", "mixtral", "phi"]
        }
        
        # Проверка валидности выбранного провайдера
        if self.provider not in self.SUPPORTED_PROVIDERS:
            logger.warning(f"Неподдерживаемый провайдер: {self.provider}. Используется провайдер по умолчанию: claude")
            self.provider = "claude"
        
        # Проверка валидности выбранной модели
        if self.model not in self.available_models.get(self.provider, []):
            default_model = self.available_models[self.provider][0]
            logger.warning(f"Модель {self.model} недоступна для провайдера {self.provider}. Используется модель по умолчанию: {default_model}")
            self.model = default_model
        
        self._initialized = True
        logger.debug("Конфигурация LLM загружена успешно")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Возвращает все настройки в виде словаря.
        
        Returns:
            Dict[str, Any]: Словарь с настройками LLM
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def get_api_key(self, provider: Optional[str] = None) -> 'Result[str]':
        """
        Получение API ключа для указанного провайдера.
        
        Args:
            provider (Optional[str], optional): Название провайдера. 
                Если не указано, используется текущий провайдер.
            
        Returns:
            Result[str]: API ключ или ошибка, если ключ не найден
        """
        from src.utils.result import Result
        
        provider = provider or self.provider
        
        # Ollama не требует API ключа
        if provider == "ollama":
            return Result.ok("")
        
        api_key = self.api_keys.get(provider, "")
        
        if not api_key:
            return Result.fail(f"API ключ для провайдера {provider} не найден")
        
        return Result.ok(api_key)
    
    def get_models(self, provider: Optional[str] = None) -> List[str]:
        """
        Получение списка доступных моделей для указанного провайдера.
        
        Args:
            provider (Optional[str], optional): Название провайдера.
                Если не указано, используется текущий провайдер.
            
        Returns:
            List[str]: Список доступных моделей
        """
        provider = provider or self.provider
        return self.available_models.get(provider, [])
    
    def set_provider(self, provider: str) -> bool:
        """
        Установка текущего провайдера.
        
        Args:
            provider (str): Название провайдера
            
        Returns:
            bool: True, если провайдер успешно установлен, иначе False
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            logger.warning(f"Неподдерживаемый провайдер: {provider}")
            return False
        
        self.provider = provider
        
        # Обновление модели, если текущая несовместима с новым провайдером
        if self.model not in self.available_models.get(provider, []):
            self.model = self.available_models[provider][0]
            logger.info(f"Модель изменена на {self.model} для провайдера {provider}")
        
        logger.info(f"Установлен провайдер: {provider}")
        return True
    
    def set_model(self, model: str) -> bool:
        """
        Установка текущей модели.
        
        Args:
            model (str): Название модели
            
        Returns:
            bool: True, если модель успешно установлена, иначе False
        """
        if model not in self.available_models.get(self.provider, []):
            logger.warning(f"Модель {model} недоступна для провайдера {self.provider}")
            return False
        
        self.model = model
        logger.info(f"Установлена модель: {model}")
        return True
    
    def validate_api_key(self, provider: Optional[str] = None) -> bool:
        """
        Проверка наличия API ключа для указанного провайдера.
        
        Args:
            provider (Optional[str], optional): Название провайдера.
                Если не указано, используется текущий провайдер.
            
        Returns:
            bool: True, если API ключ существует, иначе False
        """
        provider = provider or self.provider
        
        # Ollama не требует API ключа
        if provider == "ollama":
            return True
        
        api_key_result = self.get_api_key(provider)
        return api_key_result.success
    
    def get_provider_config(self, provider: str) -> 'Result[Dict[str, Any]]':
        """
        Получение конфигурации для указанного провайдера.
        
        Args:
            provider (str): Название провайдера
            
        Returns:
            Result[Dict[str, Any]]: Конфигурация провайдера или ошибка
        """
        from src.utils.result import Result
        
        if provider not in self.SUPPORTED_PROVIDERS:
            return Result.fail(f"Неподдерживаемый провайдер: {provider}")
        
        # Формируем конфигурацию провайдера
        config = {
            "provider": provider,
            "default_model": self.available_models[provider][0] if provider in self.available_models else None,
            "available_models": self.available_models.get(provider, []),
            "api_key": self.api_keys.get(provider, ""),
            "api_key_available": self.validate_api_key(provider)
        }
        
        return Result.ok(config)
    
    @staticmethod
    def get_instance() -> 'LLMConfig':
        """
        Статический метод для получения экземпляра конфигурации.
        
        Returns:
            LLMConfig: Экземпляр класса конфигурации
        """
        return LLMConfig() 