"""
Модуль для управления конфигурацией приложения.

Загружает и предоставляет доступ к общим настройкам приложения,
таким как язык интерфейса, директория для вывода, максимальное
количество итераций рекурсивного рассуждения и т.д.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Исправляем импорт для работы при запуске из командной строки
try:
    from ..utils.logger import get_logger
except ImportError:
    from src.utils.logger import get_logger

logger = get_logger(__name__)

class AppConfig:
    """
    Класс для управления конфигурацией приложения.
    
    Загружает настройки из файла .env и предоставляет к ним доступ.
    """
    
    _instance = None
    
    def __new__(cls):
        """Реализация паттерна Singleton для обеспечения единственного экземпляра конфигурации."""
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Инициализирует экземпляр конфигурации.
        
        Загружает настройки из переменных окружения и устанавливает значения по умолчанию.
        """
        # Проверка инициализации
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
        
        # Базовые настройки приложения
        self.language = clean_env_value("APP_LANGUAGE", "en")
        self.max_iterations = int(clean_env_value("APP_MAX_ITERATIONS", "10"))
        self.output_dir = Path(clean_env_value("APP_OUTPUT_DIR", "./output"))
        
        # Настройки логирования
        self.log_level = clean_env_value("LOG_LEVEL", "INFO")
        self.log_file = clean_env_value("LOG_FILE", "app.log")
        
        # Настройки производительности
        self.cache_enabled = clean_env_value("CACHE_ENABLED", "true").lower() == "true"
        self.cache_max_size = int(clean_env_value("CACHE_MAX_SIZE", "100"))
        self.batch_size = int(clean_env_value("BATCH_SIZE", "5"))
        
        # Создание выходной директории, если она не существует
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        logger.debug("Конфигурация приложения загружена успешно")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Возвращает все настройки в виде словаря.
        
        Returns:
            Dict[str, Any]: Словарь с настройками приложения
        """
        return {
            "language": self.language,
            "max_iterations": self.max_iterations,
            "output_dir": str(self.output_dir),
            "log_level": self.log_level,
            "log_file": self.log_file,
            "cache_enabled": self.cache_enabled,
            "cache_max_size": self.cache_max_size,
            "batch_size": self.batch_size
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения настройки по ключу.
        
        Args:
            key (str): Ключ настройки
            default (Any, optional): Значение по умолчанию, если настройка не найдена
            
        Returns:
            Any: Значение настройки или значение по умолчанию
        """
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Установка значения настройки.
        
        Args:
            key (str): Ключ настройки
            value (Any): Новое значение
        """
        setattr(self, key, value)
        logger.debug(f"Настройка {key} изменена на {value}")
    
    @staticmethod
    def get_instance() -> 'AppConfig':
        """
        Статический метод для получения экземпляра конфигурации.
        
        Returns:
            AppConfig: Экземпляр класса конфигурации
        """
        return AppConfig() 