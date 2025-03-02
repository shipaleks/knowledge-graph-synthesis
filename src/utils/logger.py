"""
Модуль для настройки и управления логированием в приложении.

Предоставляет функции для получения логгеров с единой
конфигурацией для всего приложения.
"""

import os
import logging
import colorlog
from typing import Optional

# Глобальные переменные для настройки логирования
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Словарь для преобразования строковых названий уровней логирования в константы
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Настройка форматирования логов
_LOG_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Цветовая схема для разных уровней логирования
_LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white"
}

# Флаг, указывающий, была ли уже выполнена инициализация корневого логгера
_INITIALIZED = False


def initialize_logger() -> None:
    """
    Инициализирует корневой логгер с заданными настройками.
    
    Эта функция должна вызываться один раз при запуске приложения.
    """
    global _INITIALIZED
    
    if _INITIALIZED:
        return
    
    # Получение уровня логирования
    log_level = _LOG_LEVELS.get(_LOG_LEVEL, logging.INFO)
    
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Удаление существующих обработчиков
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Создание форматтера для консольного вывода с поддержкой цветов
    console_formatter = colorlog.ColoredFormatter(
        _LOG_FORMAT, 
        log_colors=_LOG_COLORS
    )
    
    # Создание обработчика для консольного вывода
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # Добавление обработчика к корневому логгеру
    root_logger.addHandler(console_handler)
    
    # Создание обработчика для записи в файл
    file_formatter = logging.Formatter(_LOG_FILE_FORMAT)
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    
    # Добавление обработчика к корневому логгеру
    root_logger.addHandler(file_handler)
    
    # Установка флага инициализации
    _INITIALIZED = True
    
    # Логирование успешной инициализации
    root_logger.debug("Логгер инициализирован с уровнем %s", _LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает логгер с заданным именем.
    
    Если корневой логгер еще не инициализирован, инициализирует его.
    
    Args:
        name (str): Имя логгера, обычно __name__ модуля
        
    Returns:
        logging.Logger: Настроенный объект логгера
    """
    if not _INITIALIZED:
        initialize_logger()
    
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Устанавливает уровень логирования для всех логгеров.
    
    Args:
        level (str): Уровень логирования ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    if level.upper() not in _LOG_LEVELS:
        logging.warning("Неизвестный уровень логирования: %s. Используется INFO.", level)
        level = "INFO"
    
    log_level = _LOG_LEVELS[level.upper()]
    
    # Установка уровня для корневого логгера и всех обработчиков
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    
    logging.info("Уровень логирования изменен на %s", level.upper()) 