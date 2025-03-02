"""
Модуль для замера времени выполнения операций.

Предоставляет класс Timer для замера времени выполнения функций и блоков кода.
"""

import time
import functools
from typing import Optional, Callable, Any, TypeVar

# Исправляем импорты для работы при запуске из командной строки
try:
    from .logger import get_logger
except ImportError:
    from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')  # Тип возвращаемого значения декорируемой функции


class Timer:
    """
    Класс для замера времени выполнения операций.
    
    Может использоваться как декоратор функции или как контекстный менеджер.
    
    Examples:
        # Использование как декоратор
        @Timer(name="my_function")
        def my_function():
            # ...код функции...
        
        # Использование как контекстный менеджер
        with Timer(name="operation"):
            # ...код операции...
    """
    
    def __init__(self, name: Optional[str] = None, log_level: str = "DEBUG"):
        """
        Инициализация объекта Timer.
        
        Args:
            name (Optional[str], optional): Название таймера. По умолчанию None.
            log_level (str, optional): Уровень логирования. По умолчанию "DEBUG".
        """
        self.name = name
        self.start_time = 0.0
        self.log_level = log_level.upper()
    
    def __enter__(self) -> 'Timer':
        """
        Метод входа в контекстный менеджер.
        
        Returns:
            Timer: Экземпляр таймера
        """
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Метод выхода из контекстного менеджера.
        
        Args:
            exc_type: Тип исключения, если оно возникло
            exc_val: Значение исключения
            exc_tb: Трассировка исключения
        """
        self.stop()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Метод вызова для использования класса как декоратора.
        
        Args:
            func (Callable[..., T]): Декорируемая функция
            
        Returns:
            Callable[..., T]: Декорированная функция
        """
        name = self.name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with Timer(name=name, log_level=self.log_level):
                return func(*args, **kwargs)
        
        return wrapper
    
    def start(self) -> None:
        """Запускает таймер."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Останавливает таймер и возвращает затраченное время.
        
        Returns:
            float: Затраченное время в секундах
        """
        elapsed_time = time.time() - self.start_time
        
        if self.name:
            message = f"Время выполнения {self.name}: {elapsed_time:.4f} сек."
            
            if self.log_level == "DEBUG":
                logger.debug(message)
            elif self.log_level == "INFO":
                logger.info(message)
            elif self.log_level == "WARNING":
                logger.warning(message)
            else:
                logger.debug(message)
        
        return elapsed_time
    
    def elapsed(self) -> float:
        """
        Возвращает затраченное время без остановки таймера.
        
        Returns:
            float: Затраченное время в секундах
        """
        return time.time() - self.start_time


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Декоратор для замера времени выполнения функции.
    
    Args:
        func (Callable[..., T]): Декорируемая функция
        
    Returns:
        Callable[..., T]: Декорированная функция
    """
    return Timer()(func) 