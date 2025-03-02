"""
Модуль для замера времени выполнения операций.

Предоставляет класс Timer для замера времени выполнения функций и блоков кода.
"""

import time
import functools
from typing import Optional, Callable, Any, TypeVar, Dict

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
    
    Может использоваться как декоратор функции, как контекстный менеджер
    или как инструмент для отслеживания нескольких таймеров одновременно.
    
    Examples:
        # Использование как декоратор
        @Timer(name="my_function")
        def my_function():
            # ...код функции...
        
        # Использование как контекстный менеджер
        with Timer(name="operation"):
            # ...код операции...
            
        # Использование для отслеживания нескольких таймеров
        timer = Timer()
        timer_id = timer.start("operation1")
        # ...выполнение операции 1...
        timer.stop(timer_id)
        
        timer_id2 = timer.start("operation2")
        # ...выполнение операции 2...
        timer.stop(timer_id2)
        
        # Получение всех замеров времени
        timings = timer.get_all_timings()
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
        
        # Словарь для хранения нескольких таймеров
        self.timers: Dict[str, Dict[str, Any]] = {}
        # Словарь для хранения результатов замеров
        self.timings: Dict[str, float] = {}
    
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
    
    def start(self, name: Optional[str] = None) -> str:
        """
        Запускает таймер с указанным именем.
        
        Args:
            name: Имя таймера (если None, используется имя из конструктора)
            
        Returns:
            str: Идентификатор таймера для последующего останова
        """
        timer_name = name or self.name or "default"
        timer_id = f"{timer_name}_{int(time.time() * 1000)}"
        
        self.timers[timer_id] = {
            "name": timer_name,
            "start_time": time.time()
        }
        
        if not name:  # Если это основной таймер (из контекстного менеджера)
            self.start_time = self.timers[timer_id]["start_time"]
        
        return timer_id
    
    def stop(self, timer_id: Optional[str] = None) -> float:
        """
        Останавливает таймер и возвращает затраченное время.
        
        Args:
            timer_id: Идентификатор таймера (если None, останавливается основной таймер)
            
        Returns:
            float: Затраченное время в секундах
        """
        if timer_id is None:
            # Случай контекстного менеджера или простого таймера
            elapsed_time = time.time() - self.start_time
            
            if self.name:
                message = f"Время выполнения {self.name}: {elapsed_time:.4f} сек."
                self._log_message(message)
                
                # Сохраняем время в словаре таймингов
                self.timings[self.name] = elapsed_time
            
            return elapsed_time
        
        # Случай отслеживания нескольких таймеров
        if timer_id not in self.timers:
            logger.warning(f"Таймер с ID {timer_id} не найден")
            return 0.0
        
        timer_data = self.timers[timer_id]
        timer_name = timer_data["name"]
        elapsed_time = time.time() - timer_data["start_time"]
        
        # Сохраняем время в словаре таймингов
        self.timings[timer_name] = elapsed_time
        
        # Логируем, если нужно
        message = f"Время выполнения {timer_name}: {elapsed_time:.4f} сек."
        self._log_message(message)
        
        # Удаляем таймер из словаря активных таймеров
        self.timers.pop(timer_id)
        
        return elapsed_time
    
    def _log_message(self, message: str) -> None:
        """
        Логирует сообщение с нужным уровнем логирования.
        
        Args:
            message: Сообщение для логирования
        """
        if self.log_level == "DEBUG":
            logger.debug(message)
        elif self.log_level == "INFO":
            logger.info(message)
        elif self.log_level == "WARNING":
            logger.warning(message)
        else:
            logger.debug(message)
    
    def elapsed(self, timer_id: Optional[str] = None) -> float:
        """
        Возвращает затраченное время без остановки таймера.
        
        Args:
            timer_id: Идентификатор таймера (если None, используется основной таймер)
            
        Returns:
            float: Затраченное время в секундах
        """
        if timer_id is None:
            # Случай контекстного менеджера или простого таймера
            return time.time() - self.start_time
        
        # Случай отслеживания нескольких таймеров
        if timer_id not in self.timers:
            logger.warning(f"Таймер с ID {timer_id} не найден")
            return 0.0
        
        return time.time() - self.timers[timer_id]["start_time"]
    
    def get_timing(self, name: str) -> Optional[float]:
        """
        Возвращает сохраненное время выполнения для указанного имени.
        
        Args:
            name: Имя таймера
            
        Returns:
            Optional[float]: Время выполнения или None, если таймер не найден
        """
        return self.timings.get(name)
    
    def get_all_timings(self) -> Dict[str, float]:
        """
        Возвращает все сохраненные замеры времени.
        
        Returns:
            Dict[str, float]: Словарь с замерами времени
        """
        return self.timings


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Декоратор для замера времени выполнения функции.
    
    Args:
        func (Callable[..., T]): Декорируемая функция
        
    Returns:
        Callable[..., T]: Декорированная функция
    """
    return Timer()(func) 