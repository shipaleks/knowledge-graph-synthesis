"""
Модуль для работы с результатами операций.

Предоставляет класс Result для единообразной обработки результатов и ошибок
в различных частях приложения. Это позволяет избежать использования исключений
для управления потоком программы и делает код более понятным и надежным.
"""

from typing import TypeVar, Generic, Optional, Any, Callable, List
import traceback

# Исправляем импорты для работы при запуске из командной строки
try:
    from .logger import get_logger
except ImportError:
    from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')  # Тип успешного результата


class Result(Generic[T]):
    """
    Класс для обработки результатов операций.
    
    Позволяет единообразно обрабатывать как успешные результаты, 
    так и ошибки без использования исключений.
    
    Attributes:
        value (Optional[T]): Значение успешного результата
        error (Optional[str]): Сообщение об ошибке
        error_details (Optional[Any]): Дополнительные детали ошибки
        success (bool): Флаг успешности операции
    """
    
    def __init__(
        self, 
        value: Optional[T] = None, 
        error: Optional[str] = None,
        error_details: Optional[Any] = None
    ):
        """
        Инициализация объекта Result.
        
        Args:
            value (Optional[T], optional): Значение успешного результата. По умолчанию None.
            error (Optional[str], optional): Сообщение об ошибке. По умолчанию None.
            error_details (Optional[Any], optional): Дополнительные детали ошибки. По умолчанию None.
        """
        self.value = value
        self.error = error
        self.error_details = error_details
        self.success = error is None
    
    @staticmethod
    def ok(value: T) -> 'Result[T]':
        """
        Создает успешный результат с указанным значением.
        
        Args:
            value (T): Значение успешного результата
            
        Returns:
            Result[T]: Объект Result с успешным результатом
        """
        return Result(value=value)
    
    @staticmethod
    def fail(error: str, details: Any = None) -> 'Result[T]':
        """
        Создает результат с ошибкой.
        
        Args:
            error (str): Сообщение об ошибке
            details (Any, optional): Дополнительные детали ошибки. По умолчанию None.
            
        Returns:
            Result[T]: Объект Result с ошибкой
        """
        logger.error(f"{error} {details if details else ''}")
        return Result(error=error, error_details=details)
    
    @staticmethod
    def from_exception(
        exception: Exception, 
        error_message: Optional[str] = None
    ) -> 'Result[T]':
        """
        Создает результат из исключения.
        
        Args:
            exception (Exception): Исключение
            error_message (Optional[str], optional): Дополнительное сообщение об ошибке. 
                По умолчанию None.
            
        Returns:
            Result[T]: Объект Result с ошибкой, основанной на исключении
        """
        message = error_message or str(exception)
        stack_trace = traceback.format_exc()
        logger.error(f"{message}: {exception}\n{stack_trace}")
        return Result(
            error=message,
            error_details={
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'stack_trace': stack_trace
            }
        )
    
    def map(self, func: Callable[[T], Any]) -> 'Result[Any]':
        """
        Применяет функцию к значению успешного результата.
        
        Args:
            func (Callable[[T], Any]): Функция для применения к значению результата
            
        Returns:
            Result[Any]: Новый объект Result с преобразованным значением или исходной ошибкой
        """
        if not self.success:
            return Result(error=self.error, error_details=self.error_details)
        
        try:
            new_value = func(self.value)
            return Result.ok(new_value)
        except Exception as e:
            return Result.from_exception(e, "Ошибка при маппинге результата")
    
    def flat_map(self, func: Callable[[T], 'Result[Any]']) -> 'Result[Any]':
        """
        Применяет функцию, возвращающую Result, к значению успешного результата.
        
        Args:
            func (Callable[[T], Result[Any]]): Функция для применения к значению результата
            
        Returns:
            Result[Any]: Результат применения функции или исходная ошибка
        """
        if not self.success:
            return Result(error=self.error, error_details=self.error_details)
        
        try:
            return func(self.value)
        except Exception as e:
            return Result.from_exception(e, "Ошибка при плоском маппинге результата")
    
    def unwrap(self) -> T:
        """
        Возвращает значение успешного результата или вызывает исключение при ошибке.
        
        Returns:
            T: Значение успешного результата
            
        Raises:
            ValueError: Если результат содержит ошибку
        """
        if not self.success:
            error_message = f"Ошибка при извлечении результата: {self.error}"
            if self.error_details:
                error_message += f" Детали: {self.error_details}"
            logger.error(error_message)
            raise ValueError(error_message)
        
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """
        Возвращает значение успешного результата или значение по умолчанию при ошибке.
        
        Args:
            default (T): Значение по умолчанию
            
        Returns:
            T: Значение успешного результата или значение по умолчанию
        """
        if not self.success:
            logger.warning(f"Используется значение по умолчанию из-за ошибки: {self.error}")
            return default
        
        return self.value
    
    def unwrap_or_else(self, func: Callable[[], T]) -> T:
        """
        Возвращает значение успешного результата или результат вызова функции при ошибке.
        
        Args:
            func (Callable[[], T]): Функция для получения значения по умолчанию
            
        Returns:
            T: Значение успешного результата или результат вызова функции
        """
        if not self.success:
            logger.warning(f"Вычисляется значение по умолчанию из-за ошибки: {self.error}")
            return func()
        
        return self.value
    
    def __str__(self) -> str:
        """
        Возвращает строковое представление объекта Result.
        
        Returns:
            str: Строковое представление
        """
        if self.success:
            return f"Ok({self.value})"
        else:
            details = f", details={self.error_details}" if self.error_details else ""
            return f"Fail({self.error}{details})"
    
    def __repr__(self) -> str:
        """
        Возвращает строковое представление объекта Result для отладки.
        
        Returns:
            str: Строковое представление для отладки
        """
        if self.success:
            return f"Result.ok({repr(self.value)})"
        else:
            return f"Result.fail({repr(self.error)}, {repr(self.error_details)})"


def collect_results(results: List[Result[T]]) -> Result[List[T]]:
    """
    Собирает список результатов в один результат со списком значений.
    
    Если хотя бы один результат содержит ошибку, возвращает первую найденную ошибку.
    
    Args:
        results (List[Result[T]]): Список результатов
        
    Returns:
        Result[List[T]]: Результат со списком значений или первая найденная ошибка
    """
    values = []
    
    for result in results:
        if not result.success:
            return Result(error=result.error, error_details=result.error_details)
        values.append(result.value)
    
    return Result.ok(values) 