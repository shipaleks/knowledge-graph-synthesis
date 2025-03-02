"""
Модуль утилит приложения.

Содержит вспомогательные функции и классы, используемые во всем приложении.
"""

from .logger import get_logger, set_log_level, initialize_logger, configure_logging
from .result import Result, collect_results
from .language import (
    detect_language, get_language_name, is_supported_language,
    get_prompt_for_language, get_closest_supported_language
)
from .text_utils import (
    normalize_text, split_into_paragraphs, truncate_text,
    count_tokens_approx, extract_segments, find_best_split_point,
    strip_markdown, is_natural_language_text
)
from .timer import Timer, timed

__all__ = [
    # Логирование
    'get_logger', 'set_log_level', 'initialize_logger', 'configure_logging',
    
    # Обработка результатов
    'Result', 'collect_results',
    
    # Языки
    'detect_language', 'get_language_name', 'is_supported_language',
    'get_prompt_for_language', 'get_closest_supported_language',
    
    # Текст
    'normalize_text', 'split_into_paragraphs', 'truncate_text',
    'count_tokens_approx', 'extract_segments', 'find_best_split_point',
    'strip_markdown', 'is_natural_language_text',
    
    # Таймер
    'Timer', 'timed'
]
