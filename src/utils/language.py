"""
Модуль для работы с языками текста.

Предоставляет функции для определения языка текста и преобразования
кодов языков в понятные названия.
"""

from typing import Optional, Dict, List, Tuple
import re
from langdetect import detect, DetectorFactory, LangDetectException

# Исправляем импорты для работы при запуске из командной строки
try:
    from .logger import get_logger
    from .result import Result
except ImportError:
    from src.utils.logger import get_logger
    from src.utils.result import Result

# Инициализация фабрики детекторов языка с фиксированным seed для стабильности результатов
DetectorFactory.seed = 0

logger = get_logger(__name__)

# Словарь соответствия кодов языков и их полных названий
LANGUAGE_NAMES: Dict[str, str] = {
    "ru": "Русский",
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "nl": "Nederlands",
    "pl": "Polski",
    "uk": "Українська",
    "be": "Беларуская",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어"
}

# Языки, поддерживаемые системой для полной обработки
SUPPORTED_LANGUAGES: List[str] = ["ru", "en"]


def get_language_name(lang_code: str) -> str:
    """
    Возвращает полное название языка по его коду.
    
    Args:
        lang_code (str): Код языка (например, 'ru', 'en')
        
    Returns:
        str: Полное название языка или исходный код, если название не найдено
    """
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code)


def detect_language(text: str) -> Result[str]:
    """
    Определяет язык текста.
    
    Args:
        text (str): Текст для определения языка
        
    Returns:
        Result[str]: Результат с кодом языка или ошибкой
    """
    if not text or not text.strip():
        return Result.fail("Невозможно определить язык: пустой текст")
    
    # Убираем лишние пробелы, переносы строк и другие символы, которые могут помешать определению
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    # Берем первые 1000 символов для более быстрого определения
    sample = cleaned_text[:1000]
    
    try:
        lang = detect(sample)
        logger.debug(f"Определен язык: {lang} ({get_language_name(lang)})")
        return Result.ok(lang)
    except LangDetectException as e:
        logger.error(f"Ошибка при определении языка: {e}")
        return Result.fail(f"Ошибка при определении языка: {str(e)}")


def is_supported_language(lang_code: str) -> bool:
    """
    Проверяет, поддерживается ли язык системой для полной обработки.
    
    Args:
        lang_code (str): Код языка
        
    Returns:
        bool: True, если язык поддерживается, иначе False
    """
    return lang_code.lower() in SUPPORTED_LANGUAGES


def get_prompt_for_language(lang_code: str, prompt_key: str, prompt_dict: Dict[str, Dict[str, str]]) -> Result[str]:
    """
    Возвращает промпт для указанного языка и ключа.
    
    Args:
        lang_code (str): Код языка
        prompt_key (str): Ключ промпта
        prompt_dict (Dict[str, Dict[str, str]]): Словарь промптов
        
    Returns:
        Result[str]: Результат с текстом промпта или ошибкой
    """
    lang = lang_code.lower()
    
    # Если язык не поддерживается, используем английский
    if not is_supported_language(lang):
        logger.warning(f"Язык {lang} не поддерживается. Используется английский.")
        lang = "en"
    
    # Проверяем наличие промптов для языка
    if lang not in prompt_dict:
        logger.error(f"Промпты для языка {lang} не найдены")
        return Result.fail(f"Промпты для языка {lang} не найдены")
    
    # Проверяем наличие конкретного промпта
    if prompt_key not in prompt_dict[lang]:
        logger.error(f"Промпт '{prompt_key}' для языка {lang} не найден")
        return Result.fail(f"Промпт '{prompt_key}' для языка {lang} не найден")
    
    # Возвращаем промпт
    return Result.ok(prompt_dict[lang][prompt_key])


def get_closest_supported_language(lang_code: str) -> str:
    """
    Возвращает ближайший поддерживаемый язык к указанному.
    
    Для языков, которые не поддерживаются системой полностью, 
    выбирает наиболее подходящий из поддерживаемых языков.
    
    Args:
        lang_code (str): Код языка
        
    Returns:
        str: Код ближайшего поддерживаемого языка
    """
    if is_supported_language(lang_code):
        return lang_code
    
    # Языковые группы с ближайшими соответствиями
    language_groups: Dict[str, List[str]] = {
        "ru": ["uk", "be"],  # Восточнославянские языки
        "en": ["de", "nl", "fr", "es", "it", "pt", "pl"]  # Западные языки
    }
    
    # Поиск ближайшего поддерживаемого языка
    for supported, similar in language_groups.items():
        if lang_code in similar:
            logger.info(f"Язык {lang_code} не поддерживается полностью. Используется {supported}.")
            return supported
    
    # По умолчанию возвращаем английский
    logger.info(f"Язык {lang_code} не поддерживается. Используется английский.")
    return "en" 