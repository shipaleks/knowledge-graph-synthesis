"""
Модуль с вспомогательными функциями для работы с текстом.

Предоставляет функции для обработки, нормализации и
фрагментации текста.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

# Исправляем импорты для работы при запуске из командной строки
try:
    from .logger import get_logger
    from .result import Result
except ImportError:
    from src.utils.logger import get_logger
    from src.utils.result import Result

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """
    Нормализует текст: удаляет лишние пробелы, нормализует переносы строк и т.д.
    
    Args:
        text (str): Исходный текст
        
    Returns:
        str: Нормализованный текст
    """
    if not text:
        return ""
    
    # Заменяем различные типы переносов строк на стандартный \n
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Заменяем множественные переносы строк на одинарные
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Удаляем лишние пробелы в строках
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Удаляем пробелы в начале и конце строк
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Удаляем пробелы в начале и конце текста
    text = text.strip()
    
    return text


def split_into_paragraphs(text: str) -> List[str]:
    """
    Разбивает текст на абзацы.
    
    Args:
        text (str): Исходный текст
        
    Returns:
        List[str]: Список абзацев
    """
    if not text:
        return []
    
    # Нормализуем текст
    normalized = normalize_text(text)
    
    # Разбиваем по двойным переносам строк
    paragraphs = re.split(r'\n\n+', normalized)
    
    # Удаляем пустые абзацы
    paragraphs = [p for p in paragraphs if p.strip()]
    
    return paragraphs


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Обрезает текст до указанной длины, сохраняя целостность слов.
    
    Args:
        text (str): Исходный текст
        max_length (int): Максимальная длина
        add_ellipsis (bool, optional): Добавлять ли многоточие в конце. По умолчанию True.
        
    Returns:
        str: Обрезанный текст
    """
    if not text or len(text) <= max_length:
        return text
    
    # Находим последний пробел перед максимальной длиной
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    # Добавляем многоточие при необходимости
    if add_ellipsis:
        truncated += "..."
    
    return truncated


def count_tokens_approx(text: str, chars_per_token: int = 4) -> int:
    """
    Приблизительно оценивает количество токенов в тексте.
    
    Args:
        text (str): Исходный текст
        chars_per_token (int, optional): Примерное количество символов на токен. По умолчанию 4.
        
    Returns:
        int: Примерное количество токенов
    """
    if not text:
        return 0
    
    # Убираем пробелы для более точной оценки
    text = text.strip()
    
    # Возвращаем приблизительную оценку
    return max(1, len(text) // chars_per_token)


def extract_segments(text: str, max_segment_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Разбивает текст на смысловые сегменты.
    
    Args:
        text (str): Исходный текст
        max_segment_length (int, optional): Максимальная длина сегмента. По умолчанию 1000.
        
    Returns:
        List[Dict[str, Any]]: Список сегментов с метаданными
    """
    if not text:
        return []
    
    # Нормализуем текст
    normalized = normalize_text(text)
    
    # Разбиваем на абзацы
    paragraphs = split_into_paragraphs(normalized)
    
    # Группируем абзацы в сегменты
    segments = []
    current_segment = ""
    current_position = 0
    segment_id = 1
    
    for paragraph in paragraphs:
        # Если добавление абзаца превысит максимальную длину, создаем новый сегмент
        if len(current_segment) + len(paragraph) + 2 > max_segment_length and current_segment:
            segments.append({
                "id": f"seg_{segment_id}",
                "text": current_segment,
                "position": current_position,
                "length": len(current_segment)
            })
            segment_id += 1
            current_segment = paragraph
            current_position += len(current_segment) + 2  # +2 для учета переноса строки
        else:
            # Добавляем абзац к текущему сегменту
            if current_segment:
                current_segment += "\n\n" + paragraph
            else:
                current_segment = paragraph
    
    # Добавляем последний сегмент
    if current_segment:
        segments.append({
            "id": f"seg_{segment_id}",
            "text": current_segment,
            "position": current_position,
            "length": len(current_segment)
        })
    
    return segments


def find_best_split_point(text: str, target_length: int) -> int:
    """
    Находит оптимальную точку для разделения текста.
    
    Ищет конец предложения или, если не найден, конец слова.
    
    Args:
        text (str): Текст для разделения
        target_length (int): Целевая длина, около которой нужно найти точку разделения
        
    Returns:
        int: Позиция для разделения текста
    """
    if len(text) <= target_length:
        return len(text)
    
    # Ограничиваем поиск областью вокруг целевой длины
    search_start = max(0, target_length - 100)
    search_end = min(len(text), target_length + 100)
    search_text = text[search_start:search_end]
    
    # Ищем ближайший конец предложения
    sentence_endings = list(re.finditer(r'[.!?]\s+', search_text))
    if sentence_endings:
        # Находим ближайший к целевой длине конец предложения
        closest_idx = min(range(len(sentence_endings)), 
                          key=lambda i: abs((sentence_endings[i].end() + search_start) - target_length))
        return sentence_endings[closest_idx].end() + search_start
    
    # Если не нашли конец предложения, ищем конец слова
    word_endings = list(re.finditer(r'\s+', search_text))
    if word_endings:
        closest_idx = min(range(len(word_endings)), 
                          key=lambda i: abs((word_endings[i].end() + search_start) - target_length))
        return word_endings[closest_idx].end() + search_start
    
    # Если не нашли ни конца предложения, ни конца слова, просто используем целевую длину
    return target_length


def strip_markdown(text: str) -> str:
    """
    Удаляет разметку Markdown из текста.
    
    Args:
        text (str): Текст с разметкой Markdown
        
    Returns:
        str: Текст без разметки Markdown
    """
    if not text:
        return ""
    
    # Удаляем заголовки
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Удаляем выделение жирным, курсивом и т.д.
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Удаляем блоки кода
    text = re.sub(r'```(?:[^`]*)\n([\s\S]*?)\n```', r'\1', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    
    # Удаляем ссылки, оставляя текст
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Удаляем изображения
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Удаляем горизонтальные линии
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    
    # Удаляем элементы списков
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Нормализуем пробелы и переносы строк
    return normalize_text(text)


def is_natural_language_text(text: str) -> bool:
    """
    Определяет, является ли текст естественным языком, а не кодом или структурированными данными.
    
    Args:
        text (str): Текст для проверки
        
    Returns:
        bool: True, если текст похож на естественный язык
    """
    if not text or len(text) < 10:
        return False
    
    # Очищаем текст от пробелов и пунктуации для анализа
    cleaned = re.sub(r'[\s\.,;:!?\'"()\[\]{}]', '', text)
    
    # Проверяем на код или JSON по характерным символам
    code_markers = ['{}', '[]', ';', '==', '=', '//', '/*', '*/', 'function', 'def ', 'class ']
    for marker in code_markers:
        if marker in text:
            return False
    
    # Проверяем на высокую долю специальных символов
    special_chars = sum(1 for c in cleaned if not c.isalnum())
    if len(cleaned) > 0 and special_chars / len(cleaned) > 0.3:
        return False
    
    # Проверяем на наличие предложений с точками или другими знаками препинания
    has_sentences = re.search(r'[A-ZА-Я].*?[.!?]', text) is not None
    
    # Проверяем среднюю длину слов (естественный язык обычно имеет среднюю длину 4-8 символов)
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return False
    
    avg_word_length = sum(len(word) for word in words) / len(words)
    natural_word_length = 3 <= avg_word_length <= 10
    
    return has_sentences and natural_word_length