"""
Главный модуль приложения.

Предоставляет точку входа в приложение и основные функции для запуска
различных компонентов системы.
"""

import argparse
import sys
from pathlib import Path

# Исправляем импорты для работы при запуске из командной строки
try:
    from utils import get_logger, initialize_logger, Timer
    from config import app_config, llm_config
except ImportError:
    from src.utils import get_logger, initialize_logger, Timer
    from src.config import app_config, llm_config

logger = get_logger(__name__)


def parse_arguments():
    """
    Разбор аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="Система синтеза графов знаний",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Общие аргументы
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=app_config.log_level,
        help="Уровень логирования"
    )
    
    # Подкоманды
    subparsers = parser.add_subparsers(dest="command", help="Команда для выполнения")
    
    # Команда process
    process_parser = subparsers.add_parser("process", help="Обработка текста")
    process_parser.add_argument(
        "--input",
        required=True,
        help="Путь к входному файлу или директории"
    )
    process_parser.add_argument(
        "--output",
        default=str(app_config.output_dir),
        help="Путь к выходной директории"
    )
    process_parser.add_argument(
        "--language",
        choices=["auto", "ru", "en"],
        default="auto",
        help="Язык текста (auto для автоматического определения)"
    )
    process_parser.add_argument(
        "--provider",
        choices=llm_config.SUPPORTED_PROVIDERS,
        default=llm_config.provider,
        help="LLM провайдер"
    )
    
    # Команда config
    config_parser = subparsers.add_parser("config", help="Управление конфигурацией")
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Показать текущую конфигурацию"
    )
    config_parser.add_argument(
        "--validate",
        action="store_true",
        help="Проверить конфигурацию"
    )
    
    return parser.parse_args()


def process_command(args):
    """
    Обработка команды process.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
    """
    logger.info(f"Запуск обработки текста из {args.input}")
    logger.info(f"Результаты будут сохранены в {args.output}")
    
    # Настройка LLM провайдера
    if args.provider != llm_config.provider:
        llm_config.set_provider(args.provider)
    
    # Проверка API ключа
    if not llm_config.validate_api_key():
        logger.error(f"API ключ для провайдера {llm_config.provider} не найден или недействителен")
        sys.exit(1)
    
    # Проверка входного файла/директории
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Входной файл или директория не существует: {args.input}")
        sys.exit(1)
    
    # Создание выходной директории
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # TODO: Реализовать обработку текста
    logger.info("Обработка текста пока не реализована")


def config_command(args):
    """
    Обработка команды config.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки
    """
    if args.show:
        print("\nКонфигурация приложения:")
        for key, value in app_config.get_config_dict().items():
            print(f"  {key}: {value}")
        
        print("\nКонфигурация LLM:")
        for key, value in llm_config.get_config_dict().items():
            print(f"  {key}: {value}")
    
    if args.validate:
        print("\nПроверка конфигурации:")
        
        # Проверка директорий
        output_dir = Path(app_config.output_dir)
        if not output_dir.exists():
            print(f"  [ПРЕДУПРЕЖДЕНИЕ] Выходная директория не существует: {output_dir}")
            print(f"  Директория будет создана при запуске обработки")
        else:
            print(f"  [OK] Выходная директория: {output_dir}")
        
        # Проверка API ключей
        for provider in llm_config.SUPPORTED_PROVIDERS:
            if provider == "ollama":
                print(f"  [OK] Провайдер {provider} не требует API ключа")
                continue
                
            if llm_config.validate_api_key(provider):
                print(f"  [OK] API ключ для провайдера {provider} найден")
            else:
                print(f"  [ОШИБКА] API ключ для провайдера {provider} не найден")


def main():
    """Основная функция приложения."""
    with Timer("Инициализация приложения", log_level="INFO"):
        # Инициализация логгера
        initialize_logger()
        
        # Разбор аргументов командной строки
        args = parse_arguments()
        
        # Установка уровня логирования
        if hasattr(args, 'log_level') and args.log_level != app_config.log_level:
            app_config.set("log_level", args.log_level)
            # Исправляем импорт для работы при запуске из командной строки
            try:
                from utils import set_log_level
            except ImportError:
                from src.utils import set_log_level
            set_log_level(args.log_level)
    
    # Проверка наличия команды
    if not hasattr(args, 'command') or not args.command:
        print("Ошибка: не указана команда")
        print("Используйте --help для получения справки")
        sys.exit(1)
    
    logger.info(f"Запуск приложения с командой: {args.command}")
    
    # Обработка команд
    if args.command == "process":
        process_command(args)
    elif args.command == "config":
        config_command(args)
    else:
        logger.error(f"Неизвестная команда: {args.command}")
        sys.exit(1)
    
    logger.info("Завершение работы приложения")


if __name__ == "__main__":
    main()
