import re

def count_syllables_ru(text: str) -> int:
    """Подсчет слогов в русском тексте по количеству гласных"""
    vowels = "аеёиоуыэюя"
    return len([char for char in text.lower() if char in vowels])

def count_phonemes_ru(text: str) -> int:
    """Подсчет букв (как упрощенная модель фонем)"""
    # Удаляем знаки препинания и пробелы
    clean_text = re.sub(r'[^а-яёa-z0-9]', '', text.lower())
    return len(clean_text)
