import os

def check_missing_labels(images_dir, labels_dir):
    """Проверяет отсутствующие файлы разметки для изображений"""
    missing = []
    
    # Получаем список файлов без расширений
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}
    
    # Находим изображения без разметки
    missing = image_files - label_files
    
    if missing:
        print(f"Найдено {len(missing)} изображений без разметки:")
        for name in sorted(missing):
            # Выводим имя с оригинальным расширением (находим первое подходящее)
            for ext in ['.jpg', '.png', '.jpeg']:
                if os.path.exists(os.path.join(images_dir, name + ext)):
                    print(f"  - {name}{ext}")
                    break
    else:
        print("Все изображения имеют соответствующие файлы разметки!")
    
    return missing

# Пример использования
dataset_base = "food_dataset"  # Путь к корневой папке датасета

for subset in ['train', 'val', 'test']:
    images_dir = os.path.join(dataset_base, subset, 'images')
    labels_dir = os.path.join(dataset_base, subset, 'labels')
    
    print(f"\nПроверка {subset}...")
    if not os.path.exists(images_dir):
        print(f"Папка {images_dir} не существует!")
        continue
    if not os.path.exists(labels_dir):
        print(f"Папка {labels_dir} не существует!")
        continue
    
    check_missing_labels(images_dir, labels_dir)