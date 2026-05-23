import numpy as np
from PIL import Image
import os

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ЗОБРАЖЕНЬ
# ==========================================

IMAGE_FOLDER = "images"

# ==========================================
# 2. ФУНКЦІЯ ПЕРЕТВОРЕННЯ JPEG -> ВЕКТОР
# ==========================================

def image_to_vector(path):

    # відкриваємо зображення
    img = Image.open(path)

    # переводимо у відтінки сірого
    img = img.convert("L")

    # змінюємо розмір до 8x8
    img = img.resize((8, 8))

    # перетворення у масив numpy
    arr = np.array(img)

    # бінаризація:
    # світлі пікселі -> 1
    # темні пікселі -> -1
    arr = np.where(arr > 127, 1, -1)

    # перетворення матриці 8x8 у вектор 64
    vector = arr.flatten()

    return vector

# ==========================================
# 3. ПОРЯДОК ЛІТЕР
# ==========================================

letter_order = [
    "bukva-п",
    "bukva-о",
    "bukva-л",
    "bukva-и",
    "bukva-н",
    "bukva-ь",
    "bukva-к",
    "bukva-о-eng",
    "bukva-к-eng"
]

# ==========================================
# 4. ЗАВАНТАЖЕННЯ ВСІХ ЛІТЕР
# ==========================================

letters = {}

for filename in os.listdir(IMAGE_FOLDER):

    if filename.endswith(".jpg") or filename.endswith(".jpeg"):

        # назва файлу без .jpg
        letter_name = os.path.splitext(filename)[0]

        path = os.path.join(IMAGE_FOLDER, filename)

        letters[letter_name] = image_to_vector(path)

# сортування у потрібному порядку
letters = {key: letters[key] for key in letter_order}

# ==========================================
# 5. ПАРАМЕТРИ МЕРЕЖІ
# ==========================================

# довжина вектора
n = len(next(iter(letters.values())))

# кількість еталонів
m = len(letters)

# ==========================================
# 6. ІНІЦІАЛІЗАЦІЯ МЕРЕЖІ ХЕММІНГА
# ==========================================

# ваги першого шару
W1 = np.array([letters[l] for l in letters]).T

# поріг
T = n / 2

# коефіцієнт пригнічення
epsilon = 0.1 / m

# другий шар (конкурентний)
W2 = np.ones((m, m)) - np.eye(m)

# ==========================================
# 7. ФУНКЦІЯ АКТИВАЦІЇ
# ==========================================

def ramp(x):
    return np.where(x > 0, x, 0)

# ==========================================
# 8. РОЗПІЗНАВАННЯ
# ==========================================

def predict(X_input):

    # ======================================
    # ПЕРШИЙ ШАР
    # ======================================

    # обчислення схожості з еталонами
    y1 = np.dot(X_input, W1) + T

    # ======================================
    # ДРУГИЙ ШАР (MAXNET)
    # ======================================

    y2 = y1.copy()

    for _ in range(100):

        y_prev = y2.copy()

        # конкурентне пригнічення
        total_sum = np.sum(y_prev)

        y2 = ramp(
            y_prev - epsilon * (total_sum - y_prev)
        )

        # якщо залишився один переможець
        if np.count_nonzero(y2) <= 1:
            break

    return y2

# ==========================================
# 9. ТЕСТУВАННЯ
# ==========================================

print("=== ТЕСТУВАННЯ ===\n")

for letter, vec in letters.items():

    print(f"Літера: {letter}")

    # ======================================
    # ТЕСТ БЕЗ ШУМУ
    # ======================================

    out = predict(vec)

    winner_index = np.argmax(out)

    winner_letter = list(letters.keys())[winner_index]

    print("Розпізнано як:", winner_letter)

    print(
        "Активації:",
        np.round(out, 2).tolist()
    )

    # ======================================
    # ТЕСТ ІЗ ШУМОМ
    # ======================================

    noisy_vec = vec.copy()

    # додаємо шум
    noisy_vec[0] *= -1
    noisy_vec[5] *= -1
    noisy_vec[10] *= -1

    out_noisy = predict(noisy_vec)

    winner_noisy = np.argmax(out_noisy)

    winner_letter_noisy = list(letters.keys())[winner_noisy]

    print(
        "Розпізнано із шумом як:",
        winner_letter_noisy
    )

    print(
        "Активації із шумом:",
        np.round(out_noisy, 2).tolist()
    )