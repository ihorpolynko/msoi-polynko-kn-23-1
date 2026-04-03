import numpy as np

# ========== Вхідні дані ==========
# Приклад: літери прізвища у вигляді бінарних векторів (+1/-1)
letters = {
    # Латиниця
    "P": np.array([1, -1, 1, -1, 1]),
    "O": np.array([-1, 1, 1, -1, -1]),
    "L": np.array([1, 1, -1, -1, 1]),
    "Y": np.array([-1, -1, 1, 1, -1]),
    "N": np.array([1, -1, -1, 1, 1]),
    "K": np.array([-1, 1, -1, 1, 1]),

    # Кирилиця
    "П": np.array([1, -1, 1, -1, 1]),  # аналог P
    "О": np.array([-1, 1, 1, -1, -1]),  # аналог O
    "Л": np.array([1, 1, -1, -1, 1]),  # аналог L
    "И": np.array([-1, -1, 1, 1, -1]),  # аналог Y
    "Н": np.array([1, -1, -1, 1, 1]),  # аналог N
    "К": np.array([-1, 1, -1, 1, 1])  # аналог K
}

# Кількість нейронів = довжина векторів
n = len(next(iter(letters.values())))
m = len(letters)

# ========== Ініціалізація мережі ==========
# Перший шар
W1 = np.array([2 * letters[l] for l in letters]).T  # n x m
T = n / 2  # поріг

# Другий шар
epsilon = 0.1 / m
W2 = np.ones((m, m)) - np.eye(m) * 1  # інгібіторні зв'язки

# ========== Функція активації ==========
def step(x):
    return np.where(x >= 0, 1, -1)


# ========== Функція розпізнавання ==========
def predict(X_input):
    # Перший шар
    y1 = step(np.dot(X_input, W1) - T)

    # Другий шар
    y2 = y1.copy()
    changed = True
    while changed:
        y_prev = y2.copy()
        s = y2 - epsilon * np.dot(W2, y2)
        y2 = step(s)
        changed = not np.array_equal(y2, y_prev)
    return y2

# ========== Тестування ==========
for letter, vec in letters.items():
    # Тест без шуму
    out = predict(vec)
    print(f"{letter}: {out}")

    # Тест з шумом
    noisy_vec = vec.copy()
    noisy_vec[0] *= -1  # наприклад, змінимо один біт
    out_noisy = predict(noisy_vec)
    print(f"{letter} (noisy): {out_noisy}")