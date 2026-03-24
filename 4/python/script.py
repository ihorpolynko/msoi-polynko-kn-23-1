import numpy as np

# ==============================
# 1. Підготовка даних
# ==============================

# Перетворюємо слово в числа (ASCII/Unicode)
def word_to_vector(word, max_len=10):
    vec = [ord(c)/1000 for c in word]  # нормалізація
    while len(vec) < max_len:
        vec.append(0)  # доповнення нулями
    return np.array(vec[:max_len])


# Дані (0 — латиниця, 1 — кирилиця)
data = [
    ("Ihor", 0),
    ("Ігор", 1),
]

# ==============================
# 2. Параметри мережі
# ==============================

input_size = 10
hidden_size = 8
output_size = 1

lr = 0.5
epochs = 2001


# ==============================
# 3. Ініціалізація ваг
# ==============================

W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)


# ==============================
# 4. Функції
# ==============================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)


# ==============================
# 5. Додавання шуму
# ==============================

def add_noise(vec, noise_level=0.1):
    noise = np.random.randn(len(vec)) * noise_level
    return vec + noise


# ==============================
# 6. НАВЧАННЯ
# ==============================

for epoch in range(epochs):

    total_error = 0

    for word, label in data:

        # підготовка входу
        x = word_to_vector(word)

        # додаємо шум
        x = add_noise(x)

        # ===== Прямий прохід =====
        h = sigmoid(np.dot(x, W1))
        y = sigmoid(np.dot(h, W2))

        # ===== Похибка =====
        error = label - y
        total_error += error**2

        # ===== Зворотне розповсюдження =====
        d_output = error * sigmoid_deriv(y)
        d_hidden = d_output.dot(W2.T) * sigmoid_deriv(h)

        # ===== Оновлення ваг =====
        W2 += lr * np.outer(h, d_output)
        W1 += lr * np.outer(x, d_hidden)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Error: {total_error}")


# ==============================
# 7. ТЕСТУВАННЯ
# ==============================

def predict(word, noise=False):
    x = word_to_vector(word)

    if noise:
        x = add_noise(x)

    h = sigmoid(np.dot(x, W1))
    y = sigmoid(np.dot(h, W2))
    return y


print("\n=== ТЕСТ ===")
print("Ihor:", predict("Ihor"))
print("Ігор:", predict("Ігор"))

# тест з шумом
print("\n=== ТЕСТ З ШУМОМ ===")
print("Ihor:", predict("Ihor", True))
print("Ігор:", predict("Ігор", True))