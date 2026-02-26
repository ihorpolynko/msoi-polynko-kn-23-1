import numpy as np

# ====== ДАНІ ======

objects = [
    "Ноутбук",
    "Настільний ПК",
    "Планшет",
    "Смартфон",
    "Принтер"
]

features = [
    "Має батарею",
    "Має сенсорний екран",
    "Має клавіатуру",
    "Портативний",
    "Підтримує мобільну мережу",
    "Друкує документи",
    "Має потужну систему охолодження",
    "Працює від мережі 220В"
]

n = len(objects)
m = len(features)

# випадкова ініціалізація
weights = np.random.uniform(0, 2, size=(n, m))


# ====== ВИВІД ======

def show_objects():
    print("\nДоступні об'єкти:")
    for idx, obj in enumerate(objects):
        print(f"{idx} - {obj}")

# ====== СИГМОЇДА ======

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ====== НОРМАЛИЗАЦІЯ ======

def normalize_weights():
    global weights
    for i in range(n):
        norm = np.linalg.norm(weights[i])
        if norm != 0:
            weights[i] = weights[i] / norm


# ====== РОЗПІЗНАВАННЯ ======

def recognize(answers):
    raw_scores = weights @ answers
    probabilities = sigmoid(raw_scores)
    return np.argmax(probabilities), raw_scores, probabilities


# ====== НАВЧАННЯ ======

def train(correct_index, answers):
    predicted_index, _, _ = recognize(answers)

    if predicted_index != correct_index:
        for i in range(m):
            if answers[i] == 1:
                weights[correct_index][i] += 0.5
                weights[predicted_index][i] -= 0.5

        normalize_weights()

        print("Система навчилась на помилці та нормалізувала ваги.")
    else:
        print("Розпізнавання правильне. Навчання не потрібне.")


# ====== ВВІД ======

def input_features():
    answers = []
    print("\nВведіть ознаки (1 - так, 0 - ні):")
    for f in features:
        val = int(input(f"{f}: "))
        answers.append(val)
    return np.array(answers)


# ====== ГОЛОВНИЙ ЦИКЛ ======

while True:
    print("\n1 - Навчання (ДО / ПІСЛЯ)")
    print("2 - Розпізнавання")
    print("0 - Вихід")

    mode = input("Оберіть режим: ")

    if mode == "1":
        answers = input_features()
        show_objects()

        while True:
            try:
                correct_object = int(input("Введіть номер правильного об'єкта: "))
                if 0 <= correct_object < n:
                    break
                else:
                    print("Невірний номер. Спробуйте ще раз.")
            except ValueError:
                print("Введіть ціле число.")

        print("\n--- ДО НАВЧАННЯ ---")
        pred_before, raw_before, prob_before = recognize(answers)
        print("Передбачення:", objects[pred_before])
        print("Сирі бали:", raw_before)
        print("Ймовірності:", prob_before)

        train(correct_object, answers)

        print("\n--- ПІСЛЯ НАВЧАННЯ ---")
        pred_after, raw_after, prob_after = recognize(answers)
        print("Передбачення:", objects[pred_after])
        print("Сирі бали:", raw_after)
        print("Ймовірності:", prob_after)

    elif mode == "2":
        answers = input_features()
        predicted, raw, prob = recognize(answers)

        print("\nЙмовірний об'єкт:", objects[predicted])
        print("Сирі бали:", raw)
        print("Ймовірності:", prob)

    elif mode == "0":
        break