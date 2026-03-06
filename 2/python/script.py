import numpy as np

# ==============================
# 1. Вихідні дані
# ==============================

# Таблица координат (5 точек, 3 признака)
points = np.array([
    [12, 5, 8],
    [3, 8, 9],
    [6, 7, 4],
    [13, 9, 14],
    [8, 15, 5]
])

# Нова точка для розпізнавання
new_point = np.array([10, 6, 7])

# Вагові коефіцієнти (для взваженої метрики)
weights = np.array([1, 1, 1])


# ==============================
# 2. Метрики відстаней
# ==============================

# Евклідова
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Квадрат Евклідової
def euclidean_squared(a, b):
    return np.sum((a - b) ** 2)

# Манхеттенська (метрика городських кварталів)
def manhattan(a, b):
    return np.sum(np.abs(a - b))

# Чебишева
def chebyshev(a, b):
    return np.max(np.abs(a - b))

# Зважена Евклідова
def weighted_euclidean(a, b, weights):
    return np.sqrt(np.sum(weights * (a - b) ** 2))


# ==============================
# 3. Побудова матриці відстаней
# ==============================

def distance_matrix(points, metric):
    n = len(points)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = metric(points[i], points[j])
            matrix[i][j] = d
            matrix[j][i] = d

    return matrix


# ==============================
# 4. Відстані до нової точки
# ==============================

def distances_to_new(points, new_point, metric):
    distances = []
    for p in points:
        distances.append(metric(p, new_point))
    return np.array(distances)


# ==============================
# 5. Обчислення вірогідностей
# ==============================

def probabilities(distances):
    # Щоб уникнути ділення на нуль
    distances = np.where(distances == 0, 1e-10, distances)

    inverse = 1 / distances
    total = np.sum(inverse)
    probs = inverse / total
    return probs


# ==============================
# Виведення таблиці точок
# ==============================


def print_points(points):
    print("=== Таблиця координат точок у просторі ознак ===")
    print("№\tx1\tx2\tx3")

    for i, p in enumerate(points, start=1):
        print(f"{i}\t{p[0]}\t{p[1]}\t{p[2]}")

# ==============================
# 6. Основна програма
# ==============================

print_points(points)

print("=== Матриця відстаней (Евклід) ===")
print(distance_matrix(points, euclidean))

print("\n=== Матриця відстаней (Манхеттен) ===")
print(distance_matrix(points, manhattan))

print("\n=== Матриця відстаней (Чебишев) ===")
print(distance_matrix(points, chebyshev))


# Відстані до нової точки
dist_euclid = distances_to_new(points, new_point, euclidean)
dist_manhattan = distances_to_new(points, new_point, manhattan)
dist_chebyshev = distances_to_new(points, new_point, chebyshev)
dist_weighted = distances_to_new(points, new_point,
                                 lambda a, b: weighted_euclidean(a, b, weights))

print("\n=== Відстань до нової точки ===")
print("Евклід:", dist_euclid)
print("Манхеттен:", dist_manhattan)
print("Чебишев:", dist_chebyshev)
print("Зважена Евклідова:", dist_weighted)


# Вірогідності
print("\n=== Вірогідності принадлежності ===")
print("Евклід:", probabilities(dist_euclid))
print("Манхеттен:", probabilities(dist_manhattan))
print("Чебишев:", probabilities(dist_chebyshev))
print("Зважена Евклідова:", probabilities(dist_weighted))


# Визначення класу
max_index = np.argmax(probabilities(dist_euclid))
print(f"\nНова точка належить до класу № {max_index + 1} (по Евклідовій метриці)")