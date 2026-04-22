import numpy as np  # підключаємо бібліотеку для роботи з числами та масивами

# ==============================
# 1. Опис термів (назви)
# ==============================

# Для кожної універсальної множини (A, B, C) задаємо назви термів
terms_A = ["A1", "A2", "A3"]
terms_B = ["B1", "B2", "B3"]
terms_C = ["C1", "C2", "C3"]

# ==============================
# 2. Параметри функцій
# ==============================

# --- Для A (експоненціальні функції) ---
a_A = [10, 20, 30]
b_A = [0.5, 0.4, 0.5]

# --- Для B ---
c_B = [160, None, None]
b_B = [175, 210, None]
a_B = [None, 154, 200]
c2 = 190
c3 = 215

# --- Для C ---
a_C = [0.1, 75, 0.09]
b_C = [1, 134, 1.5]
c_C = [40, 90, 150]
d_C = [None, 110, None]

# ==============================
# 3. Функції приналежності
# ==============================

# --- Множина A ---
def mu_A1(x):
    if x < a_A[0]:
        return 1
    else:
        return np.exp(-abs((x - a_A[0]) / (2 * b_A[0])))

def mu_A2(x):
    return np.exp(-abs((x - a_A[1]) / (2 * b_A[1])))

def mu_A3(x):
    if x > a_A[2]:
        return 1
    else:
        return np.exp(-abs((x - a_A[2]) / (2 * b_A[2])))

# --- Множина B ---
def mu_B1(x):
    if x < c_B[0]:
        return 1
    elif c_B[0] < x < b_B[0]:
        return (b_B[0] - x) / (b_B[0] - c_B[0])
    else:
        return 0

def mu_B2(x):
    if x <= a_B[1]:
        return 0
    elif a_B[1] < x < c2:
        return (x - a_B[1]) / (c2 - a_B[1])
    elif c2 <= x < b_B[1]:
        return 1
    else:
        return 0

def mu_B3(x):
    if x <= a_B[2]:
        return 0
    elif a_B[2] < x < c3:
        return (x - a_B[2]) / (c3 - a_B[2])
    else:
        return 1

# --- Множина C ---
def mu_C1(x):
    if x < c_C[0]:
        return 1
    else:
        return 1 / (1 + (a_C[0] * (x - c_C[0])) ** b_C[0])

def mu_C2(x):
    if x <= a_C[1]:
        return 0
    elif a_C[1] < x < c_C[1]:
        return (x - a_C[1]) / (c_C[1] - a_C[1])
    elif c_C[1] <= x <= d_C[1]:
        return 1
    elif d_C[1] < x < b_C[1]:
        return (b_C[1] - x) / (b_C[1] - d_C[1])
    else:
        return 0

def mu_C3(x):
    if x >= c_C[2]:
        return 1
    else:
        return 1 / (1 + (a_C[2] * (c_C[2] - x)) ** b_C[2])

# ==============================
# 4. Ввід значень
# ==============================

x_A = float(input("Введіть значення для A: "))
x_B = float(input("Введіть значення для B: "))
x_C = float(input("Введіть значення для C: "))

# ==============================
# 5. Обчислення
# ==============================

# Обчислюємо значення функцій приналежності

AF_A = [mu_A1(x_A), mu_A2(x_A), mu_A3(x_A)]
AF_B = [mu_B1(x_B), mu_B2(x_B), mu_B3(x_B)]
AF_C = [mu_C1(x_C), mu_C2(x_C), mu_C3(x_C)]

# ==============================
# 6. Виведення результатів
# ==============================

print("\n=== РЕЗУЛЬТАТИ ФАЗЗИФІКАЦІЇ ===")

print("\nМножина A:")
for i in range(3):
    print(f"{terms_A[i]} = {AF_A[i]}")

print("\nМножина B:")
for i in range(3):
    print(f"{terms_B[i]} = {AF_B[i]}")

print("\nМножина C:")
for i in range(3):
    print(f"{terms_C[i]} = {AF_C[i]}")