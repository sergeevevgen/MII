import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import numpy as np

# 3 кластера для разделения женщин по росту и возрасту
n = 10
n_clusters = 3
error = 0.05
iters = 250
w = 2

arr = []
xy = []
for i in range(n):
    h = random.randint(130, 195)
    age = random.randint(17, 50)
    # fst = random.uniform(0, 0.33)
    # snd = random.uniform(0, 0.33 - fst)
    # thd = 1 - (fst + snd)
    # arr.append([h, age, fst, snd, thd])
    xy.append([h, age])

print(xy)
data = np.array(xy)

# Применение нечеткой кластеризации, используя метод С-средних, к нашим данным с помощью библиотеки skfuzzy
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T,
    n_clusters,
    2,
    error=error,
    maxiter=iters,
    init=None)

cluster_membership = np.argmax(u, axis=0)

# Вывод центров кластеров
for i, j in zip(cntr, range(n_clusters)):
    print(f'Центр кластера {j + 1}): {i}')

for i, j, k in zip(cluster_membership, range(n), xy):
    print(f'Элемент {j + 1}: {k} принадлежит к кластеру #{i + 1}')
