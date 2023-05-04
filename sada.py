from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# Необходимый импорт
import numpy as np

model = Sequential()  # создаем модель

# добавляем 1ый слой с 3мя нейронами и функцией активации ReLU
model.add(Dense(3, input_dim=3, activation='relu'))

# добавляем 2ой слой с 4мя нейронами и функцией активации ReLU
model.add(Dense(4, activation='relu'))

# добавляем 3ий слой с 1 нейроном и функцией активации sigmoid
model.add(Dense(1, activation='sigmoid'))
X = np.array([[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0],
              [1, 1, 1]])
print(model.predict(X))

