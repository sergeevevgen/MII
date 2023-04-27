import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# Входные переменные
# Генерируем последовательности с (1 - начальное значение, 2 - конечное, 3 - шаг, 4 - название)
distance = ctrl.Antecedent(np.arange(0, 121, 1), 'distance')
speed = ctrl.Antecedent(np.arange(0, 110, 1), 'speed')


# Выходная переменная
# Генерируем последовательности с (1 - начальное значение, 2 - конечное, 3 - шаг, 4 - название)
volume = ctrl.Consequent(np.arange(0, 11, 1), 'volume')

# Определение функций принадлежности
distance['short'] = fuzz.trimf(distance.universe, [0, 5, 20])
distance['middle'] = fuzz.trimf(distance.universe, [15, 30, 70])
distance['long'] = fuzz.trimf(distance.universe, [40, 80, 120])
distance['far_far'] = fuzz.trimf(distance.universe, [100, 110, 200])


speed['slow'] = fuzz.trimf(speed.universe, [0, 10, 25])
speed['medium'] = fuzz.trimf(speed.universe, [15, 29, 50])
speed['fast'] = fuzz.trimf(speed.universe, [45, 65, 109])


volume['low'] = fuzz.trimf(volume.universe, [0, 4, 6])
volume['medium'] = fuzz.trimf(volume.universe, [3, 5, 8])
volume['high'] = fuzz.trimf(volume.universe, [6, 9, 10])

distance.view()
speed.view()
volume.view()

# Формирование базы правил
rule_1 = ctrl.Rule(distance['short'] & speed['slow'], volume['high'])
rule_2 = ctrl.Rule(distance['short'] & speed['medium'], volume['medium'])
rule_3 = ctrl.Rule(distance['short'] & speed['fast'], volume['high'])
rule_4 = ctrl.Rule(distance['middle'] & speed['slow'], volume['high'])
rule_5 = ctrl.Rule(distance['middle'] & speed['medium'], volume['low'])
rule_6 = ctrl.Rule(distance['middle'] & speed['fast'], volume['medium'])
rule_7 = ctrl.Rule(distance['long'] & speed['slow'], volume['high'])
rule_8 = ctrl.Rule(distance['long'] & speed['medium'], volume['medium'])
rule_9 = ctrl.Rule(distance['long'] & speed['fast'], volume['low'])
rule_10 = ctrl.Rule(distance['far_far'] & speed['fast'], volume['low'])

# rule_6.view()

# Создание системы нечеткого вывода
volume_ctrl = ctrl.ControlSystem([rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9, rule_10])
volume_simulation = ctrl.ControlSystemSimulation(volume_ctrl)

# Установка значений входных переменных
Distance = 100
Speed = 40

volume_simulation.input['distance'] = Distance
volume_simulation.input['speed'] = Speed

# Аккумулирование заключений
volume_simulation.compute()

# Дефаззификация выходной переменной
Volume = volume_simulation.output['volume']

volume.view(sim=volume_simulation)

print(f'Расстояние для преодоления:  {Distance} км')
print(f'Скорость:  {Speed} км/ч')
print(f'Объем топлива:  {round(Volume, 2)} л / км/ч ')

plt.show()
