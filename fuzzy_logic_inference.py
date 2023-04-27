import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Входные переменные
distance = ctrl.Antecedent(np.arange(0, 1001, 1), 'distance')
size = ctrl.Antecedent(np.arange(0, 11, 1), 'size')
speed = ctrl.Antecedent(np.arange(0, 501, 1), 'speed')

# Выходная переменная
danger = ctrl.Consequent(np.arange(0, 101, 1), 'danger')

# Определение функций принадлежности

distance['close'] = fuzz.trimf(distance.universe, [0, 0, 500])
distance['medium'] = fuzz.trimf(distance.universe, [0, 500, 1000])
distance['far'] = fuzz.trimf(distance.universe, [500, 1000, 1000])


size['small'] = fuzz.trimf(size.universe, [0, 0, 5])
size['medium'] = fuzz.trimf(size.universe, [0, 5, 10])
size['large'] = fuzz.trimf(size.universe, [5, 10, 10])


speed['slow'] = fuzz.trimf(speed.universe, [0, 0, 250])
speed['medium'] = fuzz.trimf(speed.universe, [0, 250, 500])
speed['fast'] = fuzz.trimf(speed.universe, [250, 500, 500])


danger['low'] = fuzz.trimf(danger.universe, [0, 0, 50])
danger['medium'] = fuzz.trimf(danger.universe, [0, 50, 100])
danger['high'] = fuzz.trimf(danger.universe, [50, 100, 100])

distance.view()
size.view()
speed.view()
danger.view()

# Определение правил
rule1 = ctrl.Rule(distance['close'] & size['small'] & speed['slow'], danger['high'])
rule2 = ctrl.Rule(distance['close'] & size['small'] & speed['medium'], danger['medium'])
rule3 = ctrl.Rule(distance['close'] & size['small'] & speed['fast'], danger['low'])
rule4 = ctrl.Rule(distance['close'] & size['medium'] & speed['slow'], danger['high'])
rule5 = ctrl.Rule(distance['close'] & size['medium'] & speed['medium'], danger['medium'])
rule6 = ctrl.Rule(distance['close'] & size['medium'] & speed['fast'], danger['low'])
rule7 = ctrl.Rule(distance['close'] & size['large'] & speed['slow'], danger['high'])
rule8 = ctrl.Rule(distance['close'] & size['large'] & speed['medium'], danger['high'])
rule9 = ctrl.Rule(distance['close'] & size['large'] & speed['fast'], danger['medium'])
rule10 = ctrl.Rule(distance['medium'] & size['small'] & speed['slow'], danger['medium'])

rule6.view()

# Создание системы нечеткого вывода
danger_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
danger_simulation = ctrl.ControlSystemSimulation(danger_ctrl)

# Установка значений входных переменных
Distance = 100
Size = 10
Speed = 350

danger_simulation.input['distance'] = Distance
danger_simulation.input['size'] = Size
danger_simulation.input['speed'] = Speed

# Выполнение расчетов
danger_simulation.compute()

# Получение значения выходной переменной
Danger = danger_simulation.output['danger']

danger.view(sim=danger_simulation)

print('---Входные данные---')
print(f'\nРасстояние до земли:  {Distance} км\n')
print(f'Размер небесного тела:  {Size} м\n')
print(f'Скорость движения к земле:  {Speed} км/ч\n')
print('---Выходные данные---')
print(f'\nСтепень опасности:  {round(Danger,2)}/100')

plt.show()