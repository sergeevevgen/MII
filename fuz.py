a = 2
b = 3
c = 6
d = 10

a1 = 6
b1 = 7.5
c1 = 9.5
d1 = 13

x = [1, 4, 6, 8]
x1 = [3, 5, 7, 10]

for i, j in zip(x, x1):
    f1 = max(min((i - a) / (b - a), 1, (d - i) / (d - c)), 0)
    f2 = max(min((j - a1) / (b1 - a1), 1, (d1 - j) / (d1 - c1)), 0)
    print(min(f1, f2))
