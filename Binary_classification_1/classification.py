import matplotlib.pyplot as plt
from data_generate import *
from math import *

X, Y = generate_two_class_data(n_samples=100,noise=0.5,random_seed=42)

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', alpha=0.7)
plt.xlabel('Признак x1')
plt.ylabel('Признак x2')
plt.title('Сгенерированные данные')
plt.colorbar(label='Класс')
plt.grid(True)
# plt.show()

w = [-1, -1]
a = lambda x: np.sign(x[0]*w[0] + x[1]*w[1])
N = 50
L = 0.1
e = 0.1

last_error_index = -1

for n in range(50):
    for i in range(len(X)):
        if Y[i] * a(X[i]) < 0:
            w[0] = w[0]+L*Y[i]
            last_error_index = i
    Q = sum([1 for i in range(len(X)) if Y[i]*a(X[i]) < 0])
    if Q == 0:
        break

if last_error_index > -1:
    w[0] = w[0] + e * Y[last_error_index]
print(X)
line_x = list(range(int(min(X[:,0])),int(max(X[:,0]))))
line_y = [w[0]*x for x in line_x]
plt.plot(line_x, line_y, color='green')
plt.xlim([-45,45])
plt.ylim([-75,75])
plt.xlabel('длина')
plt.ylabel('ширина')
plt.grid(True)
plt.show()
