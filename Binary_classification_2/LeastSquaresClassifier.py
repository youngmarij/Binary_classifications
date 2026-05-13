import numpy as np
import matplotlib.pyplot as plt
from project1.binary_classification import x_train, y_train

x_train = np.hstack((x_train, np.ones((len(x_train), 1))))
pt = np.sum([x*y for x,y in zip(x_train, y_train)], axis = 0)
xxt = np.sum([np.outer(x, x) for x in x_train], axis = 0)
w = np.dot(pt, np.linalg.inv(xxt))
# print(w)

line_x = list(range(int(max(x_train[:, 0]))+1)) # формирование графика разделяющей линии
line_y = [-x*w[0]/w[1] - w[2]/w[1] for x in line_x]

x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

plt.scatter(x_0[:, 0], x_0[:, 1], color = 'red')
plt.scatter(x_1[:, 0], x_1[:, 1], color = 'blue')
plt.plot(line_x, line_y, color = 'green')
