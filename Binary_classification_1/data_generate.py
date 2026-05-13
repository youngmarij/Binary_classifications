import random
import numpy as np
def generate_two_class_data(n_samples=100, noise=0.5, random_seed=42):
    np.random.seed(random_seed)
    class_1 = np.random.multivariate_normal([2,2],[[noise,0],[0,noise]],n_samples)
    class_2 = np.random.multivariate_normal([-2, -2], [[noise, 0], [0, noise]], n_samples)
    y1 = np.zeros(n_samples)
    y2 = np.ones(n_samples)
    classes_x = np.vstack([class_1,class_2])
    classes_y = np.hstack([y1,y2])
    indices = np.random.permutation(len(classes_y))
    X = classes_x[indices]
    Y = classes_y[indices]
    return X,Y
