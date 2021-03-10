import numpy as np
import matplotlib.pyplot as plt

def central_diff(x, y, dx):
        diff = (y[2:] - y[:-2]) / (2*dx)
        return x[1:-1], diff

def central_2_diff(x, y, dx):
        diff = (y[:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:]) / (12*dx)
        return x[2:-2], diff

def central_3_diff(x, y, dx):
        diff = (-y[:-6] + 9*y[1:-5] - 45*y[2:-4] + 45*y[4:-2] - 9*y[5:-1] + y[6:]) / (60*dx)
        return x[3:-3], diff

def forward_diff(x, y, dx):
        diff = (y[1:] - y[:-1]) / dx
        return (x[:-1] + x[1:]) / 2, diff

dx = 0.1
x = np.arange(-4, 4, dx)
y = np.sin(3 * x**2)

exact_x = np.arange(-4, 4, 0.0001)
exact_y = np.sin(3 * exact_x**2)
exact_diff_y = 6*exact_x * np.cos(3 * exact_x**2)

central_x, central_diff_y = central_diff(x, y, dx)
central_2_x, central_2_diff_y = central_2_diff(x, y, dx)
central_3_x, central_3_diff_y = central_3_diff(x, y, dx)
forward_x, forward_diff_y = forward_diff(x, y, dx)

plt.plot(exact_x, exact_y, label='y', linewidth=0.5)
plt.plot(exact_x, exact_diff_y, label="y', exact", linewidth=0.5)
plt.scatter(central_x, central_diff_y, label="y', central differences", s=1)
plt.scatter(central_2_x, central_2_diff_y, label="y', 2nd-order central differences", s=1)
plt.scatter(central_3_x, central_3_diff_y, label="y', 3nd-order central differences", s=1)
plt.scatter(forward_x, forward_diff_y, label="y', forward differences", s=1)
plt.legend()
plt.savefig(f'thing.png', dpi=300)
