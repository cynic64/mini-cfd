import numpy as np
import matplotlib.pyplot as plt
import math

def find_root(x, y, dx):
        # This is an index, not an x coordinate
        guess = len(y) // 2
        for iter in range(30):
                if guess < 1 or guess >= len(y) - 1:
                        return guess

                f = y[guess]
                f_diff_x = (y[guess + 1] - y[guess - 1]) / (2*dx)

                guess = math.floor(guess - f / f_diff_x)
                print(f'Guess {iter}: {guess}')

        return guess

len_x = 5
dx = 0.01
X = np.arange(0, len_x+dx, dx)
Y = -X**2 + 5

root_idx = find_root(X, Y, dx)
print(root_idx*dx)
print(Y[root_idx-2:root_idx+3])

plt.plot(X, Y)
plt.show()
