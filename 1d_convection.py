import numpy as np
import matplotlib.pyplot as plt

def convect(u, dx, dt):
        middle = u[2:-2]

        diff_forward = ( -u[4:] + 6*u[3:-1] + -3*middle + -2*u[1:-3] ) / (6*dx)
        diff_backward = ( 2*u[3:-1] + 3*middle + -6*u[1:-3] + u[:-4] ) / (6*dx)
        coef_forward = np.clip(middle, None, 0)
        coef_backward = np.clip(middle, 0, None)

        step = coef_forward*diff_forward + coef_backward*diff_backward

        u[2:-2] -= dt*step

Lx = 2
# Number of cells, not data points
nx = 200
x_points = np.linspace(0, Lx, nx+1)
dx = Lx / nx
dt = 0.001

u = np.zeros(nx+1)
u[10:20] = 10

for i in range(1000001):
        print(i)

        if i % 30 == 0 or i<10:
                plt.plot(x_points, u, scaley=False)
                plt.yticks(np.arange(-15, 16, 2))
                plt.xlabel('Position')
                plt.ylabel('Velocity')
                plt.title(f't={i*dt}')

                plt.savefig(f'img/{i:07}.png', dpi=300)
                plt.clf()

        convect(u, dx, dt)
