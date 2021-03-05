import numpy as np
import matplotlib.pyplot as plt

def convect(u, dx, dt):
        u_right_avg = (u[1:-1] + u[2:]) / 2
        u_left_avg = (u[1:-1] + u[:-2]) / 2

        u_plus = np.maximum(u_left_avg, np.zeros_like(u_left_avg))
        u_minus = np.minimum(u_right_avg, np.zeros_like(u_right_avg))

        u_minus_diff_x = (u[1:-1] - u[:-2]) / dx
        u_plus_diff_x = (u[2:] - u[1:-1]) / dx

        u[1:-1] -= dt * (u_plus*u_minus_diff_x + u_minus*u_plus_diff_x)

Lx = 2
# Number of cells, not data points
nx = 100
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
