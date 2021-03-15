import numpy as np
import matplotlib.pyplot as plt

def convect(u, c, dx, dt):
        u_forward_diff_x = (u[2:] - u[1:-1]) / dx
        predicted = u.copy()
        predicted[1:-1] += -u[1:-1] * dt * u_forward_diff_x

        corrected = u.copy()
        predicted_backward_diff_x = (predicted[1:-1] - predicted[:-2])/dx
        corrected[1:-1] = (u[1:-1] + predicted[1:-1]) / 2 - 0.5*u[1:-1]*dt*predicted_backward_diff_x

        u[1:-1] = corrected[1:-1]

Lx = 2
# Number of cells, not data points
nx = 200
x_points = np.linspace(0, Lx, nx+1)
dx = Lx / nx
dt = 0.001
c = 5

u = np.ones(nx+1)
u[10:20] = 10
prev_u = u.copy()

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

        convect(u, c, dx, dt)
