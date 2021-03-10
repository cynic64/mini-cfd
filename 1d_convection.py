import numpy as np
import matplotlib.pyplot as plt

def convect(u, prev_u, c, dx, dt):
        # First: FTCS
        u_diff_x = (u[2:] - u[:-2]) / (2*dx)

        new_u = u.copy()[1:-1]
        new_u += dt * -c * u_diff_x

        # Adjust
        u_diff2_t = (prev_u[1:-1] - 2*u[1:-1] + new_u) / (dt**2)

        new_u += 0.5 * dt**2 * u_diff2_t
        u[1:-1] = new_u

Lx = 2
# Number of cells, not data points
nx = 200
x_points = np.linspace(0, Lx, nx+1)
dx = Lx / nx
dt = 0.001
c = 5

u = np.zeros(nx+1)
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

        u_orig = u.copy()
        convect(u, prev_u, c, dx, dt)
        prev_u = u_orig
