import numpy as np
import matplotlib.pyplot as plt
import common
import time

def average_velocities(u, v):
        u_avg = (u[1:-1,1:] + u[1:-1,:-1]) / 2
        v_avg = (v[1:,1:-1] + v[:-1,1:-1]) / 2

        return u_avg, v_avg

def convect(u, v, dx, dy, dt):
        # First: u*u_x
        u_right_avg = (u[1:-1,2:] + u[1:-1,1:-1]) / 2
        u_left_avg = (u[1:-1,:-2] + u[1:-1,1:-1]) / 2

        u_diff_x_forward = (u[1:-1,2:] - u[1:-1,1:-1]) / dx
        u_diff_x_backward = (u[1:-1,1:-1] - u[1:-1,:-2]) / dx
        # Only use forward if u is negative
        u_forward_coef = np.clip(u_right_avg, None, 0)
        # Only use backward if u is positive
        u_backward_coef = np.clip(u_left_avg, 0, None)

        u[1:-1,1:-1] -= dt * (u_forward_coef*u_diff_x_forward + u_backward_coef*u_diff_x_backward)

# Apply once to each dimension in the velocity field
def apply_viscosity(u, nu, dx, dy, dt):
        rhs = u.copy()

        div = nu*(-2/(dx**2) - 2/(dy**2))

        for iter in range(100):
                u[1:-1,1:-1] = (rhs[1:-1,1:-1] + dt*nu*common.laplacian(u.copy(), dx, dy)) / div

# Setup mesh
len_x, len_y = 2, 2
# Number of pressure points
nx, ny = 100, 100
dx, dy = len_x / nx, len_y / ny
# These are the positions of the cell centers
x_points = np.linspace(dx/2, len_x-dx/2, nx)
y_points = np.linspace(dy/2, len_y-dy/2, ny)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)
# U meshes: horizontally from 0..len_x, vertically from -0.5*dy..len_y+0.5*dy
u_x_mesh, u_y_mesh = np.meshgrid(np.linspace(0, len_x, nx+1), np.linspace(-0.5*dy, len_y+0.5*dy, ny+2))
# V meshes: horizontally from -0.5*dx..len_x+0.5*dx, vertically from 0..len_y
v_x_mesh, v_y_mesh = np.meshgrid(np.linspace(-0.5*dx, len_x+0.5*dx, nx+2), np.linspace(0, len_y, ny+1))

# Constants
dt = 0.001
nu = 0.1

# U: (ny+2, nx+1)
# V: (ny+1, nx+2)
u = np.ones((ny+2, nx+1))
v = np.zeros((ny+1, nx+2))
u[10,40:50] = 10

for i in range(1000001):
        start_time = time.time()
        print(i)

        convect(u, v, dx, dy, dt)
        #apply_viscosity(u, nu, dx, dy, dt)
        #apply_viscosity(v, nu, dx, dy, dt)

        u_avg, v_avg = average_velocities(u, v)
        speed = np.sqrt(u_avg**2+v_avg**2)
        dt = min(0.1, 0.1 * min(dx, dy) / np.max(speed))
        print(f'Max speed: {np.max(speed)}, dt: {dt}')

        end_time = time.time()

        if i % 30 == 0 or end_time - start_time > 1:
                divergence = common.divergence(u_avg, v_avg, dx, dy)
                plt.imshow(divergence, cmap=plt.cm.coolwarm, origin='lower', extent=(0, len_x, 0, len_y))
                plt.colorbar()

                lw = 5*speed/speed.max()
                #plt.streamplot(x_mesh[:-1,:-1], y_mesh[:-1,:-1], u_avg, v_avg, linewidth=lw)
                plt.quiver(x_mesh, y_mesh, u_avg, v_avg, scale=200)
                plt.savefig(f'img/{i:07}.png', dpi=300)
                plt.clf()