import numpy as np
import matplotlib.pyplot as plt
import common

# Returns (m, n), (m, n)
def next_given_pressure(u, v, p, rho, nu, dx, dy, dt):
        u_middle = u[1:-1,1:-1]
        v_middle = v[1:-1,1:-1]
        # Central differences for pressure
        p_diff_x = (p[1:-1,2:] - p[1:-1,:-2]) / (2*dx)
        p_diff_y = (p[2:,1:-1] - p[:-2,1:-1]) / (2*dy)

        # Backward differences for first velocity derivatives
        u_diff_x = (u_middle - u[1:-1,:-2]) / dx
        u_diff_y = (u_middle - u[:-2,1:-1]) / dy

        v_diff_x = (v_middle - v[1:-1,:-2]) / dx
        v_diff_y = (v_middle - v[:-2,1:-1]) / dy

        # 5-point thing for second velocity derivatives
        u_diff2_x = (u[1:-1,2:] - 2*u_middle + u[1:-1,:-2]) / dx**2
        u_diff2_y = (u[2:,1:-1] - 2*u_middle + u[:-2,1:-1]) / dy**2

        v_diff2_x = (v[1:-1,2:] - 2*v_middle + v[1:-1,:-2]) / dx**2
        v_diff2_y = (v[2:,1:-1] - 2*v_middle + v[:-2,1:-1]) / dy**2

        u_next, v_next = u.copy(), v.copy()
        u_next[1:-1,1:-1] += dt * ((-1/rho)*p_diff_x + nu*(u_diff2_x + u_diff2_y) - (u_middle*u_diff_x + v_middle*u_diff_y))
        v_next[1:-1,1:-1] += dt * ((-1/rho)*p_diff_y + nu*(v_diff2_x + v_diff2_y) - (u_middle*v_diff_x + v_middle*v_diff_y))

        # Velocity boundary conditions
        apply_velocity_bc(u_next, v_next)

        return u_next, v_next

def apply_velocity_bc(u, v):
        u[0,:] = 0
        u[-1,:] = 10
        u[:,0] = 0
        u[:,-1] = 0
        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0

# Mesh
nx = 50
ny = 50
Lx = 2
Ly = 2
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x_points = np.linspace(0, Lx, nx)
y_points = np.linspace(0, Ly, ny)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)
# Each thing is 1/200=.005 across
# Top velocity is 10, dt = 0.0001 --> 0.001 each time step

# Constants
rho = 1
nu = 0.1
dt = 0.001

# Velocity
u = x_mesh**2
u = np.zeros(x_mesh.shape)
v = np.zeros(y_mesh.shape)
apply_velocity_bc(u, v)
pressure = np.zeros(u.shape)

for i in range(1001):
        print(i)
        rhs = common.poisson_rhs(u, v, rho, dx, dy, dt)
        pressure = common.poisson_solve(pressure, rhs, dx, dy)
        u, v = next_given_pressure(u, v, pressure, rho, nu, dx, dy, dt)
        divergence = common.divergence(u, v, dx, dy)

        if i % 1000 == 0:
                plt.imshow(divergence, cmap=plt.cm.coolwarm, origin='lower', extent=(0, Lx, 0, Ly))
                plt.colorbar()
                speed = np.sqrt(u**2+v**2)
                lw = 5*speed/speed.max()
                plt.streamplot(x_mesh, y_mesh, u, v, linewidth=lw)
                plt.quiver(x_mesh[::4], y_mesh[::4], u[::4], v[::4])
                plt.savefig(f'img/{i}.png', dpi=300)
                plt.clf()

plt.show()

