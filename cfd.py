import numpy as np
import matplotlib.pyplot as plt
import common

# (m-2,n-2)
# w is the vector field to take the partial with respect to (should be u for the u term, v for the
# v term)
def calc_convection(u, v, w, dx, dy):
        out = np.zeros_like(w)

        w_left = w[1:-1,:-2]
        w_right = w[1:-1,2:]
        w_up = w[:-2,1:-1]
        w_down = w[2:,1:-1]

        # Central differences
        w_diff_x = (w_right - w_left) / (2*dx)
        w_diff_y = (w_down - w_up) / (2*dy)

        return u[1:-1,1:-1] * w_diff_x + v[1:-1,1:-1] * w_diff_y

def calc_viscosity(u, nu, dx, dy):
        return nu * common.laplacian(u, dx, dy)

def calc_pressure_corrector_rhs(u, v, rho, dx, dy, dt):
        return rho / dt * common.divergence(u, v, dx, dy)

def corrector_pressure(u, v, p, rho, dx, dy, dt):
        # Central differences for pressure
        pressure_diff_x = (p[1:-1,2:] - p[1:-1,:-2]) / (2*dx)
        pressure_diff_y = (p[2:,1:-1] - p[:-2,1:-1]) / (2*dy)

        new_u, new_v = u.copy(), v.copy()
        new_u[1:-1,1:-1] = u[1:-1,1:-1] - dt * (1 / rho) * pressure_diff_x
        new_v[1:-1,1:-1] = v[1:-1,1:-1] - dt * (1 / rho) * pressure_diff_y

        return new_u, new_v

def apply_velocity_bc(u, v):
        u[0,:] = 0
        u[-1,:] = 0
        u[:,0] = 0
        u[:,-1] = 0
        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0

        u[50:51,10:-10] = 100

# Mesh
nx = 200
ny = 200
Lx = 4
Ly = 4
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x_points = np.linspace(0, Lx, nx)
y_points = np.linspace(0, Ly, ny)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)

# Constants
rho = 1
nu = 0.1
dt = 0.00001

# Velocity
u = np.zeros_like(x_mesh)
v = np.zeros_like(y_mesh)
apply_velocity_bc(u, v)
pressure = np.zeros_like(u)

for i in range(1000001):
        print(i)
        u_convection = calc_convection(u, v, u, dx, dy)
        v_convection = calc_convection(u, v, v, dx, dy)
        u_viscosity = calc_viscosity(u, nu, dx, dy)
        v_viscosity = calc_viscosity(v, nu, dx, dy)
        u[1:-1,1:-1] += dt * (u_viscosity - u_convection)
        v[1:-1,1:-1] += dt * (v_viscosity - v_convection)

        rhs = calc_pressure_corrector_rhs(u, v, rho, dx, dy, dt)
        pressure = common.poisson_solve(pressure, rhs, dx, dy)
        u, v = corrector_pressure(u, v, pressure, rho, dx, dy, dt)

        apply_velocity_bc(u, v)

        if i % 30 == 0:
                #divergence = common.divergence(u, v, dx, dy)
                plt.imshow(rhs, cmap=plt.cm.coolwarm, origin='lower', extent=(0, Lx, 0, Ly))
                plt.colorbar()
                speed = np.sqrt(u**2+v**2)
                lw = 5*speed/speed.max()
                plt.streamplot(x_mesh, y_mesh, u, v, linewidth=lw)
                #plt.quiver(x_mesh[::4], y_mesh[::4], u[::4], v[::4], scale=200)
                plt.savefig(f'img/{i:07}.png', dpi=300)
                plt.clf()
