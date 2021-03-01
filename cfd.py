import numpy as np
import matplotlib.pyplot as plt
import common
import time

# ((m-2,n-2), (m-2,n-2))
def calc_convection(u, v, dx, dy):
        # Central differences
        u_diff_x = (u[1:-1,2:] - u[1:-1,:-2]) / (2*dx)
        u_diff_y = (u[2:,1:-1] - u[:-2,1:-1]) / (2*dy)

        v_diff_x = (v[1:-1,2:] - v[1:-1,:-2]) / (2*dx)
        v_diff_y = (v[2:,1:-1] - v[:-2,1:-1]) / (2*dy)

        u_step = u[1:-1,1:-1] * u_diff_x + v[1:-2,1:] * u_diff_y
        v_step = u[1:,1:-2] * v_diff_x + v[1:-1,1:-1] * v_diff_y

        return u_step, v_step

def calc_viscosity(u, nu, dx, dy):
        return nu * common.laplacian(u, dx, dy)

def calc_pressure_corrector_rhs(u, v, rho, dx, dy, dt):
        u_diff_x = (u[:,1:] - u[:,:-1]) / dx
        v_diff_y = (v[1:,:] - v[:-1,:]) / dy

        return rho / dt * (u_diff_x + v_diff_y)

def corrector_pressure(u, v, p, rho, dx, dy, dt):
        p_diff_x = (p[:,1:] - p[:,:-1]) / dx
        p_diff_y = (p[1:,:] - p[:-1,:]) / dy

        new_u, new_v = u.copy(), v.copy()
        new_u[:,1:-1] = u[:,1:-1] - dt * (1 / rho) * p_diff_x
        new_v[1:-1,:] = v[1:-1,:] - dt * (1 / rho) * p_diff_y

        return new_u, new_v

def apply_velocity_bc(u, v):
        u[:,0] = 0
        u[:,-1] = 0
        v[0,:] = 0
        v[-1,:] = 0

        u[50:51,20:-20] = 30

# Mesh
# Size of pressure grid
nx = 200
ny = 200
Lx = 4
Ly = 4
dx = Lx / nx
dy = Ly / ny
x_points = np.linspace(0, Lx, nx+1)
y_points = np.linspace(0, Ly, ny+1)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)

# Constants
rho = 1
nu = 0.1
dt = 0.0001

# Velocity
u = np.zeros((ny, nx+1))
v = np.zeros((ny+1, nx))
apply_velocity_bc(u, v)
pressure = np.ones((nx, ny))

for i in range(1000001):
        start_time = time.time()
        print(i)
        u_convection, v_convection = calc_convection(u, v, dx, dy)
        u_viscosity = calc_viscosity(u, nu, dx, dy)
        v_viscosity = calc_viscosity(v, nu, dx, dy)
        u[1:-1,1:-1] += dt * (u_viscosity - u_convection)
        v[1:-1,1:-1] += dt * (v_viscosity - v_convection)
        apply_velocity_bc(u, v)

        rhs = calc_pressure_corrector_rhs(u, v, rho, dx, dy, dt)
        pressure = common.poisson_solve(pressure, rhs, dx, dy, target_diff=0.0001)
        u, v = corrector_pressure(u, v, pressure, rho, dx, dy, dt)

        apply_velocity_bc(u, v)
        end_time = time.time()

        if i % 30 == 0 or end_time - start_time > 1:
                u_cropped, v_cropped = u[:,:-1], v[:-1,:]
                divergence = common.divergence(u_cropped, v_cropped, dx, dy)
                plt.imshow(pressure, cmap=plt.cm.coolwarm, origin='lower', extent=(0, Lx, 0, Ly))
                plt.colorbar()
                speed = np.sqrt(u_cropped**2+v_cropped**2)
                lw = 5*speed/speed.max()
                plt.streamplot(x_mesh[:-1,:-1], y_mesh[:-1,:-1], u_cropped, v_cropped, linewidth=lw)
                #plt.quiver(x_mesh[::4], y_mesh[::4], u[::4], v[::4], scale=200)
                plt.savefig(f'img/{i:07}.png', dpi=300)
                plt.clf()
