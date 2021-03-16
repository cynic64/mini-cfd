import numpy as np
import matplotlib.pyplot as plt
import common
import time

def average_velocities(u, v):
        u_avg = (u[1:-1,1:] + u[1:-1,:-1]) / 2
        v_avg = (v[1:,1:-1] + v[:-1,1:-1]) / 2

        return u_avg, v_avg

def vorticity(u, v, dx, dy):
        u_diff_x = (u[:,1:] - u[:,:-1]) / dx
        u_diff_y = (u[1:,:] - u[:-1,:]) / dy
        u_diff_y_centered = (u_diff_y[1:,1:] + u_diff_y[1:,:-1] + u_diff_y[:-1,1:] + u_diff_y[:-1,:-1]) / 4

        v_diff_x = (v[:,1:] - v[:,:-1]) / dx
        v_diff_x_centered = (v_diff_x[1:,1:] + v_diff_x[1:,:-1] + v_diff_x[:-1,1:] + v_diff_x[:-1,:-1]) / 4
        v_diff_y = (v[1:,:] - v[:-1,:]) / dy

        return v_diff_x - u_diff_y

def convect(u, v, dx, dy, dt):
        v_at_u = (v[1:,1:-2] + v[1:,2:-1] + v[:-1,1:-2] + v[:-1,2:-1]) / 4
        u_at_v = (u[1:-2,1:] + u[2:-1,1:] + u[1:-2,:-1] + u[2:-1,:-1]) / 4

        # Predict
        predicted_u = u.copy()
        u_diff_x_forward = (u[1:-1,2:] - u[1:-1,1:-1]) / dx
        u_diff_y_forward = (u[2:,1:-1] - u[1:-1,1:-1]) / dy
        predicted_u[1:-1,1:-1] += -dt * (u[1:-1,1:-1] * u_diff_x_forward + v_at_u * u_diff_y_forward)

        predicted_v = v.copy()
        v_diff_x_forward = (v[1:-1,2:] - v[1:-1,1:-1]) / dx
        v_diff_y_forward = (v[2:,1:-1] - v[1:-1,1:-1]) / dy
        predicted_v[1:-1,1:-1] += -dt * (u_at_v * v_diff_x_forward + v[1:-1,1:-1] * v_diff_y_forward)

        # Correct
        u_corrected = (u + predicted_u) / 2
        u_pred_diff_x_backward = (predicted_u[1:-1,1:-1] - predicted_u[1:-1,:-2]) / dx
        u_pred_diff_y_backward = (predicted_u[1:-1,1:-1] - predicted_u[:-2,1:-1]) / dy
        u_corrected[1:-1,1:-1] += -0.5 * dt * (u[1:-1,1:-1] * u_pred_diff_x_backward + v_at_u * u_pred_diff_y_backward)

        v_corrected = (v + predicted_v) / 2
        v_pred_diff_x_backward = (predicted_v[1:-1,1:-1] - predicted_v[1:-1,:-2]) / dx
        v_pred_diff_y_backward = (predicted_v[1:-1,1:-1] - predicted_v[:-2,1:-1]) / dy
        v_corrected[1:-1,1:-1] += -0.5 * dt * (u_at_v * v_pred_diff_x_backward + v[1:-1,1:-1] * v_pred_diff_y_backward)

        u[1:-1,1:-1] = u_corrected[1:-1,1:-1]
        v[1:-1,1:-1] = v_corrected[1:-1,1:-1]

# Apply once to each dimension in the velocity field
def apply_viscosity(u, nu, dx, dy, dt):
        rhs = u.copy()[1:-1,1:-1]

        denom = (2*nu*dt)/(dx**2) + (2*nu*dt)/(dy**2) + 1
        for iter in range(100):
                horiz = (u[1:-1,2:] + u[1:-1,:-2]) / (dx**2)
                vert = (u[2:,1:-1] + u[:-2,1:-1]) / (dy**2)
                u[1:-1,1:-1] = (rhs + nu*dt*(horiz + vert) ) / denom

def apply_bc(u, v):
        # Prescribed values
        u_north, u_south, u_west, u_east = 10, 0, 0, 0
        v_north, v_south, v_west, v_east = 0, 0, 0, 0

        u[0,:] = 2*u_south - u[1,:]
        u[-1,:] = 2*u_north - u[-2,:]
        u[:,0] = u_west
        u[:,-1] = u_east
        v[0,:] = v_south
        v[-1,:] = v_north
        v[:,0] = 2*v_west - v[:,1]
        v[:,-1] = 2*v_east - v[:,-2]

def pressure_rhs(u, v, dx, dy, dt):
        u_diff_x = (u[1:-1,1:] - u[1:-1,:-1]) / dx
        v_diff_y = (v[1:,1:-1] - v[:-1,1:-1]) / dy

        return (1/dt) * (u_diff_x + v_diff_y)

def poisson_solve(p, rhs, dx, dy):
        for iter in range(100):
                horiz = dy**2 * (p[1:-1,2:] + p[1:-1,:-2])
                vert = dx**2 * (p[2:,1:-1] + p[:-2,1:-1])

                p[1:-1,1:-1] = (horiz + vert - dx**2 * dy**2 * rhs) / (2*dx**2 + 2 *dy**2)
                p[0,:] = p[1,:]
                p[-1,:] = p[-2,:]
                p[:,0] = p[:,1]
                p[:,-1] = p[:,-2]

def apply_pressure(u, v, p, dx, dy, dt):
        p_diff_x = (p[1:-1,1:] - p[1:-1,:-1]) / dx
        p_diff_y = (p[1:,1:-1] - p[:-1,1:-1]) / dy

        u[1:-1,:] -= dt * p_diff_x
        v[:,1:-1] -= dt * p_diff_y

# Setup mesh
len_x, len_y = 2, 2
# Number of pressure points
nx, ny = 400,400
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
dt = 0.0002
nu = 0.0001

# U: (ny+2, nx+1)
# V: (ny+1, nx+2)
u = np.zeros((ny+2, nx+1))
v = np.zeros((ny+1, nx+2))
#u = np.sin(u_x_mesh**2 + u_y_mesh**2) * 10
#v = np.cos(v_x_mesh**2 + v_y_mesh**2) * 10
pressure = np.zeros((ny+2, nx+2))
apply_bc(u, v)

start_time = 0
end_time = time.time()

for i in range(1000001):
        u_avg, v_avg = average_velocities(u, v)
        speed = np.sqrt(u_avg**2+v_avg**2)
        print(f'Max speed: {np.max(speed)}, dt: {dt}')

        if i % 50 == 0 or end_time - start_time > 1 or i <= 10:
                #plt.subplot(1, 2, 1)
                #divergence = common.divergence(u_avg, v_avg, dx, dy)
                vort = vorticity(u, v, dx, dy)
                plt.imshow(vort, cmap=plt.cm.coolwarm, origin='lower', extent=(0, len_x, 0, len_y), vmin=-10, vmax=10)
                plt.colorbar()

                lw = 5*speed/speed.max()
                #plt.streamplot(x_mesh, y_mesh, u_avg, v_avg, linewidth=lw)
                plt.quiver(x_mesh[::8,::8], y_mesh[::8,::8], u_avg[::8,::8], v_avg[::8,::8], scale=50)

                '''
                plt.subplot(1, 2, 2)
                plt.imshow(v, cmap=plt.cm.coolwarm, origin='lower', extent=(0, len_x, 0, len_y), vmin=-20, vmax=20)
                plt.colorbar()
                '''

                plt.savefig(f'img/{i:07}.png', dpi=300)
                plt.clf()

        start_time = time.time()
        print(i)

        apply_bc(u, v)
        convect(u, v, dx, dy, dt)
        apply_viscosity(u, nu, dx, dy, dt)
        apply_viscosity(v, nu, dx, dy, dt)
        rhs = pressure_rhs(u, v, dx, dy, dt)
        poisson_solve(pressure, rhs, dx, dy)
        apply_pressure(u, v, pressure, dx, dy, dt)

        end_time = time.time()
