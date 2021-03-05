import numpy as np
import matplotlib.pyplot as plt
import common
import time

def average_velocities(u, v):
        u_avg = (u[1:-1,1:] + u[1:-1,:-1]) / 2
        v_avg = (v[1:,1:-1] + v[:-1,1:-1]) / 2

        return u_avg, v_avg

def convect(u, v, dx, dy, dt):
        ## Changing u
        # u*u_x
        u_right_avg = (u[1:-1,2:] + u[1:-1,1:-1]) / 2
        u_left_avg = (u[1:-1,:-2] + u[1:-1,1:-1]) / 2
        u_diff_x_forward = (u[1:-1,2:] - u[1:-1,1:-1]) / dx
        u_diff_x_backward = (u[1:-1,1:-1] - u[1:-1,:-2]) / dx
        # Only use forward if u is negative
        u_diff_x_forward_coef = np.clip(u_right_avg, None, 0)
        # Only use backward if u is positive
        u_diff_x_backward_coef = np.clip(u_left_avg, 0, None)
        uu_x = u_diff_x_forward_coef*u_diff_x_forward + u_diff_x_backward_coef*u_diff_x_backward

        # v*u_y
        v_down_avg = (v[:-1,1:-2] + v[:-1,2:-1]) / 2
        v_up_avg = (v[1:,1:-2] + v[1:,2:-1]) / 2
        u_diff_y_forward = (u[2:,1:-1] - u[1:-1,1:-1]) / dy
        u_diff_y_backward = (u[1:-1,1:-1] - u[:-2,1:-1]) / dy
        u_diff_y_forward_coef = np.clip(v_down_avg, None, 0)
        u_diff_y_backward_coef = np.clip(v_up_avg, 0, None)
        vu_y = u_diff_y_forward_coef*u_diff_y_forward + u_diff_y_backward_coef*u_diff_y_backward

        ## Changing v
        # v*v_y
        v_down_avg = (v[2:,1:-1] + v[1:-1,1:-1]) / 2
        v_up_avg = (v[1:-1,1:-1] + v[:-2,1:-1]) / 2
        v_diff_y_forward = (v[2:,1:-1] - v[1:-1,1:-1]) / dy
        v_diff_y_backward = (v[1:-1,1:-1] - v[:-2,1:-1]) / dy
        v_diff_y_forward_coef = np.clip(v_down_avg, None, 0)
        v_diff_y_backward_coef = np.clip(v_up_avg, 0, None)
        vv_y = v_diff_y_forward_coef*v_diff_y_forward + v_diff_y_backward_coef*v_diff_y_backward

        # u*v_y
        u_right_avg = (u[1:-2,1:] + u[2:-1,1:]) / 2
        u_left_avg = (u[1:-2,:-1] + u[2:-1,:-1]) / 2
        v_diff_x_forward = (v[1:-1,2:] - v[1:-1,1:-1]) / dx
        v_diff_x_backward = (v[1:-1,1:-1] - v[1:-1,:-2]) / dx
        v_diff_x_forward_coef = np.clip(u_right_avg, None, 0)
        v_diff_x_backward_coef = np.clip(u_left_avg, None, 0)
        uv_x = v_diff_x_forward_coef*v_diff_x_forward + v_diff_y_forward_coef*v_diff_y_forward

        u[1:-1,1:-1] -= dt * (uu_x + vu_y)
        v[1:-1,1:-1] -= dt * (vv_y + uv_x)

# Apply once to each dimension in the velocity field
def apply_viscosity(u, nu, dx, dy, dt):
        rhs = u.copy()

        div = nu*(-2/(dx**2) - 2/(dy**2))

        for iter in range(100):
                u[1:-1,1:-1] = (rhs[1:-1,1:-1] + dt*nu*common.laplacian(u.copy(), dx, dy)) / div

def apply_bc(u, v):
        u[0,:] = 0
        u[-1,:] = 0
        u[:,0] = 0
        u[:,-1] = 0
        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0

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
u = np.sin(u_x_mesh**2 + u_y_mesh**2)
v = np.cos(v_x_mesh**2 + v_y_mesh**2)
apply_bc(u, v)

start_time = 0
end_time = time.time()

for i in range(1000001):
        u_avg, v_avg = average_velocities(u, v)
        speed = np.sqrt(u_avg**2+v_avg**2)
        print(f'Max speed: {np.max(speed)}, dt: {dt}')

        if i % 30 == 0 or end_time - start_time > 1:
                divergence = common.divergence(u_avg, v_avg, dx, dy)
                plt.imshow(divergence, cmap=plt.cm.coolwarm, origin='lower', extent=(0, len_x, 0, len_y))
                plt.colorbar()

                lw = 5*speed/speed.max()
                #plt.streamplot(x_mesh[:-1,:-1], y_mesh[:-1,:-1], u_avg, v_avg, linewidth=lw)
                plt.quiver(x_mesh, y_mesh, u_avg, v_avg, scale=200)
                plt.savefig(f'img/{i:07}.png', dpi=300)
                plt.clf()

        start_time = time.time()
        print(i)

        convect(u, v, dx, dy, dt)
        #apply_viscosity(u, nu, dx, dy, dt)
        #apply_viscosity(v, nu, dx, dy, dt)

        end_time = time.time()

