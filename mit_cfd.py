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
        u_diff_x_forward = (-u[2:-2,4:] + 6*u[2:-2,3:-1] - 3*u[2:-2,2:-2] - 2*u[2:-2,1:-3]) / (6*dx)
        u_diff_x_backward = (2*u[2:-2,3:-1] + 3*u[2:-2,2:-2] - 6*u[2:-2,1:-3] + u[2:-2,:-4]) / (6*dx)
        #u_diff_x_forward = (u[2:-2,3:-1] - u[2:-2,2:-2]) / dx
        #u_diff_x_backward = (u[2:-2,2:-2] - u[2:-2,1:-3]) / dx
        u_diff_x_forward_coef = np.clip(u[2:-2,2:-2], None, 0)
        u_diff_x_backward_coef = np.clip(u[2:-2,2:-2], 0, None)
        uu_x = u_diff_x_forward_coef*u_diff_x_forward + u_diff_x_backward_coef*u_diff_x_backward

        # v*u_y
        v_avg = (v[2:-1,2:-3] + v[2:-1,3:-2] + v[1:-2,2:-3] + v[1:-2,3:-2]) / 4
        u_diff_y_forward = (-u[4:,2:-2] + 6*u[3:-1,2:-2] - 3*u[2:-2,2:-2] - 2*u[1:-3,2:-2]) / (6*dy)
        u_diff_y_backward = (2*u[3:-1,2:-2] + 3*u[2:-2,2:-2] - 6*u[1:-3,2:-2] + u[:-4,2:-2]) / (6*dy)
        #u_diff_y_forward = (u[3:-1,2:-2] - u[2:-2,2:-2]) / dy
        #u_diff_y_backward = (u[2:-2,2:-2] - u[1:-3,2:-2]) / dy
        u_diff_y_forward_coef = np.clip(v_avg, None, 0)
        u_diff_y_backward_coef = np.clip(v_avg, 0, None)
        vu_y = u_diff_y_forward_coef*u_diff_y_forward + u_diff_y_backward_coef*u_diff_y_backward

        ## Changing v
        # v*v_y
        v_diff_y_forward = (-v[4:,2:-2] + 6*v[3:-1,2:-2] - 3*v[2:-2,2:-2] - 2*v[1:-3,2:-2]) / (6*dy)
        v_diff_y_backward = (2*v[3:-1,2:-2] + 3*v[2:-2,2:-2] - 6*v[1:-3,2:-2] + v[:-4,2:-2]) / (6*dy)
        #v_diff_y_forward = (v[3:-1,2:-2] - v[2:-2,2:-2]) / dy
        #v_diff_y_backward = (v[2:-2,2:-2] - v[1:-3,2:-2]) / dy
        v_diff_y_forward_coef = np.clip(v[2:-2,2:-2], None, 0)
        v_diff_y_backward_coef = np.clip(v[2:-2,2:-2], 0, None)
        vv_y = v_diff_y_forward_coef*v_diff_y_forward + v_diff_y_backward_coef*v_diff_y_backward

        # u*v_x
        u_avg = (u[2:-3,2:-1] + u[3:-2,2:-1] + u[2:-3,1:-2] + u[3:-2,1:-2]) / 4
        v_diff_x_forward = (-v[2:-2,4:] + 6*v[2:-2,3:-1] - 3*v[2:-2,2:-2] - 2*v[2:-2,1:-3]) / (6*dx)
        v_diff_x_backward = (2*v[2:-2,3:-1] + 3*v[2:-2,2:-2] - 6*v[2:-2,1:-3] + v[2:-2,:-4]) / (6*dx)
        #v_diff_x_forward = (v[2:-2,3:-1] - v[2:-2,2:-2]) / dx
        #v_diff_x_backward = (v[2:-2,2:-2] - v[1:-3,2:-2]) / dx
        v_diff_x_forward_coef = np.clip(u_avg, None, 0)
        v_diff_x_backward_coef = np.clip(u_avg, None, 0)
        uv_x = v_diff_x_forward_coef*v_diff_x_forward + v_diff_x_forward_coef*v_diff_x_forward

        u[2:-2,2:-2] -= dt * (uu_x + vu_y)
        v[2:-2,2:-2] -= dt * (vv_y + uv_x)

def diff_comp(u, dx):
        diff_3 = (-u[2:-2,4:] + 6*u[2:-2,3:-1] - 3*u[2:-2,2:-2] - 2*u[2:-2,1:-3]) / (6*dx)
        diff_1 = (u[2:-2,3:-1] - u[2:-2,2:-2]) / dx

        return diff_1, diff_3

# Apply once to each dimension in the velocity field
def apply_viscosity(u, nu, dx, dy, dt):
        rhs = u.copy()[2:-2,2:-2]

        denom = (2*nu*dt)/(dx**2) + (2*nu*dt)/(dy**2) + 1
        for iter in range(100):
                horiz = (u[2:-2,3:-1] + u[2:-2,1:-3]) / (dx**2)
                vert = (u[3:-1,2:-2] + u[1:-3,2:-2]) / (dy**2)
                u[2:-2,2:-2] = (rhs + nu*dt*(horiz + vert) ) / denom

def apply_bc(u, v):
        # Prescribed values
        u_north, u_south, u_west, u_east = 10, 0, 0, 0
        v_north, v_south, v_west, v_east = 0, 0, 0, 0

        u[1,:] = 2*u_south - u[2,:]
        u[-2,:] = 2*u_north - u[-3,:]
        u[:,1] = u_west
        u[:,-2] = u_east
        v[1,:] = v_south
        v[-2,:] = v_north
        v[:,1] = 2*v_west - v[:,2]
        v[:,-2] = 2*v_east - v[:,-3]

        u[0,:] = 2*u_south - u[2,:]
        u[-1,:] = 2*u_north - u[-3,:]
        u[:,0] = u_west
        u[:,-1] = u_east
        v[0,:] = v_south
        v[-1,:] = v_north
        v[:,0] = 2*v_west - v[:,2]
        v[:,-1] = 2*v_east - v[:,-3]

def pressure_rhs(u, v, dx, dy, dt):
        u_diff_x = (u[2:-2,2:-1] - u[2:-2,1:-2]) / dx
        v_diff_y = (v[2:-1,2:-2] - v[1:-2,2:-2]) / dy

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

        u[2:-2,1:-1] -= dt * p_diff_x
        v[1:-1,2:-2] -= dt * p_diff_y

# Setup mesh
len_x, len_y = 2, 2
# Number of pressure points
nx, ny = 400,400
dx, dy = len_x / nx, len_y / ny
# These are the positions of the cell centers
x_points = np.linspace(-dx/2, len_x+dx/2, nx+2)
y_points = np.linspace(-dy/2, len_y+dy/2, ny+2)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)
# U meshes: horizontally from -1..len_x+1, vertically from -1.5*dy..len_y+1.5*dy
u_x_mesh, u_y_mesh = np.meshgrid(np.linspace(-1, len_x+1, nx+3), np.linspace(-1.5*dy, len_y+1.5*dy, ny+4))
# V meshes: horizontally from -1.5*dx..len_x+1.5*dx, vertically from -1..len_y+1
v_x_mesh, v_y_mesh = np.meshgrid(np.linspace(-1.5*dx, len_x+1.5*dx, nx+4), np.linspace(-1, len_y+1, ny+3))

# Constants
dt = 0.0001
nu = 1e-3

# U: (ny+4, nx+3)
# V: (ny+3, nx+4)
u = np.zeros((ny+4, nx+3))
v = np.zeros((ny+3, nx+4))
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
                #vorticity = common.vorticity(u, v, dx, dy)
                #plt.subplot(1, 2, 1)
                #divergence = common.divergence(u_avg, v_avg, dx, dy)
                plt.imshow(pressure, cmap=plt.cm.coolwarm, origin='lower', extent=(0, len_x, 0, len_y), vmin=-10, vmax=10)
                plt.colorbar()

                lw = 5*speed/speed.max()
                plt.streamplot(x_mesh, y_mesh, u_avg, v_avg, linewidth=lw)
                #plt.quiver(x_mesh, y_mesh, u_avg, v_avg, scale=20)

                '''
                plt.subplot(1, 2, 2)
                plt.imshow(diff_3, cmap=plt.cm.coolwarm, origin='lower', extent=(0, len_x, 0, len_y), vmin=-1, vmax=1)
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

