import numpy as np
import matplotlib.pyplot as plt
import common

# Returns array with shape (m-2,n-2)
def poisson_rhs(u, v, rho, dx, dy, dt):
        u_diff_x = (u[1:-1,2:]-u[1:-1,:-2]) / (2*dx)
        u_diff_y = (u[2:,1:-1]-u[:-2,1:-1]) / (2*dy)

        v_diff_x = (v[1:-1:,2:]-v[1:-1,:-2]) / (2*dx)
        v_diff_y = (v[2:,1:-1]-v[:-2,1:-1]) / (2*dy)

        rhs = (u_diff_x + v_diff_y) / dt \
                - u_diff_x**2 - 2 \
                - 2*u_diff_y*v_diff_x \
                - v_diff_y**2

        return rho * rhs

# Returns array with shape (m+2,n+2)
# If given the previous pressure, it will be more accurate
def poisson_solve(p, rhs, dx, dy):
        for i in range(0, 100):
                horiz_part = (p[1:-1,2:] + p[1:-1,:-2]) * dy**2
                vert_part = (p[2:,1:-1] + p[:-2,1:-1]) * dx**2

                p[1:-1,1:-1] = (horiz_part + vert_part - (rhs * dx**2 * dy**2)) \
                        / (2 * dx**2 + 2 * dy**2)

                # Derivatives at borders should be 0
                p[0,:] = p[1,:]
                p[-1,:] = 0
                p[:,0] = p[:,1]
                p[:,-1] = p[:,-2]

        return p

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
        u[-1,:] = 100
        u[:,0] = 0
        u[:,-1] = 0
        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0

# Mesh
nx = 100
ny = 100
Lx = 1
Ly = 1
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x_points = np.linspace(0, Lx, nx)
y_points = np.linspace(0, Ly, ny)
x_mesh, y_mesh = np.meshgrid(x_points, y_points)

# Constants
rho = 1
nu = 0.1
dt = 0.0001

# Velocity
u = x_mesh**2
u = np.zeros(x_mesh.shape)
v = np.zeros(y_mesh.shape)
apply_velocity_bc(u, v)
pressure = np.zeros(u.shape)

for i in range(100):
        rhs = poisson_rhs(u, v, rho, dx, dy, dt)
        pressure = poisson_solve(pressure, rhs, dx, dy)
        u, v = next_given_pressure(u, v, pressure, rho, nu, dx, dy, dt)
        divergence = common.divergence(u, v, dx, dy)

plt.imshow(divergence, cmap=plt.cm.coolwarm, origin='lower', extent=(0, Lx, 0, Ly))
plt.colorbar()
speed = np.sqrt(u**2+v**2)
lw = 5*speed/speed.max()
plt.streamplot(x_mesh, y_mesh, u, v)
#plt.quiver(x_mesh[::2], y_mesh[::2], u[::2], v[::2])
plt.show()

