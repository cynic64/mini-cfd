import numpy as np

# Returns array with shape (m-1,n-1)
def divergence(u, v, dx, dy):
        u_diff_x = (u[:-1,1:] - u[:-1,:-1]) / dx
        v_diff_y = (v[1:,:-1] - v[:-1,:-1]) / dy

        return u_diff_x + v_diff_y

# Returns array with shape (m-2,n-2)
def laplacian(u, dx, dy):
        u_diff2_x = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dx**2
        u_diff2_y = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dy**2

        return u_diff2_x + u_diff2_y

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
def poisson_solve(p, rhs, dx, dy, target_diff=0.0001):
        for i in range(1, 100000):
                horiz_part = (p[1:-1,2:] + p[1:-1,:-2]) * dy**2
                vert_part = (p[2:,1:-1] + p[:-2,1:-1]) * dx**2

                p_next = np.empty_like(p)
                p_next[1:-1,1:-1] = (horiz_part + vert_part - (rhs * dx**2 * dy**2)) \
                        / (2 * dx**2 + 2 * dy**2)

                # Derivatives at borders should be 0
                p_next[0,:] = p_next[1,:]
                p_next[-1,:] = p_next[-2,:]
                p_next[:,0] = p_next[:,1]
                p_next[:,-1] = p_next[:,-2]

                diff = np.sum(np.abs(p_next - p)) / np.sum(np.abs(p_next))
                if diff < target_diff:
                        break

                p = p_next
        print(f'Took {i} iterations.')

        return p

