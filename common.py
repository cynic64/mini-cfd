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
