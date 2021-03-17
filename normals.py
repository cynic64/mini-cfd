import numpy as np
import matplotlib.pyplot as plt

def calc_boundary_normals(line, nx, ny, dx, dy):
        '''
        Returns 2 numpy arrays, u and v of each normal. (ny,nx), (ny,nx).
        All non-boundary cells are 0, 0
        '''
        u, v = np.zeros((ny, nx)), np.zeros((ny, nx))

        step_size = min(dx, dy)
        for i in range(0, len(line) - 1):
                start, end = line[i+1], line[i]
                length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                step_count = int(length / step_size) + 1
                for j in range(step_count + 1):
                        x = start[0] + (end[0] - start[0]) / step_count * j
                        y = start[1] + (end[1] - start[1]) / step_count * j
                        if x < 0 or x >= nx*dx or y < 0 or y >= ny*dy:
                                continue

                        ix, iy = int(x / dx), int(y / dy)
                        u[iy,ix] = -(end[1] - start[1]) / length
                        v[iy,ix] = (end[0] - start[0]) / length

        return u, v

x_points, y_points = [], []
segments = 5
for i in range(segments+1):
        x_points.append(0.5 + 0.5 * np.cos(2 * np.pi / segments * i))
        y_points.append(0.5 + 0.5 * np.sin(2 * np.pi / segments * i))

line = []
for i in range(len(x_points)):
        line.append([x_points[i], y_points[i]])

Lx, Ly, nx, ny = 1, 1, 10, 10
dx, dy = Lx / nx, Ly / ny
x_range, y_range = np.linspace(dx/2, Lx-dx/2, nx), np.linspace(dy/2, Ly-dy/2, ny)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
normals_u, normals_v = calc_boundary_normals(line, nx, ny, dx, dy)
print(line)
print()
print(normals_u)
print()
print(normals_v)
print()
print(x_mesh)

plt.plot(x_points, y_points)
plt.quiver(x_mesh, y_mesh, normals_u, normals_v, scale=20, angles='xy')
plt.show()
