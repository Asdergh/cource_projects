import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plt.style.use("dark_background")
fig = plt.figure()
axis = fig.add_subplot(projection="3d")

def animation(time):

    axis.clear()
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)

    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = np.cos(np.sqrt(grid_x * time ** 2 + grid_y * time ** 2)) / (np.sqrt(grid_x ** 2 * time + grid_y ** 2 * time))
    axis.plot_surface(grid_x, grid_y, grid_z, cmap="coolwarm", alpha=0.45)

demo = FuncAnimation(fig, animation, interval=100)
plt.show()

