#!/bin/python3


import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


fig = plt.figure()
axis = fig.add_subplot(projection="3d")

def animation(time):

	axis.clear()

	first_grid, second_grid = np.meshgrid(

		np.linspace(-100, 100, 100),
		np.linspace(-100, 100, 100)

		)

	function = np.sin(np.sqrt((first_grid ** 2) * time + (second_grid ** 2) * time)) / np.sqrt((first_grid ** 2) * time + (second_grid ** 2) * time)

	axis.plot_surface(first_grid, second_grid, function, cmap="coolwarm", alpha=0.34)
	axis.scatter(first_grid, second_grid, function, c=function, cmap="binary", s=0.12)

demo = FuncAnimation(fig, animation, interval=10)
plt.show()
