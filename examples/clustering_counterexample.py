# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

"""
Example showing how different views of the same data can appear
to show two separate clusterings.
"""

N = np.array([[1,1,1],[1,1.5,1],[1,1.3,5],[1,1.1,6]])
C = np.array([[1,-1,1],[1,-1.5,1],[1,-1.3,5],[1,-1.1,6]])

X = np.vstack([N,C])

x,y,z = X.T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x,y,z)
plt.show()
