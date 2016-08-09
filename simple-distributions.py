#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib.mlab as mlab

np.random.seed(0)

X = np.arange(-3, 3, 0.001)
Y = norm.pdf(X, 0, 1)

fig = plt.figure()
plt.plot(X, Y)
plt.tight_layout()
plt.savefig("normal-pdf.png")

nSamples = 35
X = np.random.normal(0, 1, nSamples)
Y = np.zeros(nSamples)
fig = plt.figure(figsize=(7,3))
plt.scatter(X, Y, color='k')
plt.xlim((-3,3))
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
plt.savefig("normal-samples.png")

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)

nSamples = 200
mean = [0, 0]
cov = [[1,0], [0,1]]
X, Y = np.random.multivariate_normal(mean, cov, nSamples).T
plt.scatter(X, Y, color='k')

plt.savefig("normal-2d.png")
