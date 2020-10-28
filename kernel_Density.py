# Import stuff
from cupy import array, asarray, asnumpy, concatenate, zeros, where, unique, bincount, argmax, argpartition, \
argsort, arange, meshgrid, vstack, exp, unravel_index, max as cmax, min as cmin, random as crandom
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from functions import data_load, randomSampling, trip_Distance, Calculate_distance,Calculate_Matrix, myHdbscan, ScaledHdbscan, dbcluster, mat2, pick
#from mpl_toolkits.basemap import Basemap

dclus, pp, dd = data_load()
# # 매칭 매트릭스 계산
# PD = Calculate_Matrix(episode, dclus, pp, dd)
# OD = trip_Distance(episode, pp)
# clusterer, labelNum = ScaledHdbscan(episode, pp, cluster_size = 7)
# DS = dbcluster(episode, pp, clusterer, labelNum, k = 3)q
# matrix = PD * OD * DS

idx = 0
idx2 = 5
train_data = asnumpy(pp[idx][:, :2])
train_data2 = asnumpy(pp[idx2][:, :2])

# Define kernel
kernel = KernelDensity(kernel="gaussian", bandwidth=1)

# Set some parameters for the synthetic data
mean = [0, 0]
cov = [[0.2, 1], [0, 1]]

# Create two data sets with different densities
x1, y1 = train_data[:,0], train_data[:,1]
x2, y2 = train_data2[:,0], train_data2[:,1]
# print(x1, y1, x1.shape, y1.shape)
print(x1.shape, y1.shape, x2.shape, y2.shape)

x1Min = x1.min() 
x1Max = x1.max() 
y1Min = y1.min() 
y1Max = y1.max() 
x2Min = x2.min() 
x2Max = x2.max() 
y2Min = y2.min() 
y2Max = y2.max() 

# Create grid
xgrid = np.arange(x1Min, x1Max, 0.001)
ygrid = np.arange(y1Min, y1Max, 0.001)
xy_coo = np.meshgrid(xgrid, ygrid)
grid = np.array([xy_coo[0].reshape(-1), xy_coo[1].reshape(-1)])

# Prepare data
data1 = np.vstack([x1, y1])
data2 = np.vstack([x2, y2])

# Evaluate density
log_dens1 = kernel.fit(data1.T).score_samples(grid.T)
dens1 = np.exp(log_dens1).reshape([len(xgrid), len(ygrid)])
log_dens2 = kernel.fit(data2.T).score_samples(grid.T)
dens2 = np.exp(log_dens2).reshape([len(xgrid), len(ygrid)])

# # Plot the distributions and densities
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

im1 = ax1.imshow(dens1, extent=[x1Min, x1Max, y1Min, y1Max], origin="lower", vmin=0.8, vmax=1.2)
ax1.scatter(x1, y1, s=1, marker=".")
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("top", size="10%", pad=0.4)
cbar1 = plt.colorbar(im1, cax=cax1, orientation="horizontal", ticks=MultipleLocator(0.02), format="%.2f")

im2 = ax2.imshow(dens2, extent=[x2Min, x2Max, y2Min, y2Max], origin="lower", vmin=0, vmax=0.1)
ax2.scatter(x2, y2, s=1, marker=".")
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("top", size="10%", pad=0.4)
cbar2 = plt.colorbar(im2, cax=cax2, orientation="horizontal", ticks=MultipleLocator(0.02), format="%.2f")

plt.show()