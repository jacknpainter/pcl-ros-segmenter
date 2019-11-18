import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
from mpl_toolkits.mplot3d.axes3d import Axes3D
from open3d import *

#imports pcd file and sets as numpy array
pc = read_point_cloud("../data/inliers.pcd")
inliers = np.asarray(pc.points)

#creates pyplot for Axes3D function to write to
plot = plt.figure()
ax = Axes3D(plot)

#performs K-Means Algorithm on point cloud data
centroid, label = kmeans2(whiten(inliers), 3)

x = np.array(inliers[:,0])
y = np.array(inliers[:,1])
z = np.array(inliers[:,2])

#creates scatter graph of cloud data, assigning colours to each segmented piece
ax.scatter(x, y, z, c=label)

#saves pyplot as file
plot.savefig("../Results/inliers_segmented.png")
