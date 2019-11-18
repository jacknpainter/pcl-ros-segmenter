from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from open3d import *
import numpy as np

data = np.asarray((read_point_cloud("../data/inliers.pcd")).points)

ssd = []
for i in range(1,15):
	k = KMeans(n_clusters=i)
	k = k.fit(data)
	ssd.append(k.inertia_)

graph = plt.figure()
plt.plot(range(1,15),ssd, 'kx-')
plt.xlabel('k')
plt.ylabel('SSD')
graph.savefig("../Results/optimal.png")
