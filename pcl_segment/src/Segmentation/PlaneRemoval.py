import numpy as np
from open3d import *
import pcl

points = read_point_cloud("../data/pcd_three_obj.pcd")
draw_geometries([points])
x = np.asarray(points)
print(x)
points = pcl.load("../data/pcd_three_obj.pcd")

cloud = pcl.PointCloud(np.asarray((points), dtype = np.float32))

seg = cloud.make_segmenter_normals(ksearch=50)
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_normal_distance_weight(0.05)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_max_iterations(100)
seg.set_distance_threshold(0.005)
inliers, model = seg.segment()

cloud_plane = cloud.extract(inliers, negative = False)
pcl.save(cloud_plane, "../Results/plane_segment.pcd")

segment = read_point_cloud('../Results/plane_segment.pcd')
draw_geometries([segment])
