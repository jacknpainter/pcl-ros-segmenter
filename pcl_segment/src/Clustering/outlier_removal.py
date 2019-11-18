import pcl

cloud = pcl.load("../data/pcd_three_obj.pcd")

fil = cloud.make_statistical_outlier_filter()
fil.set_mean_k(50)
fil.set_std_dev_mul_thresh(1.0)

fil.filter().to_file(b"../data/inliers.pcd")

fil.set_negative(True)
fil.filter().to_file(b"../data/outliers.pcd")
