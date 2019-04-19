import open3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt


def rgbd_to_point_cloud(data_dir, ind, show=False):
    color_raw = open3d.read_image(f"{data_dir}/{ind}.color.png")
    depth_raw = open3d.read_image(f"{data_dir}/{ind}.depth.png")
    rgbd_image = open3d.create_rgbd_image_from_color_and_depth(color_raw, depth_raw, depth_trunc=10)
    # print(rgbd_image)
    intrinstic = open3d.camera.PinholeCameraIntrinsic()
    prev_dir = data_dir.replace("/seq-01","")
    with open(os.path.join(prev_dir, "camera-intrinsics.txt")) as f:
        content = f.readlines()
    fx = float(content[0].split("\t")[0]) / 1
    fy = float(content[1].split("\t")[1]) / 1
    cx = float(content[0].split("\t")[2]) / 1
    cy = float(content[1].split("\t")[2]) / 1
    intrinstic.set_intrinsics(640, 480, fx, fy, cx, cy)
    matrix = np.loadtxt(f"{data_dir}/{ind}.pose.txt")
    pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinstic, extrinsic=matrix)
    if show:
        open3d.draw_geometries([pcd])
    # pts = np.asarray(pcd.points)
    return pcd


def cal_local_normal(pcd):
    if open3d.geometry.estimate_normals(pcd):
        return True
    else:
        print("Calculate Normal Error")
        return False


def select_referenced_point(pcd, num_patches=2048):
    # A point sampling algorithm for 3d matching of irregular geometries.
    pts = np.asarray(pcd.points)
    num_points = pts.shape[0]
    inds = np.random.choice(range(num_points), num_patches, replace=False)
    return open3d.geometry.select_down_sample(pcd, inds)


def collect_local_neighbor(ref_pcd, pcd, vicinity=0.3, num_points_per_patch=1024):
    # collect local neighbor within vicinity for each interest point.
    # each local patch is downsampled to 1024 (setting of PPFNet p5.)
    kdtree = open3d.geometry.KDTreeFlann(pcd)
    dict = []
    for point in ref_pcd.points:
        # Bug fix: here the first returned result will be itself. So the calculated ppf will be nan.
        [k, idx, variant] = kdtree.search_radius_vector_3d(point, vicinity)
        # random select fix number [num_points] of points to form the local patch.
        if k > num_points_per_patch:
            idx = np.random.choice(idx[1:], num_points_per_patch, replace=False)
        else:
            idx = np.random.choice(idx[1:], num_points_per_patch)
        dict.append(idx)
    return dict


def build_local_patch(ref_pcd, pcd, neighbor):
    num_ref_point = len(ref_pcd.points)
    num_point_per_patch = len(neighbor[0])
    # shape: num_ref_point, num_point_per_patch, 4
    local_patch = np.zeros([num_ref_point, num_point_per_patch, 4], dtype=float)
    # for each reference point
    for j, ref_point, ref_point_normal, inds in zip(range(num_ref_point), ref_pcd.points, ref_pcd.normals, neighbor):
        ppfs = np.zeros([num_point_per_patch, 4])
        # for each point in this local patch
        ppfs = _ppf(ref_point, ref_point_normal, np.asarray(pcd.points)[inds], np.asarray(pcd.normals)[inds])
        # origin version: calculate one ppf each time, very SLOW!
        # for i, ind in zip(range(num_point_per_patch), inds):
        #     ppf = _ppf(ref_point, ref_point_normal, pcd.points[ind], pcd.normals[ind])
        #     ppfs[i] = ppf
        local_patch[j] = ppfs
    return local_patch


def _ppf(point1, normal1, point2, normal2):
    # origin version: calculate one ppf each time, very SLOW!
    # d = point1 - point2
    # len_d = np.sqrt(np.dot(d, d))
    # dim1 = np.dot(normal1, d) / len_d
    # dim2 = np.dot(normal2, d) / len_d
    # dim3 = np.dot(normal1, normal2)
    # return np.array([dim1, dim2, dim3, len_d])

    d = point1 - point2  # [1024, 3]
    len_d = np.sqrt(np.diag(np.dot(d, d.transpose())))  # [1024, 1]
    # element wise multiply https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html
    dim1 = np.sum(np.multiply(normal1, d), axis=1) / len_d  # [1024, 1]
    dim2 = np.sum(np.multiply(normal2, d), axis=1) / len_d  # [1024, 1]
    dim3 = np.sum(np.multiply(normal1, normal2), axis=1)
    return np.array([dim1, dim2, dim3, len_d]).transpose()


def input_preprocess(data_dir, id, save_dir):
    # rgbd -> point cloud
    # start_time = time.time()
    pcd = rgbd_to_point_cloud(data_dir, id)
    # print("rgbd->pcd: ", time.time() - start_time)

    # calculate local normal for point cloud
    cal_local_normal(pcd)
    # print("cal normal: ", time.time() - start_time)

    # select referenced points (default number 2048)
    ref_pcd = select_referenced_point(pcd)
    # print("select ref: ", time.time() - start_time)
    ref_pts = np.asarray(ref_pcd.points)
    # assert ref_pts.shape[0] == 2048

    # collect local patch for each reference point
    neighbor = collect_local_neighbor(ref_pcd, pcd)
    # print("collect neighbor:", time.time() - start_time)
    # assert len(neighbor) == ref_pts.shape[0]

    # calculate the point pair feature for each patch
    local_patch = build_local_patch(ref_pcd, pcd, neighbor)
    # print("cal ppf for each pair:", time.time() - start_time)

    # save the local_patch and reference point cloud for one point cloud fragment.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(f'{save_dir}/{id}.npy', local_patch)
    open3d.write_point_cloud(f"{save_dir}/{id}.pcd", ref_pcd)


def get_local_patches_on_the_fly(data_dir, ind, num_patches, num_points_per_patch=1024):
    """
    similar function with input_preprocess, on-the-fly select the local patch
    """
    pcd = rgbd_to_point_cloud(data_dir, ind)
    cal_local_normal(pcd)
    ref_pcd = select_referenced_point(pcd, num_patches)
    neighbor = collect_local_neighbor(ref_pcd, pcd, num_points_per_patch=num_points_per_patch)
    local_patch = build_local_patch(ref_pcd, pcd, neighbor)
    return local_patch


if __name__ == "__main__":
    # data_dir = "./data/train/sun3d-hotel_umd-maryland_hotel3/seq-01"
    with open("/data/3DMatch/whole/scene_list_train.txt") as f:
        scene_list = f.readlines()
    for scene in scene_list:
        scene = scene.replace("\n", "")
        data_dir = f"/data/3DMatch/whole/{scene}/seq-01"
        start_time = time.time()
        for filename in os.listdir(data_dir):
            if filename.__contains__('color'):
                id = filename.split(".")[0]
                input_preprocess(data_dir, id, data_dir + '-preprocessed/')
                # get_local_patches_on_the_fly(data_dir, id, 32, 1024)
                print(id)
        print(f"Finish {scene}, time: {time.time() - start_time}")
        break

