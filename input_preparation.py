"""
This module provides functions to preprocess input data for PPF-FoldNet model.
It includes functions to convert RGBD images to point clouds, 
calculate local normals for point clouds, select referenced points, 
collect local neighbors, and build local patches for each reference point.
"""
import os
import time
import numpy as np
import open3d

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

def rgbd_to_point_cloud(data_dir, ind, downsample=0.03, aligned=True):
    """rgbd image to point cloud"""
    pcd = open3d.io.read_point_cloud(os.path.join(data_dir, f'{ind}.ply'))
    # downsample the point cloud
    if downsample != 0:
        pcd = pcd.voxel_down_sample(voxel_size=downsample)
    # align the point cloud
    if aligned is True:
        matrix = np.load(os.path.join(data_dir, f'{ind}.pose.npy'))
        pcd.transform(matrix)

    return pcd


def cal_local_normal(pcd):
    """calculate local normal for point cloud"""
    if open3d.geometry.estimate_normals(pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
        return True
    else:
        print("Calculate Normal Error")
        return False


def select_referenced_point(pcd, num_patches=2048):
    """A point sampling algorithm for 3d matching of irregular geometries."""
    pts = np.asarray(pcd.points)
    num_points = pts.shape[0]
    inds = np.random.choice(range(num_points), num_patches, replace=False)
    return open3d.geometry.select_down_sample(pcd, inds)


def collect_local_neighbor(ref_pcd, pcd, vicinity=0.3, num_points_per_patch=1024, random_state=None):
    """collect local neighbor within vicinity for each interest point.
    each local patch is downsampled to 1024 (setting of PPFNet p5.)"""
    kdtree = open3d.geometry.KDTreeFlann(pcd)
    dict = []
    for point in ref_pcd.points:
        # Bug fix: here the first returned result will be itself. So the calculated ppf will be nan.
        [k, idx, variant] = kdtree.search_radius_vector_3d(point, vicinity)
        # random select fix number [num_points] of points to form the local patch.
        if random_state is not None:
            if k > num_points_per_patch:
                idx = random_state.choice(idx[1:], num_points_per_patch, replace=False)
            else:
                idx = random_state.choice(idx[1:], num_points_per_patch)
        else:
            if k > num_points_per_patch:
                idx = np.random.choice(idx[1:], num_points_per_patch, replace=False)
            else:
                idx = np.random.choice(idx[1:], num_points_per_patch)
        dict.append(idx)
    return dict


def build_local_patch(ref_pcd, pcd, neighbor):
    """calculate the point pair feature for each patch"""
    num_ref_point = len(ref_pcd.points)
    num_point_per_patch = len(neighbor[0])
    # shape: num_ref_point, num_point_per_patch, 4
    local_patch = np.zeros([num_ref_point, num_point_per_patch, 4], dtype=float)
    # for each reference point
    for j, ref_point, ref_point_normal, inds in zip(range(num_ref_point), ref_pcd.points, ref_pcd.normals, neighbor):
        # for each point in this local patch
        ppfs = _ppf(ref_point, ref_point_normal, np.asarray(pcd.points)[inds], np.asarray(pcd.normals)[inds])
        # origin version: calculate one ppf each time, very SLOW!
        # for i, ind in zip(range(num_point_per_patch), inds):
        #     ppf = _ppf(ref_point, ref_point_normal, pcd.points[ind], pcd.normals[ind])
        #     ppfs[i] = ppf
        local_patch[j] = ppfs
    return local_patch


def _ppf(point1, normal1, point2, normal2):
    """calculate the point pair feature for one pair of points"""
    # origin version: calculate one ppf each time, very SLOW!
    # d = point1 - point2
    # len_d = np.sqrt(np.dot(d, d))
    # dim1 = np.dot(normal1, d) / len_d
    # dim2 = np.dot(normal2, d) / len_d
    # dim3 = np.dot(normal1, normal2)
    # return np.array([dim1, dim2, dim3, len_d])

    d = point1 - point2  # [1024, 3]
    len_d = np.sqrt(np.diag(np.dot(d, d.transpose()))) / 0.3  # [1024, 1]
    # element wise multiply https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html
    y = np.sum(np.multiply(normal1, d), axis=1)
    x = np.linalg.norm(np.cross(normal1, d), axis=1)
    dim1_ = np.arctan2(x, y) / np.pi
    # dim1 = np.arccos(np.sum(np.multiply(normal1, d), axis=1) / len_d) / np.pi  # [1024, 1]
    y = np.sum(np.multiply(normal2, d), axis=1)
    x = np.linalg.norm(np.cross(normal2, d), axis=1)
    dim2_ = np.arctan2(x, y) / np.pi
    # dim2 = np.arccos(np.sum(np.multiply(normal2, d), axis=1) / len_d) / np.pi  # [1024, 1]
    y = np.sum(np.multiply(normal1, normal2), axis=1)
    x = np.linalg.norm(np.cross(normal1, normal2), axis=1)
    dim3_ = np.arctan2(x, y) / np.pi
    # dim3 = np.arccos(np.clip(np.sum(np.multiply(normal1, normal2), axis=1), a_min=-1, a_max=1)) / np.pi
    return np.array([dim1_, dim2_, dim3_, len_d]).transpose()


def input_preprocess(data_dir, local_id, save_dir):
    """rgbd -> point cloud"""
    # start_time = time.time()
    pcd = rgbd_to_point_cloud(data_dir, local_id)
    # print("rgbd->pcd: ", time.time() - start_time)

    # calculate local normal for point cloud
    cal_local_normal(pcd)
    # print("cal normal: ", time.time() - start_time)

    # select referenced points (default number 2048)
    ref_pcd = select_referenced_point(pcd)
    # print("select ref: ", time.time() - start_time)
    # ref_pts = np.asarray(ref_pcd.points)
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
    np.save(f'{save_dir}/{id}.npy', local_patch.astype(np.float32))
    open3d.io.write_point_cloud(f"{save_dir}/{id}.pcd", ref_pcd)


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
    with open("./data/3DMatch/rgbd_fragments/scene_list_train.txt") as f:
        scene_list = f.readlines()
    for scene in scene_list:
        scene = scene.replace("\n", "")
        data_directory = f"./data/3DMatch/rgbd_fragments/{scene}/seq-01"
        start_time = time.time()
        for filename in os.listdir(data_directory):
            fragment_id = filename.split(".")[0]
            get_local_patches_on_the_fly(data_directory, id, 32, 1024)
            print(id)
        print(f"Finish {scene}, time: {time.time() - start_time}")
        break
