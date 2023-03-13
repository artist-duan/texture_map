import os
import numpy as np

from texture_map.utils import read_poses


def gen_npz(save_path, intrinsic, world2cameras):
    """
    intrinsic: 3x3 matrix
    world2cameras: list of camera pose, [T1, T2, ...], T_i -> 4x4 matrix
    """

    tmp = np.eye(4)
    tmp[:3, :3] = intrinsic
    intrinsic = tmp

    datas = {}
    for i in range(len(world2cameras)):
        datas[f"camera_mat_{i}"] = intrinsic
        datas[f"world_mat_{i}"] = intrinsic @ world2cameras[i]

    np.save(os.path.join(save_path, "cameras_sphere.npz"), datas)


if __name__ == "__main__":
    # intrinsic, poses = read_poses("./examples/qingfeng/cameras_sphere.npz", -1)
    # intrinsic = intrinsic[:3,:3]
    # print(intrinsic, len(poses))

    intrinsic = None  # your intrinsic matrix, 3x3
    poses = []  # your extrinsic matrix for each image, 4x4
    gen_npz("./", intrinsic, poses)
