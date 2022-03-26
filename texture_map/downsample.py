import trimesh
import numpy as np
import open3d as o3d

from .utils import *


def downsample_mesh(path, ratio=2):
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=triangles.shape[0] // ratio
    )
    mesh.compute_triangle_normals()
    # o3d.visualization.draw_geometries([mesh])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=triangles, face_normals=triangle_normals
    )
    # mesh.show()
    return mesh


def downsample_pose(poses, count=20):
    count = min(count, len(poses) // 2)
    poses = np.array([np.linalg.inv(pose) for pose in poses])

    first = np.random.randint(0, len(poses))
    indexs = [int(first)]

    for i in range(count - 1):
        tmp = poses[indexs][:, :3, 3]
        dis, index = -1, -1
        for j in range(len(poses)):
            if j in indexs:
                continue
            pos = poses[j][:3, 3].reshape((-1, 3))
            pos = np.tile(pos, [len(indexs), 1])
            d = np.min((pos - tmp) ** 2)
            if d > dis:
                dis = d
                index = j
        indexs.append(index)

    indexs.sort()
    return indexs


if __name__ == "__main__":
    from utils import *

    # intrinsic, poses = read_poses("./cameras_sphere1.npz", 17)
    # indexs = downsample_pose(poses)
    # mesh = pose_camera(poses[indexs], 0.08, [1, 0, 0])
    # o3d.io.write_triangle_mesh(os.path.join('./tmp.ply'), mesh)

    downsample_mesh("./mesh.ply")
