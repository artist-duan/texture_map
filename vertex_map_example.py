import os
import argparse
import numpy as np
import open3d as o3d

from texture_map.vertex_texture import *
from texture_map.render_color_depth import *


def read_poses(path, num):
    data = np.load(path)
    intrinsic = None
    poses = []
    for i in range(num):
        intrinsic = data[f"camera_mat_{i}"]
        world_mat = data[f"world_mat_{i}"]
        pose = np.linalg.inv(intrinsic) @ world_mat
        poses.append(pose)
    return intrinsic, poses


def save_texture_mesh(vertices, colors, triangles, save_path=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, ::-1] / 255.0)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if save_path:
        o3d.io.write_triangle_mesh(save_path, mesh)


def texturing(path, mesh_name, depth=None, display=False):
    """load mesh"""
    mesh = o3d.io.read_triangle_mesh(os.path.join(path, mesh_name))
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_normals = np.asarray(mesh.vertex_normals)

    """ images """
    images = [
        os.path.join(path, "image", img)
        for img in os.listdir(os.path.join(path, "image"))
        if img.endswith(".png")
    ]
    images.sort()

    """ camera parameters """
    intrinsic, poses = read_poses(os.path.join(path, "cameras_sphere.npz"), len(images))

    """ depths """
    if not depth:
        depth = "depth"
        depth_path = os.path.join(path, depth)
        render_depth(
            images,
            intrinsic,
            poses,
            os.path.join(path, mesh_name),
            depth_path,
            display=display,
        )

    depths = [
        os.path.join(path, depth, img)
        for img in os.listdir(os.path.join(path, depth))
        if img.endswith(".png")
    ]
    depths.sort()

    os.makedirs(os.path.join(path, "textures"), exist_ok=True)

    """ texture vertex """
    # # min depth: don't need depth
    # colors = texture_min_depth(
    #     vertices, vertex_normals, intrinsic, poses, images, display=args.display
    # )
    # save_path = os.path.join(path, "textures", "min_depth.obj")
    # save_texture_mesh(vertices, colors, triangles, save_path=save_path)

    # visibles = find_visible(
    #     images, depths, poses, intrinsic, vertices, vertex_normals, depth_threshold=5
    # )

    # # best normal and camera direction angle: need depth
    # colors = texture_normal_best(visibles)
    # save_path = os.path.join(path, "textures", "best.obj")
    # save_texture_mesh(vertices, colors, triangles, save_path=save_path)

    # # mean all: need depth
    # colors = texture_normal_mean(visibles)
    # save_path = os.path.join(path, "textures", "mean.obj")
    # save_texture_mesh(vertices, colors, triangles, save_path=save_path)

    # # weight mean all: need depth
    # colors = texture_normal_weight_mean(visibles)
    # save_path = os.path.join(path, "textures", "weight_mean.obj")
    # save_texture_mesh(vertices, colors, triangles, save_path=save_path)

    # optimzation in open3d
    mesh = texture_optimzation(mesh, images, depths, intrinsic, poses, iteration=50)
    save_path = os.path.join(path, "textures", "optim.ply")
    o3d.io.write_triangle_mesh(save_path, mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./examples/qingfeng")
    parser.add_argument("--mesh", default="render/mesh.ply")
    parser.add_argument("--depth", default=None)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    texturing(args.path, args.mesh, args.depth, args.display)
