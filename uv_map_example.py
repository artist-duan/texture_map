import os
import cv2
import trimesh
import argparse
import numpy as np
import open3d as o3d

from texture_map.utils import *
from texture_map.downsample import *
from texture_map.uv_texture import *
from texture_map.render_color_depth import *


def texturing(
    path,
    mesh_name,
    depth=None,
    display=False,
    label=False,
    method="graph_optim",
    downsamplepose=True,
    downsamplemesh=False,
    ratio=2,
):
    """load mesh"""
    if downsamplemesh:
        mesh = downsample_mesh(os.path.join(path, mesh_name), ratio=ratio)
    else:
        # load by open3d
        """
        mesh = o3d.io.read_triangle_mesh(os.path.join(path, mesh_name))
        mesh.compute_triangle_normals()
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        triangle_normals = np.asarray(mesh.triangle_normals)
        """
        mesh = trimesh.load(os.path.join(path, mesh_name))
    # mesh.show()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)
    triangle_normals = np.asarray(mesh.face_normals)

    """ images """
    images = [
        os.path.join(path, "image", img)
        for img in os.listdir(os.path.join(path, "image"))
        if img.endswith(".png")
    ]
    images.sort()

    """ camera parameters """
    intrinsic, poses = read_poses(os.path.join(path, "cameras_sphere.npz"), len(images))

    """ downsample """
    if downsamplepose:
        indexs = downsample_pose(poses)
        poses = poses[indexs]
        images = [images[i] for i in indexs]
        images.sort()

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

    """ adjacencys """
    face_adjacencys = [[] for _ in range(len(triangles))]
    for fa in mesh.face_adjacency:
        f1, f2 = fa
        face_adjacencys[f1].append(f2)
        face_adjacencys[f2].append(f1)

    """ find visible """
    visibles = find_visible(
        images,
        depths,
        poses,
        intrinsic,
        vertices,
        triangles,
        triangle_normals,
        depth_threshold=5,
    )

    """ uv map """
    # best view
    if method == "best_biew":
        triangle_uvs, triangle_ids, texture_imgs = texture_best(visibles, images)
        save_path = os.path.join(path, "textures/uv_best")
        save_label = os.path.join(path, "textures/uv_best_label") if label else None

    # max projection area
    if method == "max_projection_area":
        triangle_uvs, triangle_ids, texture_imgs = texture_max_area(visibles, images)
        save_path = os.path.join(path, "textures/uv_max")
        save_label = os.path.join(path, "textures/uv_max_label") if label else None

    # graph optim
    if method == "graph_optim":
        triangle_uvs, triangle_ids, texture_imgs = texture_graph_optimzation(
            intrinsic,
            poses,
            visibles,
            images,
            triangles,
            vertices,
            face_adjacencys,
            softmax_coefficient=1.0,
            adjacency_level=8,
            sample_num=100,
            affinity_coefficient=1.0,
            momenta=0,
            order=1,
            itern=3,
        )
        save_path = os.path.join(path, "textures/uv_graph")
        save_label = os.path.join(path, "textures/uv_graph_label") if label else None

    del visibles
    del mesh
    save_texture_mesh(
        vertices,
        triangles,
        triangle_uvs,
        triangle_ids,
        texture_imgs,
        images,
        save_path,
        name="color",
        save_label=save_label,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./examples/qingfeng")
    parser.add_argument("--mesh", default="render/mesh.ply")
    parser.add_argument("--depth", default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--method", default="graph_optim")
    parser.add_argument("--downsample_pose", action="store_true")
    parser.add_argument("--downsample_mesh", action="store_true")
    parser.add_argument("--ratio", default=2)
    args = parser.parse_args()

    texturing(
        args.path,
        args.mesh,
        args.depth,
        args.display,
        args.label,
        args.method,
        downsamplepose=args.downsample_pose,
        downsamplemesh=args.downsample_mesh,
        ratio=args.ratio,
    )
