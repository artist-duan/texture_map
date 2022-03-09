import os
import cv2
import trimesh
import argparse
import numpy as np
import open3d as o3d

from texture_map.uv_texture import *
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


def save_texture_mesh(
    vertices,
    triangles,
    triangle_uvs,
    triangle_ids,
    texture_imgs,
    images=None,
    save_path=None,
    name=None,
    save_label=None,
):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    mesh.triangle_material_ids = o3d.utility.IntVector(triangle_ids)
    mesh.textures = texture_imgs

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(save_path, f"{name}.obj"), mesh)
        if images:
            for i, img in enumerate(images):
                img = cv2.imread(img)
                cv2.imwrite(os.path.join(save_path, f"{name}_{i}.png"), img[::-1, :, :])

    if save_label:
        os.makedirs(save_label, exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(save_label, f"{name}.obj"), mesh)
        if images:
            h, w, _ = cv2.imread(images[0]).shape
            for i in range(len(images)):
                color = (
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                )
                color = np.tile(np.array(color).reshape((1, 1, 3)), (h, w, 1)).astype(
                    np.uint8
                )
                cv2.imwrite(
                    os.path.join(save_label, f"{name}_{i}.png"), color[::-1, :, :]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./examples/qingfeng")
    parser.add_argument("--mesh", default="render/mesh.ply")
    parser.add_argument("--depth", default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args()

    """ load mesh """
    # load by open3d
    """
    mesh = o3d.io.read_triangle_mesh(os.path.join(args.path, args.mesh))
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)
    """
    # load by trimesh
    mesh = trimesh.load(os.path.join(args.path, args.mesh))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)
    triangle_normals = np.asarray(mesh.face_normals)

    """ images """
    images = [
        os.path.join(args.path, "image", img)
        for img in os.listdir(os.path.join(args.path, "image"))
        if img.endswith(".png")
    ]
    images.sort()

    """ camera parameters """
    intrinsic, poses = read_poses(
        os.path.join(args.path, "cameras_sphere.npz"), len(images)
    )

    """ depths """
    depth = args.depth
    if not depth:
        depth = "depth"
        depth_path = os.path.join(args.path, depth)
        render_depth(
            images,
            intrinsic,
            poses,
            os.path.join(args.path, args.mesh),
            depth_path,
            display=args.display,
        )

    depths = [
        os.path.join(args.path, depth, img)
        for img in os.listdir(os.path.join(args.path, depth))
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
    """
    triangle_uvs, triangle_ids, texture_imgs = texture_best(visibles, images)
    save_path = os.path.join(args.path, "textures/uv_best")
    save_label = os.path.join(args.path, "textures/uv_best_label") if args.label else None
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
    """

    # max projection area
    """
    triangle_uvs, triangle_ids, texture_imgs = texture_max_area(visibles, images)
    save_path = os.path.join(args.path, "textures/uv_max")
    save_label = os.path.join(args.path, "textures/uv_max_label") if args.label else None
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
    """

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
    save_path = os.path.join(args.path, "textures/uv_graph")
    save_label = (
        os.path.join(args.path, "textures/uv_graph_label") if args.label else None
    )
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
