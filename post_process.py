import os
import argparse
import open3d as o3d

import pymeshfix
import pyvista as pv

from texture_map.vertex_texture import *
from texture_map.render_color_depth import *
from uv_map_example import texturing as uv_texturing
from vertex_map_example import texturing as vertex_texturing


def crop(path, mesh_name, min_bbox=[-1.0, -1.0, -1.0], max_bbox=[1.0, 1.0, -0.02]):
    min_bbox = np.array(min_bbox).reshape((3, 1))
    max_bbox = np.array(max_bbox).reshape((3, 1))
    mesh = o3d.io.read_triangle_mesh(os.path.join(path, mesh_name))
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bbox, max_bbox)
    mesh = mesh.crop(bbox)
    mesh_name = mesh_name.replace(".ply", "_crop.ply")
    o3d.io.write_triangle_mesh(os.path.join(path, mesh_name), mesh)
    return path, mesh_name


# not support obj
def fill_hole_for_ply(path, mesh_name, save_path=None, display=False):
    m = pv.read(os.path.join(path, mesh_name))
    meshfix = pymeshfix.MeshFix(m)
    meshfix.repair(verbose=True)
    mesh = meshfix.mesh
    if display:
        mesh.plot()
    mesh_name = mesh_name.replace(".ply", "_fix.ply")
    mesh.save(os.path.join(path, mesh_name))
    return path, mesh_name


def render_mask_by_mesh(path, mesh_name, display=False):
    images = [
        os.path.join(path, "image", img)
        for img in os.listdir(os.path.join(path, "image"))
        if img.endswith(".png")
    ]
    images.sort()

    intrinsic, poses = read_poses(os.path.join(path, "cameras_sphere.npz"), len(images))

    depths = render_depth(
        images,
        intrinsic,
        poses,
        os.path.join(path, mesh_name),
        None,
        display=display,
    )

    for i in range(len(images)):
        image = cv2.imread(images[i])
        depth = depths[i]
        depth[depth != 0] = 1
        image[depth == 0] = (255, 255, 255)
        depth = np.expand_dims(255 * depth.astype(np.uint8), axis=-1)
        depth = np.concatenate([depth, depth, depth], -1)
        print(depth.shape, depth.min(), depth.max())
        print(images[i])
        print(images[i].replace("/image/", "/mask/"))

        cv2.imwrite(images[i], image)
        cv2.imwrite(images[i].replace("/image/", "/mask/"), depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../../Datasets/qingfeng2")
    parser.add_argument("--mesh", default="render/00300000.ply")
    parser.add_argument("--crop_mesh", action="store_true")
    parser.add_argument("--fix_hole", action="store_true")
    parser.add_argument("--mask", action="store_true")
    # A4:195mm, A3:285mm
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--texture_map", type=str, default="vertex")
    parser.add_argument("--downsample_mesh", action="store_true")
    parser.add_argument("--downsample_ratio", type=int, default=1)
    parser.add_argument("--downsample_pose", action="store_true")
    parser.add_argument("--downsample_count", type=int, default=20)
    args = parser.parse_args()

    path = args.path
    mesh_name = args.mesh
    if args.crop_mesh:
        min_bbox, max_bbox = [-0.7, -0.7, -1], [0.7, 0.7, -0.02]
        path, mesh_name = crop(path, mesh_name, min_bbox, max_bbox)

    if args.fix_hole:
        path, mesh_name = fill_hole_for_ply(path, mesh_name)

    if args.mask:
        render_mask_by_mesh(path, mesh_name, display=False)

    if args.texture_map == "uv":
        uv_texturing(
            path,
            mesh_name,
            depth=None,
            display=False,
            label=False,
            method="graph_optim",
            downsamplepose=args.downsample_pose,
            downsamplemesh=args.downsample_mesh,
            count=args.downsample_count,
            ratio=args.downsample_ratio,
        )
        mesh_name = "textures/uv_graph"
    elif args.texture_map == "vertex":
        vertex_texturing(path, mesh_name)
        mesh_name = "textures/optim.ply"
    else:
        raise NotImplementedError

    if args.scale is not None:
        if args.texture_map == "uv":
            texture_mesh_path = os.path.join(path, mesh_name)
            images_name = [
                os.path.join(texture_mesh_path, name)
                for name in os.listdir(texture_mesh_path)
                if name.endswith(".png")
            ]
            images_name.sort()
            images = [cv2.imread(name) for name in images_name]
            texture_imgs = [
                o3d.geometry.Image((img[..., ::-1] / 255.0).astype(np.float32))
                for img in images
            ]

            mesh = o3d.io.read_triangle_mesh(
                os.path.join(texture_mesh_path, "color.obj")
            )
            vertices = np.asarray(mesh.vertices) * args.scale / 1000.0 / (2**0.5)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangle_material_ids = o3d.utility.IntVector(
                np.asarray(mesh.triangle_material_ids) - 1
            )
            mesh.textures = texture_imgs

            o3d.io.write_triangle_mesh(
                os.path.join(texture_mesh_path, "color.obj"), mesh
            )

            for name, img in zip(images_name, images):
                cv2.imwrite(name, img)

        elif args.texture_map == "vertex":
            mesh_path = os.path.join(path, mesh_name)
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            vertices = np.asarray(mesh.vertices) * args.scale / 1000.0 / (2**0.5)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d.io.write_triangle_mesh(mesh_path, mesh)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
