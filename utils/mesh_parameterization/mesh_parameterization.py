import cv2
import igl
import argparse
import scipy as sp
import numpy as np
import open3d as o3d

import meshplot
from meshplot import plot, subplot, interact

import xatlas
import trimesh


def pixel_unwrap(uv, size, img=None, face=None):
    uv[:, 0] = (
        (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min()) * (size - 1)
    )
    uv[:, 1] = (
        (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min()) * (size - 1)
    )
    if img is not None:
        uv_show = uv.astype(np.int32)
        for i in range(face.shape[0]):
            indexs = face[i]
            pt1, pt2, pt3 = uv_show[indexs[0]], uv_show[indexs[1]], uv_show[indexs[2]]
            cv2.line(img, tuple(pt1), tuple(pt2), (0, 0, 0), 1)
            cv2.line(img, tuple(pt1), tuple(pt3), (0, 0, 0), 1)
            cv2.line(img, tuple(pt2), tuple(pt3), (0, 0, 0), 1)
        return uv, img
    return uv


def load_mesh(path, ratio=1):
    mesh = o3d.io.read_triangle_mesh(path)
    if ratio != 1:
        triangles = np.asarray(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=triangles.shape[0] // ratio
        )
    mesh.compute_triangle_normals()
    return mesh


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", type=str, default="./example/00002500_crop_fix.obj")
#     args = parser.parse_args()

#     size = 2048

#     vertices, faces = igl.read_triangle_mesh(args.path)
#     # mesh = o3d.io.read_triangle_mesh(args.path)
#     # vertices = np.asarray(mesh.vertices)
#     # faces = np.asarray(mesh.triangles)
#     bnd = igl.boundary_loop(faces)

#     print(vertices.shape, faces.shape, bnd.shape)
#     ## Map the boundary to a circle, preserving edge proportions
#     bnd_uv = igl.map_vertices_to_circle(vertices, bnd)
#     ## Harmonic parametrization for the internal vertices
#     uv = igl.harmonic_weights(vertices, faces, bnd, bnd_uv, 1)

#     img = 255*np.ones((size,size,3), dtype = np.uint8)
#     uv_pixel, img = pixel_unwrap(uv, size, img, faces)
#     cv2.imshow('unwrap_before', img)

#     arap = igl.ARAP(vertices, faces, 2, np.zeros(0))
#     uv = arap.solve(np.zeros((0, 0)), uv)

#     img = 255*np.ones((size,size,3), dtype = np.uint8)
#     uv_pixel, img = pixel_unwrap(uv, size, img, faces)
#     cv2.imshow('unwrap_after', img)
#     cv2.imwrite('./unwrap.png', img)
#     cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./example/00002500_crop_fix.obj")
    args = parser.parse_args()

    texture_img_size = 2048

    mesh = load_mesh(args.path, ratio=4)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    faces_normals = np.asarray(mesh.triangle_normals)

    print(vertices.shape, faces.shape, faces_normals.shape)

    vmapping, faces, uvs = xatlas.parametrize(vertices, faces)
    vertices = vertices[vmapping]

    img = 255 * np.ones((texture_img_size, texture_img_size, 3), dtype=np.uint8)
    uv_pixel, img = pixel_unwrap(uvs, texture_img_size, img, faces)
    cv2.imwrite("./unwrap.png", img)

    uv_pixel = uv_pixel[faces.reshape((-1,)), :].astype(np.float32) / texture_img_size
    uv_pixel[:, 1] = 1 - uv_pixel[:, 1]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_pixel)
    # mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((faces.shape[0],), dtype=np.int32))
    texture_imgs = [o3d.geometry.Image((img / 255.0).astype(np.float32))]
    mesh.textures = texture_imgs
    o3d.io.write_triangle_mesh("./test.obj", mesh)


if __name__ == "__main__":
    main()
