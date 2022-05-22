import cv2
import numpy as np
import scipy.sparse
import open3d as o3d
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from .utils import *
from .graph_oprimization import *


def find_visible(
    images,
    depths,
    poses,
    intrinsic,
    vertices,
    triangles,
    triangle_normals,
    depth_threshold=5,
):
    """
    triangles: Nx3
    """
    tqdm.write("Finding visible >>>>>>>>>>")

    visibles = [[] for _ in range(triangles.shape[0])]
    vertices = vertices[triangles, :].reshape((-1, 3))  # Nx3x3 -> 3Nx3
    for i in tqdm(range(len(images)), total=len(images)):
        pose, image, depth = poses[i], images[i], depths[i]
        image = cv2.imread(image)
        depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
        H, W, _ = image.shape

        tmp = np.ones((vertices.shape[0], 4), dtype=np.float32)
        tmp[:, :3] = vertices
        cam_vertices = np.matmul(
            intrinsic[:3, :3], np.matmul(pose, tmp.T)[:3]
        ).T  # 3Nx3
        xs, ys, ds = cam_vertices[:, 0], cam_vertices[:, 1], cam_vertices[:, 2]
        us, vs = xs / ds, ys / ds

        camera_normal = np.array([0, 0, 1, 1])
        direct = np.reshape(-1.0 * (np.matmul(pose, camera_normal)[:3]), (-1, 3))
        directs = np.tile(direct, (triangle_normals.shape[0], 1))
        norms1 = np.sqrt(np.sum(directs * directs, axis=-1))
        norms2 = np.sqrt(np.sum(triangle_normals * triangle_normals, axis=-1))
        cos = np.sum(directs * triangle_normals, axis=-1) / (norms1 * norms2)
        angles = np.degrees(np.arccos(cos))

        ds = ds.reshape((-1, 3))
        us, vs = us.reshape((-1, 3)), vs.reshape((-1, 3))
        cond1 = np.logical_and(us[:, 0] < W, us[:, 0] >= 0)
        cond2 = np.logical_and(us[:, 1] < W, us[:, 1] >= 0)
        cond3 = np.logical_and(us[:, 2] < W, us[:, 2] >= 0)
        cond4 = np.logical_and(vs[:, 0] < H, vs[:, 0] >= 0)
        cond5 = np.logical_and(vs[:, 1] < H, vs[:, 1] >= 0)
        cond6 = np.logical_and(vs[:, 2] < H, vs[:, 2] >= 0)
        cond = np.logical_and(
            cond6,
            np.logical_and(
                cond5,
                np.logical_and(
                    cond4, np.logical_and(cond3, np.logical_and(cond1, cond2))
                ),
            ),
        )
        vs, us, ds = vs[cond], us[cond], ds[cond]
        us, vs = us.reshape((-1,)), vs.reshape((-1,))

        index = np.linspace(0, triangles.shape[0] - 1, triangles.shape[0]).astype(
            np.int32
        )
        index = index[cond]
        angles = angles[cond]

        # TODO: will modify to interpolate
        ds_ = depth[vs.astype(np.int32), us.astype(np.int32)]
        us, vs, ds, ds_ = (
            us.reshape((-1, 3)),
            vs.reshape((-1, 3)),
            ds.reshape((-1, 3)),
            ds_.reshape((-1, 3)),
        )  # 3N -> Nx3
        ds *= 1000.0

        cond7 = ds[:, 0] >= 0
        cond8 = ds[:, 1] >= 0
        cond9 = ds[:, 2] >= 0
        cond10 = np.logical_and(
            np.abs(ds[:, 0] - ds_[:, 0]) <= depth_threshold,
            np.abs(ds[:, 1] - ds_[:, 1]) <= depth_threshold,
        )
        cond = np.logical_and(cond10, np.abs(ds[:, 2] - ds_[:, 2]) <= depth_threshold)
        cond = np.logical_and(cond9, cond)
        cond = np.logical_and(cond8, cond)
        cond = np.logical_and(cond7, cond)

        index, us, vs, ds, ds_, angles = (
            index[cond],
            us[cond],
            vs[cond],
            ds[cond],
            ds_[cond],
            angles[cond],
        )

        for j in range(index.shape[0]):
            visibles[index[j]].append([i, us[j], vs[j], angles[j], ds[j], ds_[j]])

    return visibles


def texture_best(visibles, images):
    tqdm.write("Texturing >>>>>>>>>>>>>>>>")

    triangle_uvs = np.zeros((len(visibles), 3, 2), dtype=np.float64)
    triangle_ids = np.zeros((len(visibles),), dtype=np.int32)
    texture_imgs = [
        o3d.geometry.Image((cv2.imread(img)[:, :, ::-1] / 255.0).astype(np.float32))
        for img in images
    ]
    H, W, _ = cv2.imread(images[0]).shape

    def process(i):
        visible = visibles[i]
        if not visible:
            return
        visible.sort(key=lambda x: x[-2].mean())
        index, u, v, angle, d, d_ = visible[0]
        triangle_ids[i] = index
        triangle_uvs[i, 0] = (u[0] / W, v[0] / H)
        triangle_uvs[i, 1] = (u[1] / W, v[1] / H)
        triangle_uvs[i, 2] = (u[2] / W, v[2] / H)

    items = [i for i in range(len(visibles))]
    pool = ThreadPool()
    pool.map(process, items)
    pool.close()
    pool.join()

    triangle_uvs = np.reshape(triangle_uvs, (-1, 2))
    return triangle_uvs, triangle_ids, texture_imgs


def texture_max_area(visibles, images):
    def func(data):
        index, us, vs, angle, d, d_ = data
        return heron_formula(us, vs)

    tqdm.write("Texturing >>>>>>>>>>>>>>>>")

    triangle_uvs = np.zeros((len(visibles), 3, 2), dtype=np.float64)
    triangle_ids = np.zeros((len(visibles),), dtype=np.int32)
    texture_imgs = [
        o3d.geometry.Image((cv2.imread(img)[:, :, ::-1] / 255.0).astype(np.float32))
        for img in images
    ]
    H, W, _ = cv2.imread(images[0]).shape

    def process(i):
        visible = visibles[i]
        if not visible:
            return
        visible.sort(key=func, reverse=True)
        index, u, v, angle, d, d_ = visible[0]
        triangle_ids[i] = index

        triangle_uvs[i, 0] = (u[0] / W, v[0] / H)
        triangle_uvs[i, 1] = (u[1] / W, v[1] / H)
        triangle_uvs[i, 2] = (u[2] / W, v[2] / H)

    items = [i for i in range(len(visibles))]
    pool = ThreadPool()
    pool.map(process, items)
    pool.close()
    pool.join()

    triangle_uvs = np.reshape(triangle_uvs, (-1, 2))
    return triangle_uvs, triangle_ids, texture_imgs


def texture_graph_optimzation(
    intrinsic,
    poses,
    visibles,
    images,
    triangles,
    vertices,
    face_adjacencys,
    softmax_coefficient=1.0,
    adjacency_level=8,
    sample_num=40,
    affinity_coefficient=1.0,
    momenta=0,
    order=3,
    itern=3,
):
    rows, cols, labels, distances, probs, areas = gen_affinity(
        len(images),
        triangles,
        vertices,
        visibles,
        face_adjacencys,
        softmax_coefficient=softmax_coefficient,
        adjacency_level=adjacency_level,
        sample_num=sample_num,
    )

    feat = probs
    affinity = -labels - affinity_coefficient * distances
    affinity = norma(affinity)
    affinity = scipy.sparse.coo_matrix(
        (affinity, (rows, cols)), shape=(len(triangles), len(triangles))
    )
    new_feat = graph_conv_by_affinity(
        feat, affinity, momenta=momenta, order=order, itern=itern, visn=0
    )
    labels = np.argmax(new_feat, axis=-1).astype(np.int32)

    tqdm.write("Texturing >>>>>>>>>>>>>>>>")
    vertices = vertices[triangles, :].reshape((-1, 3))  # Nx3x3 -> 3Nx3
    triangle_uvs = np.zeros((len(labels), 3, 2), dtype=np.float64)
    texture_imgs = [
        o3d.geometry.Image((cv2.imread(img)[:, :, ::-1] / 255.0).astype(np.float32))
        for img in images
    ]
    H, W, _ = cv2.imread(images[0]).shape

    poses = np.array(poses)
    poses = poses[labels]  # N x 4 x 4
    poses = np.expand_dims(poses, axis=1)
    poses = np.tile(poses, (1, 3, 1, 1)).reshape((-1, 4, 4))

    tmp = np.ones((vertices.shape[0], 4), dtype=np.float32)
    tmp[:, :3] = vertices
    tmp = np.reshape(tmp, (-1, 4, 1))

    cam_vertices = np.matmul(
        intrinsic[:3, :3], np.matmul(poses, tmp)[:, :3, 0].T
    ).T  # 3Nx3
    xs, ys, ds = cam_vertices[:, 0], cam_vertices[:, 1], cam_vertices[:, 2]
    us, vs = xs / ds, ys / ds

    triangle_uvs = np.hstack([us.reshape(-1, 1) / W, vs.reshape(-1, 1) / H])
    return triangle_uvs, labels, texture_imgs
