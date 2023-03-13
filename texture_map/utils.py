import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def read_poses(path, num=-1):
    data = np.load(path)
    intrinsic = None
    poses = []

    if num == -1:
        num = 0
        while f"camera_mat_{num}" in data:
            num += 1

    for i in range(num):
        intrinsic = data[f"camera_mat_{i}"]
        world_mat = data[f"world_mat_{i}"]
        pose = np.linalg.inv(intrinsic) @ world_mat
        poses.append(pose)
    poses = np.array(poses)
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


def heron_formula(us, vs):
    a = ((us[0] - us[1]) ** 2 + (vs[0] - vs[1]) ** 2) ** (1 / 2)
    b = ((us[0] - us[2]) ** 2 + (vs[0] - vs[2]) ** 2) ** (1 / 2)
    c = ((us[1] - us[2]) ** 2 + (vs[1] - vs[2]) ** 2) ** (1 / 2)
    p = (a + b + c) / 2
    s = max((p * (p - a) * (p - b) * (p - c)), 0) ** (1 / 2)
    return s


def softmax(data, coefficient=1.0):
    data -= np.max(data)
    data = np.exp(coefficient * data)
    data = data / np.sum(data)
    return data


def KLDivergence(p, q):
    return -np.sum(p * np.log(q + 1e-5)) + np.sum(p * np.log(p + 1e-5))


def gen_affinity(
    n_images,
    triangles,
    vertices,
    visibles,
    face_adjacencys,
    softmax_coefficient=1.0,
    adjacency_level=1,
    sample_num=40,
):
    tqdm.write("Affinity >>>>>>>>>>>>>>>>")

    probs, areas = [], []
    for i in tqdm(range(len(visibles)), total=len(visibles)):
        visible = visibles[i]
        prob = np.zeros((n_images,), dtype=np.float32)
        if not visible:
            areas.append(prob)
            prob = softmax(prob, coefficient=softmax_coefficient)
            probs.append(prob)
            continue

        for vis in visible:
            index, u, v, angle, d, d_ = vis
            prob[index] = heron_formula(u, v)
        areas.append(prob)
        prob = softmax(prob, coefficient=softmax_coefficient)
        probs.append(prob)

    rows, cols, labels, distances = [], [], [], []
    for i in tqdm(range(len(visibles)), total=len(visibles)):
        adjacencys = []
        fas = face_adjacencys[i]
        adjacencys += fas
        for j in range(adjacency_level):
            tmp = []
            for fa in fas:
                tmp += face_adjacencys[fa]
            adjacencys += tmp
            fas = tmp
        p1, p2, p3 = triangles[i]
        c = (vertices[p1] + vertices[p3] + vertices[p2]) / 3.0
        adjacencys = list(set(adjacencys))
        if len(adjacencys) > sample_num:
            adjacencys = np.random.choice(adjacencys, sample_num)

        for adj in adjacencys:
            p1, p2, p3 = triangles[adj]
            c1 = (vertices[p1] + vertices[p3] + vertices[p2]) / 3.0
            dis = (((c - c1) ** 2).sum()) ** (1 / 2)

            rows.append(i)
            cols.append(adj)
            labels.append(KLDivergence(probs[i], probs[adj]))
            distances.append(dis)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    distances = np.array(distances, dtype=np.float32)
    areas = np.array(areas, dtype=np.float32)
    probs = np.array(probs, dtype=np.float32)
    return rows, cols, labels, distances, probs, areas


def gen_camera(size=100, color=[1, 0, 0]):
    verts = [
        [0, 0, 0],
        [3 * size / 4, size / 2, size],
        [3 * size / 4, -size / 2, size],
        [-3 * size / 4, -size / 2, size],
        [-3 * size / 4, size / 2, size],
    ]
    verts = np.array(verts, dtype=np.float64)
    colors = np.array([color for _ in range(verts.shape[0])])

    triangles = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [4, 3, 2], [4, 2, 1]]
    triangles = np.array(triangles, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def pose_camera(Ts, size=100, color=[1, 0, 0]):
    M = None
    for i, T in enumerate(Ts):
        if i == 0:
            mesh = gen_camera(size, [0, 1, 0])
        elif i == len(Ts) - 1:
            mesh = gen_camera(size, [0, 0, 1])
        elif i == 10:
            mesh = gen_camera(size, [0, 0, 1])
        else:
            mesh = gen_camera(size, color)
        mesh.transform(T)
        if M is None:
            M = mesh
        else:
            M += mesh
    return M
