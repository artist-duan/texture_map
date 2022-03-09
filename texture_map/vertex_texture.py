import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


def texture_min_depth(points, normals, intrinsic, poses, imgs, display=True):
    colors = np.zeros((points.shape[0], 3)).astype(np.float32)
    depths = float("inf") * np.ones((points.shape[0],))
    tmp = np.ones((points.shape[0], 4))
    tmp[:, :3] = points
    points = tmp.T

    for pose, img in zip(poses, imgs):
        img = cv2.imread(img)
        H, W, _ = img.shape
        cam_points = np.matmul(intrinsic[:3, :3], np.matmul(pose, points)[:3]).T
        u, v, d = (
            cam_points[:, 0] / cam_points[:, 2],
            cam_points[:, 1] / cam_points[:, 2],
            cam_points[:, 2],
        )
        cond1 = np.logical_and(0 <= u, u <= W - 1)
        cond2 = np.logical_and(0 <= v, v <= H - 1)
        cond = np.logical_and(np.logical_and(cond1, cond2), d > 0)
        cond = np.logical_and(cond, depths > d)
        u, v = u.astype(np.int), v.astype(np.int)

        depths[cond] = d[cond]
        colors[cond, :] = img[v[cond], u[cond], :]

        show = np.zeros((H, W, 3))
        show[v[cond], u[cond], :] = img[v[cond], u[cond], :]

        if display:
            cv2.imshow("project", show.astype(np.uint8))
            cv2.waitKey(10)

    return colors


def find_visible(images, depths, poses, intrinsic, points, normals, depth_threshold=5):
    tqdm.write("Finding visible >>>>>>>>>>")

    visibles = [[] for _ in range(points.shape[0])]
    for i in tqdm(range(len(images)), total=len(images)):
        pose, image, depth = poses[i], images[i], depths[i]
        image = cv2.imread(image)
        depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
        H, W, _ = image.shape

        tmp = np.ones((points.shape[0], 4), dtype=np.float32)
        tmp[:, :3] = points
        cam_points = np.matmul(intrinsic[:3, :3], np.matmul(pose, tmp.T)[:3, :]).T
        xs, ys, ds = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
        us, vs = xs / ds, ys / ds

        camera_normal = np.array([0, 0, 1, 1])
        direct = np.reshape(-1.0 * (np.matmul(pose, camera_normal)[:3]), (-1, 3))
        directs = np.tile(direct, (points.shape[0], 1))
        norms1 = np.sqrt(np.sum(directs * directs, axis=-1))
        norms2 = np.sqrt(np.sum(normals * normals, axis=-1))
        cos = np.sum(directs * normals, axis=-1) / (norms1 * norms2)
        angles = np.degrees(np.arccos(cos))

        # TODO: will modify to interpolate
        us, vs = us.astype(np.int32), vs.astype(np.int32)
        ds_ = depth[vs, us]

        us, vs, ds, ds_ = (
            us.reshape((-1,)),
            vs.reshape((-1,)),
            ds.reshape((-1,)),
            ds_.reshape((-1,)),
        )  # 3N -> Nx3
        ds *= 1000.0

        cond1 = np.logical_and(us < W, us >= 0)
        cond2 = np.logical_and(vs < H, vs >= 0)
        cond = np.logical_and(np.abs(ds - ds_) <= depth_threshold, cond2)
        cond = np.logical_and(cond, cond1)

        index = np.linspace(0, points.shape[0] - 1, points.shape[0]).astype(np.int32)
        index, us, vs, ds, ds_, angles = (
            index[cond],
            us[cond],
            vs[cond],
            ds[cond],
            ds_[cond],
            angles[cond],
        )

        for j in range(index.shape[0]):
            visibles[index[j]].append(
                [i, us[j], vs[j], image[vs[j], us[j]], angles[j], ds[j] - ds_[j]]
            )

    return visibles


def texture_normal_best(visibles):
    tqdm.write("Texturing >>>>>>>>>>>>>>>>")

    items = [i for i in range(len(visibles))]
    colors = np.zeros((len(visibles), 3)).astype(np.float32)

    def process(i):
        visible = visibles[i]
        if not visible:
            return
        visible.sort(key=lambda x: np.abs(x[-2]))
        index, u, v, c, angle, d = visible[0]
        colors[i] = c

    pool = ThreadPool()
    pool.map(process, items)
    pool.close()
    pool.join()
    return colors


def texture_normal_mean(visibles):
    tqdm.write("Texturing >>>>>>>>>>>>>>>>")

    items = [i for i in range(len(visibles))]
    colors = np.zeros((len(visibles), 3)).astype(np.float32)

    def process(i):
        visible = visibles[i]
        if not visible:
            return
        color, count = np.zeros((3,), dtype=np.float32), 0
        for vis in visible:
            index, u, v, c, angle, d = vis
            color += c
            count += 1
        colors[i] = 1.0 * color / count

    pool = ThreadPool()
    pool.map(process, items)
    pool.close()
    pool.join()
    return colors


def texture_normal_weight_mean(visibles):
    tqdm.write("Texturing >>>>>>>>>>>>>>>>")

    items = [i for i in range(len(visibles))]
    colors = np.zeros((len(visibles), 3)).astype(np.float32)

    def process(i):
        visible = visibles[i]
        if not len(visible):
            return
        cs, weights = [], []
        for vis in visible:
            index, u, v, c, angle, d = vis
            if np.isnan(angle):
                continue
            cs.append(c)
            weights.append(np.cos(angle / 180.0 * np.pi) + 1)
        if not len(cs):
            return
        cs = np.array(cs).reshape((-1, 3))
        weights = np.array(weights)
        weights /= weights.sum()
        weights = np.tile(weights.reshape(-1, 1), (1, cs.shape[1]))
        colors[i] = (weights * cs).sum(axis=0)

    pool = ThreadPool()
    pool.map(process, items)
    pool.close()
    pool.join()
    return colors


def texture_optimzation(mesh, images, depths, intrinsic, poses, iteration=10):
    img = cv2.imread(images[0])
    H, W, _ = img.shape

    parameters = []
    rgbd_images = []
    intrinsic_raw = intrinsic.copy()
    for i in range(len(images)):
        depth = o3d.io.read_image(depths[i])
        color = o3d.io.read_image(images[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False
        )
        rgbd_images.append(rgbd_image)

        pose = poses[i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W,
            H,
            intrinsic_raw[0, 0],
            intrinsic_raw[1, 1],
            intrinsic_raw[0, 2],
            intrinsic_raw[1, 2],
        )
        parameter = o3d.camera.PinholeCameraParameters()
        parameter.extrinsic = pose
        parameter.intrinsic = intrinsic
        parameters.append(parameter)

    camera_trajectory = o3d.camera.PinholeCameraTrajectory()
    camera_trajectory.parameters = parameters

    # Run rigid optimization.
    # maximum_iteration = iteration
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, camera_trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
    #         mesh, rgbd_images, camera_trajectory,
    #         o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=maximum_iteration))

    # Run non-rigid optimization.
    maximum_iteration = iteration
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
            mesh,
            rgbd_images,
            camera_trajectory,
            o3d.pipelines.color_map.NonRigidOptimizerOption(
                maximum_iteration=maximum_iteration
            ),
        )

    return mesh
