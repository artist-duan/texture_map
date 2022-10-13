import os
import cv2
import trimesh
import pyrender
import numpy as np


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.astype(np.uint8)
    x = cv2.applyColorMap(x, cmap)
    return x


def render_depth(images, intrinsics, poses, mesh, depth_path=None, display=True):
    """
    images: list[PATH_TO_RGB]
    intrinsics: 3x3
    poses: Nx4x4
    mesh: PATH_TO_MESH
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    img = cv2.imread(images[0])
    H, W, _ = img.shape

    mesh = trimesh.load(mesh)
    obj_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    mesh = pyrender.Mesh.from_trimesh(mesh)
    render = pyrender.OffscreenRenderer(W, H)
    camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=0.05, zfar=10000.0)
    light = pyrender.DirectionalLight(color=0.5 * np.ones(3), intensity=30)

    if depth_path is not None:
        os.makedirs(depth_path, exist_ok=True)
    depths = []
    for i in range(len(poses)):
        extrinsics = poses[i]

        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.3], bg_color=[1, 1, 1])
        scene.add(camera, pose=np.eye(4))
        scene.add(mesh, pose=np.matmul(obj_pose, extrinsics))
        scene.add(light, pose=np.eye(4))

        image, depth = render.render(scene)
        depth = depth * 1000.0
        show_depth = visualize_depth(depth)
        depths.append(depth)
        img = images[i].split("/")[-1].split(".")[0]
        if depth_path is not None:
            cv2.imwrite(os.path.join(depth_path, img + ".png"), depth.astype(np.uint16))
        if display:
            cv2.imshow(
                "color",
                np.concatenate(
                    [image[::2, ::2, ::-1], show_depth[::2, ::2, :]], axis=1
                ),
            )
            cv2.waitKey(10)
    if depth_path is None:
        return depths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render Color and Depth")
    parser.add_argument("--path", default="../../../Datasets/cola")
    parser.add_argument("--mesh", default="render/00300000.ply")
    parser.add_argument("--camera", default="cameras_sphere.npz")
    parser.add_argument("--images", default="image")
    parser.add_argument("--save_depth", default="depth")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    path = os.path.join(args.path, "full") if args.full else args.path

    datas = np.load(os.path.join(path, args.camera))

    intrinsics = datas["camera_mat_0"].astype(np.float32)
    images = [os.path.join(path, args.images, name) for name in os.listdir(os.path.join(path, args.images))]
    poses = []
    for i in range(len(images)):
        pose = datas[f"world_mat_{i}"]
        pose = np.linalg.inv(intrinsics) @ pose
        poses.append(pose)
    mesh = os.path.join(args.path, args.mesh)
    depth_path = os.path.join(args.path, args.save_depth)
    render_depth(images, intrinsics, poses, mesh, depth_path)
