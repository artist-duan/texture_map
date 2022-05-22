import os
import argparse

import pymeshfix
import pyvista as pv

# not support obj
def fill_hole_for_ply(mesh_path, save_path=None, display=False):
    m = pv.read(mesh_path)

    meshfix = pymeshfix.MeshFix(m)
    meshfix.repair(verbose=True)
    mesh = meshfix.mesh

    if display:
        mesh.plot()

    if save_path:
        mesh.save(save_path)
        return save_path
    else:
        mesh.save(mesh_path.replace(".ply", "_fix.ply"))
        return mesh_path.replace(".ply", "_fix.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../../../Datasets/qingfeng2")
    parser.add_argument("--mesh", default="render/00300000.ply")
    args = parser.parse_args()

    save_path = fill_hole_for_ply(os.path.join(args.path, args.mesh))
