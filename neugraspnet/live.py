import argparse
import time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs # type: ignore
from types import SimpleNamespace

from neugraspnet.src.experiments.detection_implicit import NeuGraspImplicit
from neugraspnet.src.utils.perception import TSDFVolume
from neugraspnet.src.utils.transform import Transform

class FakeGripper:
    def __init__(self):
        self.max_opening_width = 0.8
        self.finger_depth = 0.5

class FakeSim:
    def __init__(self):
        self.gripper = FakeGripper()

def get_aligned_realsense_frames():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    # Warm-up
    for _ in range(10):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

    pipeline.stop()

    fx, fy = 615.37701416, 615.37701416
    cx, cy = 313.68743896, 259.01800537

    return color, depth, fx, fy, cx, cy

def integrate_tsdf(depth, fx, fy, cx, cy, size=0.3, resolution=128, down_sample_pc=0.0075):
    from neugraspnet.src.utils.perception import camera_on_sphere
    from neugraspnet.src.utils.transform import Transform

    tsdf = TSDFVolume(size, resolution)

    class Intrinsics:
        def __init__(self, width, height, fx, fy, cx, cy):
            self.width = width
            self.height = height
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy

    intr = Intrinsics(640, 480, fx, fy, cx, cy)
    center = Transform.identity()
    center.translation = np.array([0.15, 0.15, 0.0])
    import scipy.spatial.transform as st
    center.rotation = st.Rotation.from_euler('xyz', [-1.57, -1.57, 0.0])
    extrinsic = camera_on_sphere(center, 0.60, 1.13, 0.0)

    tsdf.integrate(depth, intr, extrinsic)
    pc = tsdf.get_cloud()
    pc_down = pc.voxel_down_sample(voxel_size=down_sample_pc)

    print(f"[INFO] TSDF voxel count: {len(pc.points)}")
    return tsdf, pc_down

def create_grasp_mesh(grasp):
    from neugraspnet.src.utils.visual import grasp2mesh
    mesh = grasp2mesh(grasp)
    return mesh

def trimesh_to_open3d(tri_mesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    if hasattr(tri_mesh.visual, 'vertex_colors') and len(tri_mesh.visual.vertex_colors) > 0:
        mesh.vertex_colors = o3d.utility.Vector3dVector(tri_mesh.visual.vertex_colors[:, :3] / 255.0)
    mesh.compute_vertex_normals()
    return mesh

def main(args):
    grasp_planner = NeuGraspImplicit(
        model_path=args.model,
        model_type=args.type,
        best=True,
        qual_th=0.5,
        force_detection=True,
        seen_pc_only=True,
        out_th=0.1,
        select_top=False,
        resolution=64,
        visualize=False,
        max_grasp_queries_at_once=40
    )

    color, depth, fx, fy, cx, cy = get_aligned_realsense_frames()

    tsdf, pc = integrate_tsdf(depth, fx, fy, cx, cy)
    

    if pc.is_empty():
        print("[ERROR] Empty point cloud! Adjust camera angle or depth.")
        return

    state = SimpleNamespace(tsdf=tsdf, pc=pc)
    grasps, scores, unseen_flags, _ = grasp_planner(state, sim=FakeSim())
    print(f"[INFO] Grasps found: {len(grasps)}")
    print(grasps[0].pose.as_matrix())
    print(f"[INFO] Average width: {np.mean([grasp.width for grasp in grasps])}")
    print(f"[INFO] Max width: {np.max([grasp.width for grasp in grasps])}")
    print(f"[INFO] Min width: {np.min([grasp.width for grasp in grasps])}")
    print(f"[INFO] Average score: {np.mean(scores)}")
    print(f"[INFO] Max score: {np.max(scores)}")
    print(f"[INFO] Min score: {np.min(scores)}")
    
    
    vis = o3d.visualization.Visualizer()
    vis.create_window("Grasp Visualization", width=1280, height=720)
    vis.add_geometry(pc)

    for grasp in grasps:
        g_trimesh = create_grasp_mesh(grasp)
        g_mesh_o3d = trimesh_to_open3d(g_trimesh)
        vis.add_geometry(g_mesh_o3d)

    vis.poll_events()
    vis.update_renderer()
    while True:
        vis.poll_events()
        vis.update_renderer()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default="./data/networks/neugraspnet_pile_efficient.pt", type=str)
    parser.add_argument("--type",default="neu_grasp_pn_deeper_efficient", type=str)
    args = parser.parse_args()
    main(args)