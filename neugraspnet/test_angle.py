import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import time

from neugraspnet.src.utils.perception import TSDFVolume, camera_on_sphere
from neugraspnet.src.utils.transform import Transform

def get_realsense_stream():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    return pipeline, align, depth_scale

def get_frame(pipeline, align, depth_scale):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame:
        return None
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    return depth

class Intrinsics:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.fx = 615.37701416
        self.fy = 615.37701416
        self.cx = 313.68743896
        self.cy = 259.01800537

def integrate_tsdf_from_depth(depth, size=0.3, resolution=80):
    intr = Intrinsics()
    tsdf = TSDFVolume(size, resolution)
    center = Transform.identity()
    center.translation = np.array([0.15, 0.15, 0.0])
    import scipy.spatial.transform as st
    center.rotation = st.Rotation.from_euler('xyz', [-1.57, -1.57, 0.0])
    extrinsic = camera_on_sphere(center, 0.60, 1.13, 0.0)
    print(f"[INFO] Integrating TSDF with extrinsic: {extrinsic.as_matrix()}") 
    tsdf.integrate(depth, intr, extrinsic)
    return tsdf.get_cloud()

def main():
    print("[INFO] Starting RealSense stream...")
    pipeline, align, depth_scale = get_realsense_stream()

    vis = o3d.visualization.Visualizer()
    vis.create_window("TSDF PointCloud Viewer", 640, 480)
    added = False
    pcd = o3d.geometry.PointCloud()

    try:
        while True:
            depth = get_frame(pipeline, align, depth_scale)
            if depth is None:
                print("[WARN] Empty depth frame")
                continue
            
            pc = integrate_tsdf_from_depth(depth)

            if pc.is_empty():
                print("[WARN] Empty point cloud from TSDF")
                continue
            
            # visulize pc
            # o3d.visualization.draw_geometries([pc])

            print(f"[INFO] TSDF voxel count: {len(pc.points)}")

            pcd.points = pc.points
            pcd.colors = pc.colors if pc.has_colors() else o3d.utility.Vector3dVector(np.zeros((len(pc.points), 3)))

            if not added:
                vis.add_geometry(pcd)
                added = True
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.5)

    finally:
        print("[INFO] Stopping RealSense pipeline...")
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()
