"""
ZipDepth depth -> colored 3D point cloud (image or live camera).

Usage:
    # Single image, interactive drag-view window
    venv312/bin/python scripts/depth_to_pointcloud.py \
        --model checkpoints/zipdepth_base_384x384_fp16.mlpackage --input assets/examples/im0.jpg

    # Export only, no window (works without open3d installed)
    venv312/bin/python scripts/depth_to_pointcloud.py \
        --model checkpoints/zipdepth_base_384x384_fp16.mlpackage \
        --input assets/examples/im0.jpg --save out.ply --no-show

    # Live camera, drag-view updates every frame (S = snapshot, Q / close window = quit)
    venv312/bin/python scripts/depth_to_pointcloud.py \
        --model checkpoints/zipdepth_base_384x384_fp16.mlpackage --camera

Note: ZipDepth predicts *relative* depth (scale/shift ambiguous), so the
point cloud shape is correct but absolute meters are not. Tune --z-scale /
--fov / --invert if it looks stretched or inside-out.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_coreml import CoreMLDepthInference  # noqa: E402


def points_from_rgbd(rgb, depth, fov=60.0, stride=2,
                     pmin=2.0, pmax=98.0, z_scale=1.0, invert=False):
    """Unproject an RGB-D pair to a colored point cloud (pinhole model).

    Args:
        rgb: [H, W, 3] uint8 RGB at the same resolution as depth.
        depth: [H, W] float32 relative depth from ZipDepth.
        fov: vertical field of view in degrees; fx = fy = (H/2)/tan(fov/2).
        stride: keep every Nth pixel (stride=2 -> ~1/4 points).
        pmin/pmax: percentile clip range to drop far/near outliers.
        z_scale: stretch factor on Z (relative depth has no metric scale).
        invert: flip near/far if the cloud looks inside-out.

    Returns:
        points: [N, 3] float32 (x right, y up, z forward).
        colors: [N, 3] float32 in [0, 1].
    """
    assert rgb.shape[:2] == depth.shape, f"{rgb.shape[:2]} vs {depth.shape}"
    h, w = depth.shape

    lo, hi = np.percentile(depth, [pmin, pmax])
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    d = np.clip(depth, lo, hi)
    if invert:
        d = hi - d + lo  # swap near/far, keep range positive

    z_full = d * z_scale
    f = (h / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    cx, cy = w / 2.0, h / 2.0

    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = z_full[vs, us].astype(np.float32)
    x = ((us - cx) * z / f).astype(np.float32)
    y = (-(vs - cy) * z / f).astype(np.float32)  # image-y down -> 3D-y up

    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    colors = (rgb[vs, us].reshape(-1, 3).astype(np.float32) / 255.0)
    return points, colors


def save_ply(path, points, colors):
    """Write ascii PLY without any 3D dependency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        # ponytail: python loop over ~100k verts (~1s); numpy savetxt if this matters
        for (x, y, z), (r, g, b) in zip(points, rgb_u8):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print(f"Saved: {path} ({len(points)} points)")


def show_pointcloud(points, colors):
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required for the viewer. "
                          "pip install open3d  (or use --save out.ply --no-show)")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.visualization.draw_geometries(
        [pcd], window_name="ZipDepth — drag to rotate, wheel to zoom")


def run_image(engine, args):
    bgr = cv2.imread(args.input)
    if bgr is None:
        raise ValueError(f"Cannot load: {args.input}")
    t0 = time.perf_counter()
    depth = engine.infer(bgr)
    print(f"Inference: {(time.perf_counter() - t0) * 1000:.1f} ms  "
          f"depth range [{depth.min():.3f}, {depth.max():.3f}]")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    points, colors = points_from_rgbd(
        rgb, depth, fov=args.fov, stride=args.stride,
        pmin=args.depth_min_p, pmax=args.depth_max_p,
        z_scale=args.z_scale, invert=args.invert)
    print(f"Points: {len(points)}  "
          f"x[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] "
          f"y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] "
          f"z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    out = args.save or str(Path(args.input).parent / f"{Path(args.input).stem}.ply")
    if args.save or not args.no_show:
        save_ply(out, points, colors)
    if not args.no_show:
        show_pointcloud(points, colors)


def run_camera(engine, args):
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required for live view. pip install open3d")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera #{args.camera_id}. "
            "1) Grant camera permission to your terminal app "
            "(System Settings → Privacy & Security → Camera), then fully "
            "quit and reopen that app. "
            "2) If permission is granted, try --camera-id 1.")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Camera opened but failed to capture first frame")

    snap_dir = Path.cwd() / "camera_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    state = {"idx": 0, "pcd": None, "points": None, "colors": None}

    def _on_snapshot(vis):
        p = state["points"]
        if p is None:
            return False
        save_ply(snap_dir / f"cloud_{state['idx']:04d}.ply", p, state["colors"])
        state["idx"] += 1
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="ZipDepth live cloud — drag to rotate, S = snapshot, Q = quit")
    first = True
    print("Controls: drag = rotate, wheel = zoom, right-drag = pan, [S] snapshot, [Q] quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed — stopping")
                break
            t0 = time.perf_counter()
            depth = engine.infer(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            points, colors = points_from_rgbd(
                rgb, depth, fov=args.fov, stride=args.stride,
                pmin=args.depth_min_p, pmax=args.depth_max_p,
                z_scale=args.z_scale, invert=args.invert)
            ms = (time.perf_counter() - t0) * 1000
            print(f"\r{len(points)} pts  {ms:.0f} ms  ({1000 / ms:.0f} FPS end-to-end)",
                  end="", flush=True)
            state["points"], state["colors"] = points, colors
            if first:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                vis.add_geometry(pcd)
                vis.register_key_callback(ord("S"), _on_snapshot)
                state["pcd"] = pcd
                first = False
            else:
                state["pcd"].points = o3d.utility.Vector3dVector(points.astype(np.float64))
                state["pcd"].colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                vis.update_geometry(state["pcd"])
            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        print()
        cap.release()
        vis.destroy_window()
        print(f"Snapshots: {state['idx']} in {snap_dir}")


def main():
    p = argparse.ArgumentParser(description="ZipDepth depth -> 3D point cloud")
    p.add_argument("--model", required=True, help="Path to .mlpackage")
    p.add_argument("--input", default=None, help="Image file (not needed with --camera)")
    p.add_argument("--camera", action="store_true", help="Live camera mode")
    p.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    p.add_argument("--input-size", type=int, default=384)
    p.add_argument("--compute-unit", default="all",
                   choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--fov", type=float, default=60.0, help="Vertical FOV in degrees")
    p.add_argument("--stride", type=int, default=2, help="Keep every Nth pixel")
    p.add_argument("--depth-min-p", type=float, default=2.0, help="Lower percentile clip")
    p.add_argument("--depth-max-p", type=float, default=98.0, help="Upper percentile clip")
    p.add_argument("--z-scale", type=float, default=1.0, help="Stretch factor on Z")
    p.add_argument("--invert", action="store_true", help="Flip near/far")
    p.add_argument("--save", default=None, help="Output .ply path (image mode)")
    p.add_argument("--no-show", action="store_true", help="Skip interactive viewer")
    args = p.parse_args()

    if not args.camera and not args.input:
        p.error("provide --input image.jpg or --camera")
    if args.stride < 1:
        p.error("--stride must be >= 1")

    engine = CoreMLDepthInference(
        model_path=args.model, input_size=args.input_size,
        compute_unit=args.compute_unit, warmup_iters=args.warmup)
    if args.camera:
        run_camera(engine, args)
    else:
        run_image(engine, args)


if __name__ == "__main__":
    main()
