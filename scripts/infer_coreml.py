"""
ZipDepth CoreML inference — single image, folder, or video.

Usage:
    # Single image
    python scripts/infer_coreml.py --model checkpoints/zipdepth_base_384x384.mlpackage --input image.jpg

    # Folder
    python scripts/infer_coreml.py --model checkpoints/zipdepth_base_384x384.mlpackage --input /path/to/images/ --output /path/to/output/

    # Video
    python scripts/infer_coreml.py --model checkpoints/zipdepth_base_384x384.mlpackage --input video.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage

try:
    import coremltools as ct
except ImportError:
    ct = None


COMPUTE_MAP = {
    'all':         ct.ComputeUnit.ALL if ct else None,
    'cpu_only':    ct.ComputeUnit.CPU_ONLY if ct else None,
    'cpu_and_gpu': ct.ComputeUnit.CPU_AND_GPU if ct else None,
    'cpu_and_ne':  ct.ComputeUnit.CPU_AND_NE if ct else None,
}


def make_divisible(value: float, divisor: int) -> int:
    return max(divisor, int(round(value / divisor) * divisor))


def depth_to_colormap(depth: np.ndarray, cmap: str = 'Spectral') -> np.ndarray:
    import matplotlib.pyplot as plt
    norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    cm = plt.get_cmap(cmap)
    colored = cm(norm)[:, :, :3]
    return (colored[:, :, ::-1] * 255).astype(np.uint8)


def _timed(unit, label, times_list):
    """Decorator to time a method and record elapsed ms."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000
            times_list.append(ms)
            return result
        return wrapper
    return deco


class CoreMLDepthInference:
    def __init__(
        self,
        model_path: str,
        input_size: int = 384,
        compute_unit: str = 'all',
        warmup_iters: int = 3,
    ):
        if ct is None:
            raise ImportError("coremltools is required. pip install coremltools")

        cu = COMPUTE_MAP.get(compute_unit, ct.ComputeUnit.ALL)
        print(f"Loading CoreML model: {model_path}")
        print(f"  Compute unit: {compute_unit}")
        self.model = ct.models.MLModel(model_path, compute_units=cu)
        self.input_size = input_size
        self.input_name = 'image'
        self.compiled = False
        self.warmup_iters = warmup_iters

        spec = self.model.get_spec()
        desc = spec.description
        print(f"  Input:  {desc.input[0].name}")
        print(f"  Output: {desc.output[0].name}")

        if warmup_iters > 0:
            self._warmup()

    def _warmup(self):
        dummy = PILImage.fromarray(
            np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8))
        for _ in range(self.warmup_iters):
            self.model.predict({self.input_name: dummy})
        self.compiled = True
        print(f"  Warmup: {self.warmup_iters} iters — ready")

    def preprocess(self, bgr_image: np.ndarray):
        h, w = bgr_image.shape[:2]
        resized = cv2.resize(bgr_image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1]
        return rgb, h, w

    def infer(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb, orig_h, orig_w = self.preprocess(bgr_image)
        pil_img = PILImage.fromarray(rgb)

        preds = self.model.predict({self.input_name: pil_img})
        depth_key = list(preds.keys())[0]
        depth = preds[depth_key]

        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]
        elif depth.ndim != 2:
            depth = depth.squeeze()

        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def _bench_steady_state(self, num_iters: int = 100) -> dict:
        dummy = PILImage.fromarray(
            np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8))
        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            self.model.predict({self.input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        return {
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'fps': 1000.0 / float(arr.mean()),
        }

    def predict_image(self, image_path: str, output_path: str = None, save_raw: bool = False):
        raw = cv2.imread(str(image_path))
        if raw is None:
            raise ValueError(f"Cannot load: {image_path}")
        h, w = raw.shape[:2]

        print(f"\nProcessing {Path(image_path).name}")
        print(f"  Input:  {w}×{h}  →  model input: {self.input_size}×{self.input_size}")

        t_total = time.time()
        depth = self.infer(raw)
        total_ms = (time.time() - t_total) * 1000

        depth_range = f"[{depth.min():.3f}, {depth.max():.3f}]"
        compile_note = " (includes ~28ms first-run JIT compilation)" if not self.compiled else ""
        print(f"  End-to-end: {total_ms:.1f} ms  →  {1000/total_ms:.0f} FPS{compile_note}")
        print(f"  Depth range:  {depth_range}")

        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_depth.jpg")

        colored = depth_to_colormap(depth)
        cv2.imwrite(output_path, colored)
        print(f"  Saved to: {output_path}")

        if save_raw:
            raw_path = Path(output_path).with_suffix('.npy')
            np.save(raw_path, depth)
            print(f"  Raw saved to: {raw_path}")

        return depth

    def predict_camera(
        self,
        camera_id: int = 0,
        display_height: int = 480,
        snapshot_dir: str = None,
        save_raw: bool = False,
    ):
        """Real-time camera depth estimation with live display."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera #{camera_id}. "
                               "Grant camera permission in System Settings → Privacy → Camera.")

        # Read first frame to determine camera resolution
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Camera opened but failed to capture first frame")
        cam_h, cam_w = frame.shape[:2]

        scale = display_height / cam_h
        disp_w = int(cam_w * scale)
        disp_h = display_height

        print(f"\nCamera #{camera_id}: {cam_w}×{cam_h} @ ~30 fps")
        print(f"  Display:  {disp_w}×{disp_h}  (scale {scale:.2f})")
        print(f"  Model:    {self.input_size}×{self.input_size}")
        print("  Controls: [q/ESC] quit  [s] save snapshot")
        print("  ⚠ macOS: ensure Terminal has camera permission (System Settings → Privacy → Camera)")

        if snapshot_dir is None:
            snapshot_dir = Path.cwd() / 'camera_snapshots'
        snap_path = Path(snapshot_dir)
        snap_path.mkdir(parents=True, exist_ok=True)

        window_name = 'ZipDepth — Camera [L] | Depth [R]'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_w * 2 + 40, disp_h + 60)

        times = []
        snap_idx = 0
        running = True

        while running:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed — stopping")
                break

            t0 = time.perf_counter()
            depth = self.infer(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            # Build side-by-side display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_disp = cv2.resize(frame_rgb, (disp_w, disp_h))

            depth_colored = depth_to_colormap(
                cv2.resize(depth, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR))
            depth_disp = cv2.resize(depth_colored, (disp_w, disp_h))

            combined = np.hstack([frame_disp, depth_disp])

            # FPS overlay
            if len(times) > 1:
                recent = np.mean(times[-30:])
                fps = 1000.0 / recent
                label = f"FPS: {fps:.0f}  |  {recent:.1f} ms"
            else:
                label = "warming up..."
            cv2.putText(combined, label, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            # ── recording info for live FPS ──
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("  User quit")
                running = False
                break
            elif key == ord('s'):
                snap_name = f"snap_{snap_idx:04d}"
                snap_frame = snap_path / f"{snap_name}.jpg"
                snap_depth = snap_path / f"{snap_name}_depth.jpg"
                snap_raw = snap_path / f"{snap_name}_depth.npy"
                cv2.imwrite(str(snap_frame), frame)
                cv2.imwrite(str(snap_depth), depth_colored)
                if save_raw:
                    np.save(str(snap_raw), depth)
                print(f"  Snapshot saved: {snap_frame}")
                snap_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        avg_ms = float(np.mean(times)) if times else 0
        print(f"\nSession stats:")
        print(f"  Frames processed: {len(times)}")
        print(f"  Avg end-to-end:   {avg_ms:.1f} ms  →  {1000/avg_ms:.0f} FPS")
        print(f"  Snapshots:        {snap_idx}")

    def predict_batch(
        self,
        input_dir: str,
        output_dir: str = None,
        save_raw: bool = False,
        colorize: bool = True,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp'),
    ):
        input_path = Path(input_dir)
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        image_files = sorted(set(image_files))

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images")
        if output_dir is None:
            output_dir = input_path / 'depth_output'
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        times = []
        failed = []
        pbar = tqdm(image_files, desc="Processing")
        for img_file in pbar:
            try:
                raw = cv2.imread(str(img_file))
                if raw is None:
                    raise ValueError("Cannot load image")
                t0 = time.perf_counter()
                depth = self.infer(raw)
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed)
                pbar.set_postfix({'ms': f'{elapsed:.1f}'})

                out_file = output_path / f"{img_file.stem}_depth.jpg"
                if colorize:
                    colored = depth_to_colormap(depth)
                    cv2.imwrite(str(out_file), colored)
                if save_raw:
                    np.save(str(out_file.with_suffix('.npy')), depth)
            except Exception as e:
                failed.append((img_file.name, str(e)))
        pbar.close()

        avg_ms = float(np.mean(times)) if times else 0
        print(f"\n{'='*50}")
        print(f"Processed: {len(times)}/{len(image_files)}")
        print(f"Failed:    {len(failed)}")
        print(f"Avg end-to-end:  {avg_ms:.1f} ms  →  {1000/avg_ms:.0f} FPS")
        print(f"Output: {output_path}")
        if failed:
            for name, err in failed[:5]:
                print(f"  x {name}: {err}")

    def predict_video(
        self,
        video_path: str,
        output_path: str = None,
        max_frames: int = None,
        output_height: int = None,
        save_raw: bool = False,
        colorize: bool = True,
    ):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        if output_height is not None:
            scale = output_height / height
            out_h = output_height
            out_w = make_divisible(width * scale, 2)
        else:
            out_h, out_w = height, width

        print(f"Video: {Path(video_path).name}")
        print(f"  Original:  {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"  Model:     {self.input_size}x{self.input_size}  →  output: {out_w}x{out_h}")

        if output_path is None:
            output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_depth.mp4")

        frames_dir = Path(output_path).parent / f"{Path(video_path).stem}_depth_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        times = []
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Inference")
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            t0 = time.perf_counter()
            depth = self.infer(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            pbar.set_postfix({'ms': f'{elapsed:.1f}'})

            if colorize:
                frame_resized = cv2.resize(frame, (out_w, out_h))
                depth_colored = depth_to_colormap(cv2.resize(depth, (out_w, out_h)))
                combined = np.hstack([frame_resized, depth_colored])
                cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), combined,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            if save_raw:
                np.save(str(frames_dir / f"{frame_idx:06d}.npy"),
                        cv2.resize(depth, (out_w, out_h)))

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        avg_ms = float(np.mean(times)) if times else 0
        print(f"\nFrames: {frame_idx}")
        print(f"  Avg end-to-end:  {avg_ms:.1f} ms  →  {1000/avg_ms:.0f} FPS")

        if colorize:
            frame_files = sorted(frames_dir.glob("*.jpg"))
            if frame_files:
                print(f"\nEncoding video ({len(frame_files)} frames) ...")
                first = cv2.imread(str(frame_files[0]))
                enc_h, enc_w = first.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (enc_w, enc_h))
                out.write(first)
                for fpath in tqdm(frame_files[1:], desc="Encoding"):
                    out.write(cv2.imread(str(fpath)))
                out.release()
                print(f"Output: {output_path}")

        print(f"Frames: {frames_dir}")


def cmd_benchmark(model_path: str, input_size: int = 384, num_iters: int = 100):
    """Compare performance across all compute units."""
    print(f"{'Compute Unit':<18} {'Mean(ms)':>10} {'Min(ms)':>10} {'Std(ms)':>10} {'FPS':>10}")
    print("-" * 58)
    results = {}
    for name, cu in COMPUTE_MAP.items():
        if cu is None:
            continue
        engine = CoreMLDepthInference(
            model_path=model_path,
            input_size=input_size,
            compute_unit=name,
            warmup_iters=3,
        )
        stats = engine._bench_steady_state(num_iters)
        results[name] = stats
        note = " ⚡ ANE" if name == 'cpu_and_ne' else (" ⚡ GPU" if 'gpu' in name else "")
        print(f"{name:<18} {stats['mean']:>8.2f}   {stats['min']:>8.2f}   {stats['std']:>8.2f}   {stats['fps']:>8.0f}{note}")
    return results


def main():
    parser = argparse.ArgumentParser(description='ZipDepth CoreML Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to .mlpackage')
    parser.add_argument('--input', type=str, default=None, help='Image, folder, or video')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--input-size', type=int, default=384, help='Model input size')
    parser.add_argument('--compute-unit', type=str, default='all',
                        choices=list(COMPUTE_MAP.keys()),
                        help='Compute unit for CoreML (default: all)')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations (0=skip)')
    parser.add_argument('--bench', action='store_true',
                        help='Run benchmark across all compute units instead of inference')
    parser.add_argument('--bench-iters', type=int, default=100,
                        help='Iterations for benchmark (default: 100)')
    parser.add_argument('--save-raw', action='store_true', help='Save raw .npy depth')
    parser.add_argument('--no-colormap', action='store_true', help='Skip colorized output')
    parser.add_argument('--max-frames', type=int, default=None, help='Max video frames')
    parser.add_argument('--output-size', type=int, default=None, help='Output video height')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
                        help='Image extensions for folder mode')
    parser.add_argument('--camera', action='store_true',
                        help='Real-time camera depth estimation (instead of --input)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--camera-res', type=int, default=480,
                        help='Display window height in pixels (default: 480)')
    args = parser.parse_args()

    if args.bench:
        cmd_benchmark(args.model, args.input_size, args.bench_iters)
        return

    engine = CoreMLDepthInference(
        model_path=args.model,
        input_size=args.input_size,
        compute_unit=args.compute_unit,
        warmup_iters=args.warmup,
    )

    if args.camera:
        engine.predict_camera(
            camera_id=args.camera_id,
            display_height=args.camera_res,
            save_raw=args.save_raw,
        )
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    if input_path.is_dir():
        engine.predict_batch(
            input_dir=str(input_path),
            output_dir=args.output,
            save_raw=args.save_raw,
            colorize=not args.no_colormap,
            extensions=tuple(args.extensions),
        )
    elif input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        engine.predict_video(
            video_path=str(input_path),
            output_path=args.output,
            max_frames=args.max_frames,
            output_height=args.output_size,
            save_raw=args.save_raw,
            colorize=not args.no_colormap,
        )
    else:
        engine.predict_image(
            image_path=str(input_path),
            output_path=args.output,
            save_raw=args.save_raw,
        )


if __name__ == '__main__':
    main()
