"""Real-time webcam depth estimation using ZipDepth.

Usage:
    python cam_depth.py
    python cam_depth.py --camera 1
    python cam_depth.py --resolution 384 --colormap inferno
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import time

import cv2
import numpy as np
import torch

from zipdepth import create_model
from zipdepth.utils.model_utils import strip_state_dict_prefixes, fuse_remaining_conv_bn
from zipdepth.utils.colormap import depth_to_colormap


def load_model(device="cuda"):
    model = create_model(variant="base", upsample_unfold=True)
    ckpt = torch.load(
        Path(__file__).parent / "checkpoints" / "zipdepth_base.pth",
        map_location="cpu", weights_only=True,
    )
    sd = ckpt.get("model_state_dict", ckpt)
    sd = strip_state_dict_prefixes(sd)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    model.fuse_for_inference()
    fuse_remaining_conv_bn(model)

    dummy = torch.randn(1, 3, 384, 640, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    return model


@torch.no_grad()
def infer(model, frame, resolution=518, device="cuda"):
    h, w = frame.shape[:2]
    scale = resolution / min(h, w)
    new_h = int(round(h * scale / 32) * 32)
    new_w = int(round(w * scale / 32) * 32)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

    depth = model(tensor)
    depth = torch.nn.functional.interpolate(depth, (h, w), mode="bilinear", align_corners=True)
    return depth.squeeze().float().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description="Real-time webcam depth estimation")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--resolution", type=int, default=518, help="Processing resolution (default: 518)")
    p.add_argument("--colormap", default="Spectral", help="Matplotlib colormap (default: Spectral)")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    p.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    p.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    args = p.parse_args()

    print(f"Loading ZipDepth-base ...")
    model = load_model(device=args.device)
    print("Model loaded.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        p.error(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Press 'q' to quit, 's' to save screenshot.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t0 = time.time()
        depth = infer(model, frame_bgr, resolution=args.resolution, device=args.device)
        depth_vis = depth_to_colormap(depth, cmap=args.colormap)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(depth_vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        h, w = frame_bgr.shape[:2]
        depth_vis = cv2.resize(depth_vis, (w, h))
        combined = np.hstack([frame_bgr, depth_vis])
        cv2.imshow("Original | Depth (press q to quit, s to save)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("depth_screenshot.png", combined)
            print("Saved depth_screenshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
