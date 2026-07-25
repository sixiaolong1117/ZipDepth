"""Single image -> depth estimation map.

Usage:
    python img2depth.py photo.jpg
    python img2depth.py photo.jpg --outdir result --colormap viridis
    python img2depth.py photo.jpg --raw   # save raw depth as .npy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
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

    dummy = torch.randn(1, 3, 518, 928, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    return model


@torch.no_grad()
def predict(model, image_path, resolution=518, device="cuda"):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        sys.exit(f"Cannot read image: {image_path}")

    h, w = bgr.shape[:2]
    scale = resolution / min(h, w)
    new_h = int(round(h * scale / 32) * 32)
    new_w = int(round(w * scale / 32) * 32)

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

    depth = model(tensor)
    depth = torch.nn.functional.interpolate(depth, (h, w), mode="bilinear", align_corners=True)
    return depth.squeeze().float().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description="Single image -> depth estimation map")
    p.add_argument("image", help="Input image path")
    p.add_argument("-o", "--outdir", default="output", help="Output directory (default: output)")
    p.add_argument("--resolution", type=int, default=518, help="Processing resolution (default: 518)")
    p.add_argument("--colormap", default="Spectral", help="Matplotlib colormap (default: Spectral)")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    p.add_argument("--raw", action="store_true", help="Also save raw depth as .npy")
    args = p.parse_args()

    img = Path(args.image)
    if not img.is_file():
        p.error(f"{img} not found")

    model = load_model(device=args.device)
    depth = predict(model, img, resolution=args.resolution, device=args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    colored = depth_to_colormap(depth, cmap=args.colormap)
    out_path = outdir / f"{img.stem}_depth.png"
    cv2.imwrite(str(out_path), colored)
    print(f"Saved: {out_path}")

    if args.raw:
        npy_path = outdir / f"{img.stem}_depth.npy"
        np.save(npy_path, depth)
        print(f"Saved raw: {npy_path}")


if __name__ == "__main__":
    main()
