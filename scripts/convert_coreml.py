"""
ZipDepth PyTorch → CoreML conversion.

Usage:
    python scripts/convert_coreml.py --ckpt checkpoints/zipdepth_base_npu.pth
    python scripts/convert_coreml.py --ckpt checkpoints/zipdepth_base_npu.pth --output checkpoints/zipdepth.mlpackage
"""

import argparse
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import coremltools as ct

from zipdepth.model.architecture import create_model
from zipdepth.utils.model_utils import strip_state_dict_prefixes, fuse_remaining_conv_bn


def _make_static(model, size: int):
    """Replace dynamic ops with static-shape equivalents."""
    s16 = (size // 16, size // 16)

    # GlobalContextBlock — uses B, C, H, W = x.shape
    for m in model.modules():
        if type(m).__name__ == 'GlobalContextBlock':
            def _fwd(self_m, x, _h=s16[0], _w=s16[1]):
                ctx = F.avg_pool2d(x, kernel_size=(_h, _w))
                return x + self_m.transform(ctx)
            m.forward = types.MethodType(_fwd, m)

    # MinimalCrossScale — uses x_high.shape[2:] / x_low.shape[2:]
    cs = model.encoder.cross_scale
    def _cs_fwd(self_cs, x_high, x_low, _s=s16):
        lo = F.interpolate(self_cs.low_to_high(x_low), size=_s, mode='nearest')
        hi = F.avg_pool2d(self_cs.high_to_low(x_high), 2, 2)
        return x_high + lo * 0.3, x_low + hi * 0.3
    cs.forward = types.MethodType(_cs_fwd, cs)

    # Top-level ZipDepth.forward — has H, W = x.shape[2:]
    orig_fwd = model.forward
    def _model_fwd(self_m, x):
        x_norm = (x - self_m.mean) / self_m.std
        s_half, enc_feats = self_m.encoder(x_norm)
        return self_m.decoder(s_half, enc_feats, (size, size))
    model.forward = types.MethodType(_model_fwd, model)


def main():
    parser = argparse.ArgumentParser(description='ZipDepth PyTorch → CoreML')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output .mlpackage path')
    parser.add_argument('--variant', type=str, default='base', choices=['small', 'base', 'large', 'giant'])
    parser.add_argument('--global-mode', type=str, default='balanced', choices=['none', 'balanced', 'full'])
    parser.add_argument('--image-size', type=int, default=384, help='Input image size')
    parser.add_argument('--compute-unit', type=str, default='all',
                        choices=['all', 'cpu_only', 'cpu_and_gpu', 'cpu_and_ne'],
                        help='Compute units for CoreML (default: all)')
    parser.add_argument('--compute-precision', type=str, default=None,
                        choices=['fp16', 'fp32'],
                        help='Precision for compute. fp16 enables ANE path on Apple Silicon')
    parser.add_argument('--minimum-deployment-target', type=str, default=None,
                        help='Minimum deployment target (e.g. iOS15, macOS13)')
    args = parser.parse_args()

    compute_map = {
        'all': ct.ComputeUnit.ALL,
        'cpu_only': ct.ComputeUnit.CPU_ONLY,
        'cpu_and_gpu': ct.ComputeUnit.CPU_AND_GPU,
        'cpu_and_ne': ct.ComputeUnit.CPU_AND_NE,
    }
    compute_unit = compute_map[args.compute_unit]

    output_path = args.output or str(Path(args.ckpt).with_suffix('.mlpackage'))
    size = args.image_size

    print(f"Loading ZipDepth-{args.variant} ...")
    model = create_model(variant=args.variant, global_mode=args.global_mode, upsample_unfold=False)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = strip_state_dict_prefixes(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()
    model.fuse_for_inference()
    fuse_remaining_conv_bn(model)

    _make_static(model, size)

    dummy = torch.randn(1, 3, size, size)
    traced = torch.jit.trace(model, dummy)

    precision_str = f" {args.compute_precision.upper()}" if args.compute_precision else ""
    print(f"Converting → CoreML{precision_str} ({size}x{size}) ...")
    compute_precision = None
    if args.compute_precision == 'fp16':
        compute_precision = ct.precision.FLOAT16
    elif args.compute_precision == 'fp32':
        compute_precision = ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, size, size),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )],
        compute_units=compute_unit,
        compute_precision=compute_precision,
        minimum_deployment_target=(
            getattr(ct.target, args.minimum_deployment_target)
            if args.minimum_deployment_target else None
        ),
        convert_to="mlprogram",
    )

    mlmodel.save(output_path)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"Done → {output_path}  ({size_mb:.1f} MB)")

    # Verify no aten::Int ops remain
    for n in traced.graph.nodes():
        if n.kind() == 'aten::Int':
            print(f"  ⚠ aten::Int still present: {n}")


if __name__ == '__main__':
    main()
