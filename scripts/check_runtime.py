"""Check native libraries and optional CUDA/DINOv3 inference without weights."""

import argparse
import importlib.metadata
import json
import platform
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.strtree import STRtree

import build_siaya_full  # noqa: F401
from dinov3.hub.backbones import dinov3_vitl16

MODEL_PIXELS = 512
TOKEN_COUNT = (MODEL_PIXELS // 16) ** 2
FEATURE_DIM = 1024


def check_geospatial():
    """Exercise GDAL and GEOS, including a raster round trip."""
    pixels = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    np.testing.assert_array_equal(torch.from_numpy(pixels).numpy(), pixels)
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", width=16, height=16, count=1, dtype="uint8",
            crs="EPSG:32636", transform=from_origin(600000, 10000, 0.5, 0.5),
        ) as ds:
            ds.write(pixels)
        with mem.open() as ds:
            np.testing.assert_array_equal(ds.read(), pixels)
            assert ds.crs.to_epsg() == 32636

    tree = STRtree([box(0, 0, 1, 1)])
    assert tree.query(box(0.5, 0.5, 2, 2)).tolist() == [0]
    assert "CPUExecutionProvider" in ort.get_available_providers()


def check_gpu():
    """Run real kernels; CUDA discovery alone does not prove compatibility."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run inside a GPU job with --nv")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("This NGC profile requires an A100 or newer GPU")

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(x @ x.T).all().item()
    q = torch.randn(1, 16, TOKEN_COUNT, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(F.scaled_dot_product_attention(q, q, q)).all().item()
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)


def check_backbone(batches, repeats):
    """Measure synthetic ViT-L/16 inference, excluding imagery I/O and PCA."""
    model = dinov3_vitl16(pretrained=False).eval().cuda()
    model.init_weights()
    print("Synthetic inputs and random weights; this is not a quality test.", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in batches:
            x = torch.randn(batch, 3, MODEL_PIXELS, MODEL_PIXELS, device="cuda")
            for _ in range(3):
                model.forward_features(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(repeats):
                output = model.forward_features(x)["x_norm_patchtokens"]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            assert output.shape == (batch, TOKEN_COUNT, FEATURE_DIM)
            assert torch.isfinite(output).all().item()
            print(json.dumps({
                "batch": batch,
                "seconds_per_window": elapsed / (repeats * batch),
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1 or any(batch < 1 for batch in args.batches):
        parser.error("Batches and repeats must be positive")

    print(json.dumps({
        "python": platform.python_version(), "libc": platform.libc_ver(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        **{name: importlib.metadata.version(name) for name in (
            "numpy", "rasterio", "shapely", "onnxruntime", "safetensors",
        )},
    }), flush=True)
    check_geospatial()
    if args.require_gpu or args.benchmark:
        check_gpu()
    if args.benchmark:
        check_backbone(args.batches, args.repeats)
    print("PASS: runtime checks", flush=True)


if __name__ == "__main__":
    main()
