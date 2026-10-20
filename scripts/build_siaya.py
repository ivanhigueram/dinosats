"""Build a bounded, fully covered pre/endline Vantor pilot for the local DinoSats map.

Reads existing RGB VRTs without changing source data. The manifest has bounds in
EPSG:32636 and two sources with epoch, catalog_id, date, sensor, rgb and clouds.
Cloud files are Vantor GeoJSON in EPSG:4326. This pilot deliberately rejects gaps
and mapped clouds; it is not a whole-strip mosaicker or a cloud-shadow detector.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import rasterio.shutil
import torch
import torch.nn.functional as F
from dinotxt import load_dinotxt
from embed import (
    FEATURE_DIM,
    TEMPLATES,
    TOKEN_STRIDE,
    embed,
    fit_pca,
    positional_basis,
    text_features,
    write_cog,
    write_float32,
    write_posnorm,
)
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds

CRS = "EPSG:32636"
PIXEL_SIZE = 0.5
CLOUD_MARGIN_M = 100
MAX_SIZE = 2048
SEED = 42
QUANTILE = 0.999

BACKGROUND = ["trees", "grass", "cropland", "bare soil", "road", "water", "shadow"]
QUERIES = [
    "houses",
    "metal roofs",
    "thatched roofs",
    "buildings",
    "dirt road",
    "farm fields",
    "dense vegetation",
    "bare ground",
    "construction site",
]
VOCAB = (
    BACKGROUND
    + QUERIES
    + [
        "small houses",
        "roof",
        "courtyard",
        "footpath",
        "agricultural fields",
        "large trees",
        "red soil",
        "cloud",
    ]
)


def read_area(source, bounds, crs):
    """Read only the requested area; never allow nodata into attention or PCA."""
    with rasterio.open(source["rgb"]) as ds:
        if ds.crs != rasterio.crs.CRS.from_string(crs) or ds.res != (
            PIXEL_SIZE,
            PIXEL_SIZE,
        ):
            raise ValueError("Expected the processed 0.5 m UTM RGB grid")
        if ds.count != 3 or ds.dtypes != ("uint8",) * 3:
            raise ValueError(
                "Use the uint8 RGB VRT, not the B/G/R/NIR reflectance raster"
            )
        win = from_bounds(*bounds, ds.transform)
        if any(
            abs(v - round(v)) > 1e-6
            for v in (win.col_off, win.row_off, win.width, win.height)
        ):
            raise ValueError("Bounds must align with the source pixel grid")
        win = win.round_offsets().round_lengths()
        if (
            win.col_off < 0
            or win.row_off < 0
            or win.col_off + win.width > ds.width
            or win.row_off + win.height > ds.height
        ):
            raise ValueError("Pilot bounds extend outside a source raster")
        valid = ds.read_masks(window=win).min(0) > 0
        if not valid.all():
            raise ValueError(
                f"{source['epoch']}: pilot contains nodata; select a fully covered area"
            )
        transform = ds.window_transform(win)
        image = ds.read(window=win)
    cloud = json.loads(Path(source["clouds"]).read_text())
    features = cloud.get("features", [cloud])
    shapes = [
        transform_geom("EPSG:4326", crs, f["geometry"])
        for f in features
        if f.get("geometry") and f["geometry"].get("coordinates")
    ]
    # Conservative 100 m rectangular guard around the pilot, matching the coverage rule's margin.
    margin = int(CLOUD_MARGIN_M / PIXEL_SIZE)
    cloud_transform = transform * rasterio.Affine.translation(-margin, -margin)
    cloud_shape = (image.shape[1] + 2 * margin, image.shape[2] + 2 * margin)
    if (
        shapes
        and geometry_mask(
            shapes, cloud_shape, cloud_transform, invert=True, all_touched=True
        ).any()
    ):
        raise ValueError(f"{source['epoch']}: pilot is within 100 m of mapped clouds")
    return image, transform


def fidelity(full, reconstructed, text, basis):
    """Compare final browser-style debiased cosine, not just uncorrected PCA."""

    def scores(x):
        if basis is not None:
            x = x - (x @ basis.T) @ basis
        return F.normalize(x, dim=-1) @ text.T

    a, b = scores(full), scores(reconstructed)
    result = {}
    k = max(1, len(full) // 20)
    for i, query in enumerate(QUERIES):
        x, y = a[:, i], b[:, i]
        xc, yc = x - x.mean(), y - y.mean()
        result[query] = {
            "cosine_mae": float((x - y).abs().mean()),
            "correlation": float((xc @ yc) / (xc.norm() * yc.norm()).clamp_min(1e-12)),
            "top5pct_overlap": len(
                set(x.topk(k).indices.tolist()) & set(y.topk(k).indices.tolist())
            )
            / k,
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("web/data/siaya"))
    ap.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--resize-factor", type=int, choices=[1, 2, 4], default=2)
    ap.add_argument("--components", type=int, default=128)
    ap.add_argument("--posbasis-k", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--model",
        default="models/text_encoder.onnx",
        help="Text model URL relative to web/",
    )
    ap.add_argument(
        "--tokenizer",
        default="models/bpe_simple_vocab_16e6.txt",
        help="Tokenizer URL relative to web/",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    manifest = json.loads(args.manifest.read_text())
    bounds, crs, sources = manifest["bounds"], manifest["crs"], manifest["sources"]
    if (
        crs != CRS
        or {s["epoch"] for s in sources} != {"pre", "endline"}
        or len(sources) != 2
    ):
        raise ValueError(
            "Expected exactly one pre and one endline source on EPSG:32636"
        )
    if (
        args.patch_size % TOKEN_STRIDE
        or args.stride % TOKEN_STRIDE
        or not 0 < args.stride <= args.patch_size
    ):
        raise ValueError(
            "Patch and stride must be multiples of 16, with 0 < stride <= patch"
        )
    width, height = (
        (bounds[2] - bounds[0]) / PIXEL_SIZE,
        (bounds[3] - bounds[1]) / PIXEL_SIZE,
    )
    if (
        min(width, height) < args.patch_size
        or max(width, height) > MAX_SIZE
        or width % TOKEN_STRIDE
        or height % TOKEN_STRIDE
    ):
        raise ValueError(
            "Pilot dimensions must be multiples of 16, >= patch size and <= 2048 pixels"
        )
    spacing = TOKEN_STRIDE * PIXEL_SIZE / args.resize_factor
    if any(abs(v / spacing - round(v / spacing)) > 1e-6 for v in bounds):
        raise ValueError("Bounds must align with the common embedding grid")
    if (
        not 1 <= args.components <= FEATURE_DIM
        or not 0 <= args.posbasis_k < FEATURE_DIM
    ):
        raise ValueError("Invalid PCA or positional rank")
    out = args.out_dir.resolve()
    web = Path(__file__).resolve().parents[1] / "web"
    prefix = out.relative_to(web).as_posix()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a fresh output directory")
    # Validate both sources before loading the model or writing artifacts.
    areas = [read_area(s, bounds, crs) for s in sources]
    out.mkdir(parents=True, exist_ok=True)
    opts = SimpleNamespace(
        patch_size=args.patch_size,
        stride=args.stride,
        resize_factor=args.resize_factor,
        batch_size=1,
        features="backbone",
    )
    device = torch.device(args.device)
    model, tokenizer = load_dinotxt()
    model = model.to(device).eval()
    features = []
    for source, (image, transform) in zip(sources, areas):
        epoch = source["epoch"]
        print(f"Embedding {epoch}: {source['catalog_id']}", flush=True)
        f = embed([model], image, opts, [device])
        features.append(f)
        np.save(out / f"{epoch}.reference.npy", f.numpy())
        # Write a small lossless RGB COG; source imagery remains untouched.
        with rasterio.io.MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                height=image.shape[1],
                width=image.shape[2],
                count=3,
                dtype="uint8",
                crs=crs,
                transform=transform,
                photometric="RGB",
            ) as ds:
                ds.write(image)
            with mem.open() as ds:
                rasterio.shutil.copy(
                    ds,
                    out / f"{epoch}.rgb.tif",
                    driver="COG",
                    compress="DEFLATE",
                    blocksize=256,
                )
    X = torch.cat([f.reshape(-1, FEATURE_DIM) for f in features])
    mean, comps, evr, n = fit_pca(
        X, args.components, min(len(X), 100_000), SEED, device
    )
    z = (X.to(device).float() - mean) @ comps.T
    vmin, vmax = torch.quantile(
        z, torch.tensor([(1 - QUANTILE) / 2, (1 + QUANTILE) / 2], device=device), dim=0
    )
    vmax = torch.maximum(vmax, vmin + 1e-6)
    basis = (
        positional_basis(model, opts, device, args.posbasis_k)
        if args.posbasis_k
        else None
    )
    if basis is not None:
        write_float32(out / "shared.posbasis.bin", basis, "shared positional basis")
    with torch.inference_mode():
        texts = text_features(model, tokenizer, VOCAB, device)
    (out / "vocab.json").write_text(
        json.dumps({"strings": VOCAB, "dim": FEATURE_DIM, "dtype": "float16"})
    )
    (out / "vocab_embeddings.bin").write_bytes(
        texts.cpu().numpy().astype("<f2").tobytes()
    )
    query_text = texts[len(BACKGROUND) : len(BACKGROUND) + len(QUERIES)]
    if basis is not None:
        query_text = F.normalize(query_text - (query_text @ basis.T) @ basis, dim=-1)
    pca = {
        "format": "dinotxt.pca.v1",
        "model": "dino.txt on DINOv3 ViT-L/16 LVD-1689M",
        "features": "backbone",
        "input_dim": FEATURE_DIM,
        "n_components": args.components,
        "window_px": args.patch_size,
        "stride_px": args.stride,
        "resize_factor": args.resize_factor,
        "token_size_px": TOKEN_STRIDE / args.resize_factor,
        "mean": mean.cpu().tolist(),
        "components": comps.cpu().tolist(),
        "explained_variance_ratio": evr.cpu().tolist(),
        "quantization": {
            "dtype": "uint8",
            "quantile": QUANTILE,
            "vmin": vmin.cpu().tolist(),
            "vmax": vmax.cpu().tolist(),
        },
        "text": {
            "embedding_slice": [FEATURE_DIM, 2 * FEATURE_DIM],
            "templates": TEMPLATES,
            "logit_scale": model.logit_scale.exp().item(),
        },
        "fit": {
            "epochs": [s["epoch"] for s in sources],
            "sample_tokens": n,
            "seed": SEED,
        },
    }
    (out / "shared.pca.json").write_text(json.dumps(pca))
    scenes = [
        {
            "name": f"Siaya · {s['epoch']} · {s['date']}",
            "config": f"{prefix}/{s['epoch']}.json",
        }
        for s in sources
    ]
    report = {}
    for source, f, (_, transform) in zip(sources, features, areas):
        epoch = source["epoch"]
        full = f.reshape(-1, FEATURE_DIM).to(device).float()
        coords = (full - mean) @ comps.T
        q = (
            ((coords - vmin) / (vmax - vmin) * 255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        report[epoch] = fidelity(
            full,
            mean + (vmin + q.float() / 255 * (vmax - vmin)) @ comps,
            query_text,
            basis,
        )
        report[epoch]["clipped_coordinate_fraction"] = float(
            ((coords < vmin) | (coords > vmax)).float().mean()
        )
        path = out / f"{epoch}.embeddings.tif"
        write_cog(
            str(path),
            q.reshape(f.shape[0], f.shape[1], -1).permute(2, 0, 1).cpu().numpy(),
            crs,
            transform * rasterio.Affine.scale(TOKEN_STRIDE / args.resize_factor),
            vmin.cpu().tolist(),
            vmax.cpu().tolist(),
        )
        scene = {
            "name": f"Siaya pilot — {epoch}",
            "subtitle": f"Vantor {source['sensor']} · {source['date']}",
            "rgb": f"{prefix}/{epoch}.rgb.tif",
            "embeddings": f"{prefix}/{epoch}.embeddings.tif",
            "pca": f"{prefix}/shared.pca.json",
            "attribution": "Vantor imagery via NASA CSDA · project use only",
            "source": {
                "catalog_id": source["catalog_id"],
                "date": source["date"],
                "sensor": source["sensor"],
            },
        }
        if basis is not None:
            scene["posbasis"] = f"{prefix}/shared.posbasis.bin"
            scene["posnorm"] = f"{prefix}/{epoch}.embeddings.posnorm.bin"
            write_posnorm(path, mean, comps, vmin, vmax, basis, device, 2_500_000)
        config = {
            "scene": scene,
            "scenes": scenes,
            "model": args.model,
            "vocab": args.tokenizer,
            "background": BACKGROUND,
            "examples": QUERIES,
            "point_vocab": {
                "strings": f"{prefix}/vocab.json",
                "embeddings": f"{prefix}/vocab_embeddings.bin",
            },
        }
        (out / f"{epoch}.json").write_text(json.dumps(config, indent=2))
    (out / "fidelity.json").write_text(json.dumps(report, indent=2))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "input": manifest,
                "parameters": vars(args)
                | {"manifest": str(args.manifest), "out_dir": str(args.out_dir)},
                "notes": [
                    "Fully valid pilot only",
                    "Cloud shadows not screened",
                    "Not coregistered",
                    "Shared PCA; no radiometric harmonization",
                ],
            },
            indent=2,
        )
    )
    print(f"Open http://localhost:8787/?config={prefix}/pre.json", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
