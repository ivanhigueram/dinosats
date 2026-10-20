"""Resumable, masked pre/endline embedding of all processed Vantor coverage.

Calibrate once across all strips, checkpoint each composite tile, then assemble
COGs. Original imagery remains in place. Run the same command to resume.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import rasterio.shutil
import torch
import torch.nn.functional as F
from build_siaya import BACKGROUND, QUERIES, VOCAB, fidelity
from dinotxt import load_dinotxt
from embed import (
    FEATURE_DIM,
    TEMPLATES,
    fit_pca,
    positional_basis,
    text_features,
    window_features,
    write_float32,
    write_posnorm,
)
from rasterio.enums import Resampling
from rasterio.windows import Window
from siaya_io import CRS, PIXEL_M, Strip, intersects, save_json, save_tile

PATCH = 256
STRIDE = 128
RESIZE = 2
TOKEN_PX = 8
TOKEN_M = 4
CORE = 1024
HALO = PATCH - STRIDE
COMPONENTS = 128
POS_RANK = 128
CONTEXT_MIN = 0.75
SEED = 42
QUANTILE = 0.999
EPOCHS = ("pre", "endline")
GLOBAL_TOKENS = 250_000
MIN_FREE_BYTES = 5 * 1024**3


def tile_bounds(plan, row, col, size=CORE):
    left = plan["bounds"][0] + col * CORE * PIXEL_M
    top = plan["bounds"][3] - row * CORE * PIXEL_M
    return [left, top - size * PIXEL_M, left + size * PIXEL_M, top]


def prepare(args):
    inventory = json.loads(args.inventory.read_text())["strips"]
    with args.requests.open() as stream:
        metadata = {r["catalog_id"]: r for r in csv.DictReader(stream)}
    records = []
    for index, item in enumerate(inventory, 1):
        record = dict(item)
        record["id"] = index
        record["off_nadir"] = float(metadata[item["catalog_id"]]["off_nadir"])
        record["clouds"] = str(args.clouds / f"{item['catalog_id']}.cloud.json")
        record["source_mtime_ns"] = Path(record["rgb"]).stat().st_mtime_ns
        record["cloud_sha256"] = hashlib.sha256(
            Path(record["clouds"]).read_bytes()
        ).hexdigest()
        records.append(record)
    step = CORE * PIXEL_M
    bounds = [
        math.floor(min(r["bounds"][0] for r in records) / step) * step,
        math.floor(min(r["bounds"][1] for r in records) / step) * step,
        math.ceil(max(r["bounds"][2] for r in records) / step) * step,
        math.ceil(max(r["bounds"][3] for r in records) / step) * step,
    ]
    if args.bounds:
        bounds = list(args.bounds)
        if (
            any(v % step for v in bounds)
            or bounds[0] >= bounds[2]
            or bounds[1] >= bounds[3]
        ):
            raise ValueError("Bounds must be ordered and aligned to 512 m")
    plan = {
        "bounds": bounds,
        "crs": CRS,
        "records": records,
        "rows": round((bounds[3] - bounds[1]) / step),
        "cols": round((bounds[2] - bounds[0]) / step),
        "patch": PATCH,
        "stride": STRIDE,
        "resize": RESIZE,
        "components": COMPONENTS,
        "pos_rank": POS_RANK,
        "min_context": CONTEXT_MIN,
        "samples_per_strip": args.samples_per_strip,
        "seed": SEED,
        "web_prefix": args.web_prefix,
        "policy": "Valid/cloud-buffered pixels; lower off-nadir first; catalog ID breaks ties. No cross-epoch filling.",
    }
    plan["code_sha256"] = hashlib.sha256(
        b"".join(
            Path(__file__).with_name(n).read_bytes()
            for n in ["build_siaya_full.py", "siaya_io.py", "embed.py"]
        )
    ).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["fingerprint"] = fingerprint
    if (args.out / "plan.json").exists():
        old = json.loads((args.out / "plan.json").read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("Run inputs/code changed; use a new output directory")
    else:
        save_json(args.out / "plan.json", plan)
    return plan


def tasks(plan):
    result = []
    for epoch in EPOCHS:
        records = sorted(
            (r for r in plan["records"] if r["epoch"] == epoch),
            key=lambda r: (r["off_nadir"], r["catalog_id"]),
        )
        for row in range(plan["rows"]):
            for col in range(plan["cols"]):
                bounds = tile_bounds(plan, row, col)
                selected = [r["id"] for r in records if intersects(bounds, r["bounds"])]
                if selected:
                    result.append(
                        {"epoch": epoch, "row": row, "col": col, "sources": selected}
                    )
    return sorted(result, key=lambda j: (j["row"], j["col"], j["epoch"]))


def masked_features(model, strip, bounds, device, core=CORE):
    """Use identical global window positions on adjoining tiles, then crop the halo."""
    h = HALO * PIXEL_M
    expanded = [bounds[0] - h, bounds[1] - h, bounds[2] + h, bounds[3] + h]
    image, valid = strip.read(expanded)
    size = core + 2 * HALO
    if valid.shape != (size, size):
        raise ValueError("Unexpected tile dimensions")
    dim = size // TOKEN_PX
    mask = valid.reshape(dim, TOKEN_PX, dim, TOKEN_PX).all(axis=(1, 3))
    acc = torch.zeros(dim, dim, FEATURE_DIM)
    weights = torch.zeros(dim, dim, 1)
    t = PATCH // TOKEN_PX
    hann = torch.sin(torch.pi * (torch.arange(t) + 0.5) / t) ** 2
    blend = (hann[:, None] * hann[None, :])[..., None]
    opts = SimpleNamespace(patch_size=PATCH, resize_factor=RESIZE, features="backbone")
    n_windows = 0
    for y in range(0, size - PATCH + 1, STRIDE):
        for x in range(0, size - PATCH + 1, STRIDE):
            window_mask = valid[y : y + PATCH, x : x + PATCH]
            if window_mask.mean() < CONTEXT_MIN:
                continue
            pixels = image[:, y : y + PATCH, x : x + PATCH].copy()
            # Mean-fill missing context; only completely observed tokens receive scores.
            if not window_mask.all():
                for band in pixels:
                    band[~window_mask] = round(float(band[window_mask].mean()))
            f = (
                window_features(
                    model, torch.from_numpy(pixels), [(0, 0)], opts, device
                )[0]
                .float()
                .cpu()
            )
            if not torch.isfinite(f).all():
                raise RuntimeError("Nonfinite model output")
            ty, tx = y // TOKEN_PX, x // TOKEN_PX
            w = blend * torch.from_numpy(mask[ty : ty + t, tx : tx + t])[..., None]
            acc[ty : ty + t, tx : tx + t] += f * w
            weights[ty : ty + t, tx : tx + t] += w
            n_windows += 1
    start, stop = HALO // TOKEN_PX, (HALO + core) // TOKEN_PX
    good = weights[start:stop, start:stop, 0].numpy() > 0
    features = F.normalize(
        acc[start:stop, start:stop] / weights[start:stop, start:stop].clamp_min(1e-12),
        dim=-1,
    ).half()
    return features, good, image[:, HALO : HALO + core, HALO : HALO + core], n_windows


def fit_shared(args, plan, strips, model, tokenizer, device):
    calibration = args.out / "calibration.npz"
    if calibration.exists():
        with np.load(calibration) as data:
            return {
                k: torch.from_numpy(data[k].copy())
                for k in ["mean", "components", "vmin", "vmax"]
            }
    train, validation = [], []
    for record in plan["records"]:
        rng = np.random.default_rng(SEED + record["id"])
        cache = args.out / "samples" / f"{record['catalog_id']}.npz"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            with np.load(cache) as data:
                samples = [torch.from_numpy(data[k].copy()) for k in data.files]
        else:
            samples = []
            lo, bo, ri, to = record["bounds"]
            for _ in range(2000):
                if len(samples) >= args.samples_per_strip:
                    break
                # Fixed grid origin keeps calibration windows consistent with production.
                x = (
                    math.floor(
                        rng.uniform(lo, ri - PATCH * PIXEL_M) / (STRIDE * PIXEL_M)
                    )
                    * STRIDE
                    * PIXEL_M
                )
                y = (
                    math.floor(
                        rng.uniform(bo + PATCH * PIXEL_M, to) / (STRIDE * PIXEL_M)
                    )
                    * STRIDE
                    * PIXEL_M
                )
                bounds = [x, y - PATCH * PIXEL_M, x + PATCH * PIXEL_M, y]
                _, valid = strips[record["id"]].read(bounds)
                if valid.mean() < 0.98:
                    continue
                f, good, _, _ = masked_features(
                    model, strips[record["id"]], bounds, device, core=PATCH
                )
                if good.mean() < 0.98:
                    continue
                samples.append(f[good].clone())
                print(
                    f"Calibration {record['catalog_id']}: {len(samples)}/{args.samples_per_strip}",
                    flush=True,
                )
            if len(samples) != args.samples_per_strip:
                raise RuntimeError(
                    f"Not enough clear calibration areas in {record['catalog_id']}"
                )
            save_tile(cache, **{str(i): s.numpy() for i, s in enumerate(samples)})
        train.extend(samples[:-1])
        validation.append(samples[-1])
    X = torch.cat(train)
    mean, comps, evr, n = fit_pca(X, COMPONENTS, len(X), SEED, torch.device("cpu"))
    z = (X.float() - mean) @ comps.T
    vmin, vmax = torch.quantile(
        z, torch.tensor([(1 - QUANTILE) / 2, (1 + QUANTILE) / 2]), dim=0
    )
    vmax = torch.maximum(vmax, vmin + 1e-6)
    pca = {
        "format": "dinotxt.pca.v1",
        "model": "dino.txt on DINOv3 ViT-L/16 LVD-1689M",
        "features": "backbone",
        "input_dim": FEATURE_DIM,
        "n_components": COMPONENTS,
        "window_px": PATCH,
        "stride_px": STRIDE,
        "resize_factor": RESIZE,
        "token_size_px": TOKEN_PX,
        "mean": mean.tolist(),
        "components": comps.tolist(),
        "explained_variance_ratio": evr.tolist(),
        "quantization": {
            "dtype": "uint8",
            "quantile": QUANTILE,
            "vmin": vmin.tolist(),
            "vmax": vmax.tolist(),
        },
        "text": {
            "embedding_slice": [FEATURE_DIM, 2 * FEATURE_DIM],
            "templates": TEMPLATES,
            "logit_scale": model.logit_scale.exp().item(),
        },
        "fit": {
            "training_tokens": n,
            "heldout_tokens": sum(len(v) for v in validation),
            "strips": len(strips),
            "seed": SEED,
        },
    }
    opts = SimpleNamespace(patch_size=PATCH, resize_factor=RESIZE, features="backbone")
    basis = positional_basis(model, opts, device, POS_RANK).cpu()
    write_float32(args.out / "shared.posbasis.bin", basis, "shared positional basis")
    texts = text_features(model, tokenizer, VOCAB, device).cpu()
    save_json(
        args.out / "vocab.json",
        {"strings": VOCAB, "dim": FEATURE_DIM, "dtype": "float16"},
    )
    (args.out / "vocab_embeddings.bin").write_bytes(
        texts.numpy().astype("<f2").tobytes()
    )
    query = texts[len(BACKGROUND) : len(BACKGROUND) + len(QUERIES)]
    query = F.normalize(query - (query @ basis.T) @ basis, dim=-1)
    report = {}
    for record, V in zip(plan["records"], validation):
        V = V.float()
        coords = (V - mean) @ comps.T
        q = ((coords - vmin) / (vmax - vmin) * 255).round().clamp(0, 255)
        report[record["catalog_id"]] = fidelity(
            V, mean + (vmin + q / 255 * (vmax - vmin)) @ comps, query, basis
        )
    save_json(args.out / "fidelity.json", report)
    save_json(args.out / "shared.pca.json", pca)
    # Written last: a resumable calibration is complete only when all companion files exist.
    save_tile(
        calibration,
        mean=mean.numpy(),
        components=comps.numpy(),
        vmin=vmin.numpy(),
        vmax=vmax.numpy(),
    )
    return {"mean": mean, "components": comps, "vmin": vmin, "vmax": vmax}


def encode_tile(task, plan, strips, model, pca, device):
    dim = CORE // TOKEN_PX
    codes = np.zeros((COMPONENTS, dim, dim), dtype=np.uint8)
    sources = np.zeros((dim, dim), dtype=np.uint8)
    rgb = np.zeros((3, CORE, CORE), dtype=np.uint8)
    windows = 0
    bounds = tile_bounds(plan, task["row"], task["col"])
    for sid in task["sources"]:
        if (sources > 0).all():
            break
        # Skip empty coverage before allocating the dense feature buffer or running the encoder.
        _, valid = strips[sid].read(bounds)
        potential = valid.reshape(dim, TOKEN_PX, dim, TOKEN_PX).all(axis=(1, 3)) & (
            sources == 0
        )
        if not potential.any():
            continue
        f, good, image, n = masked_features(model, strips[sid], bounds, device)
        windows += n
        take = good & (sources == 0)
        if not take.any():
            continue
        z = (f[take].float() - pca["mean"]) @ pca["components"].T
        q = (
            ((z - pca["vmin"]) / (pca["vmax"] - pca["vmin"]) * 255)
            .round()
            .clamp(0, 255)
            .byte()
            .numpy()
        )
        codes[:, take] = q.T
        sources[take] = sid
        pixels = take.repeat(TOKEN_PX, axis=0).repeat(TOKEN_PX, axis=1)
        rgb[:, pixels] = image[:, pixels]
    return {
        "codes": codes,
        "sources": sources,
        "rgb": rgb,
        "windows": np.array(windows),
    }


def assemble(args, plan, jobs):
    """Build COG overviews from masks; never average nodata into valid embeddings."""
    out = args.out
    prefix = args.web_prefix
    for epoch in EPOCHS:
        config_path = out / f"{epoch}.json"
        if config_path.exists():
            continue
        epoch_jobs = [j for j in jobs if j["epoch"] == epoch]
        common = {
            "driver": "GTiff",
            "dtype": "uint8",
            "crs": CRS,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "DEFLATE",
            "BIGTIFF": "YES",
            "SPARSE_OK": "YES",
        }
        token_transform = rasterio.Affine(
            TOKEN_M, 0, plan["bounds"][0], 0, -TOKEN_M, plan["bounds"][3]
        )
        products = [
            ("embeddings", COMPONENTS, CORE // TOKEN_PX, token_transform, "average"),
            ("validity", 1, CORE // TOKEN_PX, token_transform, "average"),
            ("sources", 1, CORE // TOKEN_PX, token_transform, "nearest"),
            (
                "rgb",
                3,
                CORE,
                rasterio.Affine(
                    PIXEL_M, 0, plan["bounds"][0], 0, -PIXEL_M, plan["bounds"][3]
                ),
                "average",
            ),
        ]
        for name, bands, size, transform, resampling in products:
            destination = out / f"{epoch}.{name}.tif"
            if destination.exists():
                continue
            scratch = out / f"{epoch}.{name}.work.tif"
            height, width = plan["rows"] * size, plan["cols"] * size
            with rasterio.Env(
                GDAL_TIFF_INTERNAL_MASK=True, GDAL_CACHEMAX=256 * 1024 * 1024
            ):
                with rasterio.open(
                    scratch,
                    "w",
                    **common,
                    height=height,
                    width=width,
                    count=bands,
                    transform=transform,
                ) as ds:
                    if name in ("embeddings", "rgb"):
                        ds.write_mask(False)
                    if name == "rgb":
                        ds.colorinterp = (
                            rasterio.enums.ColorInterp.red,
                            rasterio.enums.ColorInterp.green,
                            rasterio.enums.ColorInterp.blue,
                        )
                    for job in epoch_jobs:
                        path = out / "tiles" / f"{epoch}_{job['row']}_{job['col']}.npz"
                        with np.load(path) as tile:
                            source = tile["sources"]
                            valid = source > 0
                            if not valid.any():
                                continue
                            if name == "embeddings":
                                data = tile["codes"]
                            elif name == "sources":
                                data = source[None]
                            elif name == "validity":
                                data = (valid.astype(np.uint8) * 255)[None]
                            else:
                                data = tile["rgb"]
                            win = Window(
                                job["col"] * size, job["row"] * size, size, size
                            )
                            ds.write(data, window=win)
                            if name in ("embeddings", "rgb"):
                                mask = (
                                    valid
                                    if name == "embeddings"
                                    else valid.repeat(TOKEN_PX, 0).repeat(TOKEN_PX, 1)
                                )
                                ds.write_mask(mask.astype(np.uint8) * 255, window=win)
                    # Explicit overview count keeps auxiliary grids aligned with the embedding levels.
                    factors = []
                    factor = 2
                    while min(height, width) / factor >= 128:
                        factors.append(factor)
                        factor *= 2
                    ds.build_overviews(factors, getattr(Resampling, resampling))
                tmp = out / f"{epoch}.{name}.tmp.tif"
                rasterio.shutil.copy(
                    scratch,
                    tmp,
                    driver="COG",
                    compress="DEFLATE",
                    blocksize=256,
                    overview_resampling=resampling.upper(),
                    BIGTIFF="YES",
                    OVERVIEWS="FORCE_USE_EXISTING",
                )
                tmp.replace(destination)
                scratch.unlink()
            print(f"Assembled {destination}", flush=True)
        pca = json.loads((out / "shared.pca.json").read_text())
        tensor = torch.tensor
        basis = torch.from_numpy(
            np.fromfile(out / "shared.posbasis.bin", dtype="<f4").reshape(
                -1, FEATURE_DIM
            )
        )
        write_posnorm(
            out / f"{epoch}.embeddings.tif",
            tensor(pca["mean"]),
            tensor(pca["components"]),
            tensor(pca["quantization"]["vmin"]),
            tensor(pca["quantization"]["vmax"]),
            basis,
            torch.device("cpu"),
            GLOBAL_TOKENS,
        )
        catalog = {
            str(r["id"]): {
                "catalog_id": r["catalog_id"],
                "date": r["date"],
                "sensor": r["sensor"],
            }
            for r in plan["records"]
            if r["epoch"] == epoch
        }
        scene = {
            "name": f"Siaya — {epoch}",
            "subtitle": "Vantor · processed clear coverage",
            "rgb": f"{prefix}/{epoch}.rgb.tif",
            "embeddings": f"{prefix}/{epoch}.embeddings.tif",
            "validity": f"{prefix}/{epoch}.validity.tif",
            "sources": f"{prefix}/{epoch}.sources.tif",
            "source_catalog": catalog,
            "pca": f"{prefix}/shared.pca.json",
            "posbasis": f"{prefix}/shared.posbasis.bin",
            "posnorm": f"{prefix}/{epoch}.embeddings.posnorm.bin",
            "max_global_tokens": GLOBAL_TOKENS,
            "attribution": "Vantor via NASA CSDA · project use only",
        }
        config = {
            "scene": scene,
            "scenes": [
                {"name": f"Siaya · {e}", "config": f"{prefix}/{e}.json"} for e in EPOCHS
            ],
            "model": "models/text_encoder.onnx",
            "vocab": "models/bpe_simple_vocab_16e6.txt",
            "background": BACKGROUND,
            "examples": QUERIES,
            "point_vocab": {
                "strings": f"{prefix}/vocab.json",
                "embeddings": f"{prefix}/vocab_embeddings.bin",
            },
        }
        save_json(config_path, config)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.out / "error.json").unlink(missing_ok=True)
    plan = prepare(args)
    jobs = tasks(plan)
    save_json(args.out / "jobs.json", jobs)
    print(
        f"{len(jobs)} candidate tiles; empty coverage is skipped. Output: {args.out}",
        flush=True,
    )
    if args.plan_only:
        return
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA unavailable in this execution context")
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    tile_dir = args.out / "tiles"
    tile_dir.mkdir(exist_ok=True)

    def done(j):
        return tile_dir / f"{j['epoch']}_{j['row']}_{j['col']}.npz"

    pending = [j for j in jobs if not done(j).exists()]
    device = torch.device(args.device)
    strips = {r["id"]: Strip(r) for r in plan["records"]}
    started = time.time()
    save_json(
        args.out / "status.json",
        {
            "phase": "calibration",
            "total_tiles": len(jobs),
            "completed_tiles": len(jobs) - len(pending),
        },
    )
    try:
        if pending or not (args.out / "calibration.npz").exists():
            model, tokenizer = load_dinotxt()
            model = model.to(device).eval()
            pca = fit_shared(args, plan, strips, model, tokenizer, device)
            # Only the vision tower is needed after calibration and text precomputation.
            model.text_model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for index, job in enumerate(pending):
                if args.tile_limit and index >= args.tile_limit:
                    break
                if shutil.disk_usage(args.out).free < MIN_FREE_BYTES:
                    raise RuntimeError("Less than 5 GiB free; checkpoints retained")
                tick = time.time()
                result = encode_tile(job, plan, strips, model, pca, device)
                save_tile(done(job), **result)
                status = {
                    "phase": "embedding",
                    "total_tiles": len(jobs),
                    "completed_tiles": len(jobs) - len(pending) + index + 1,
                    "last_tile": job,
                    "last_tile_seconds": round(time.time() - tick, 2),
                    "last_valid_tokens": int((result["sources"] > 0).sum()),
                    "last_windows": int(result["windows"]),
                    "run_seconds": round(time.time() - started, 1),
                }
                save_json(args.out / "status.json", status)
                print(json.dumps(status), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        complete = all(done(j).exists() for j in jobs)
        if complete:
            save_json(
                args.out / "status.json",
                {
                    "phase": "assembling",
                    "total_tiles": len(jobs),
                    "completed_tiles": len(jobs),
                },
            )
            assemble(args, plan, jobs)
            save_json(
                args.out / "status.json",
                {
                    "phase": "complete",
                    "total_tiles": len(jobs),
                    "completed_tiles": len(jobs),
                },
            )
        else:
            print(
                "Tile limit reached. Re-run the same command without --tile-limit to continue.",
                flush=True,
            )
    finally:
        for strip in strips.values():
            strip.close()
        lock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, default=Path("data/siaya/inventory.json"))
    ap.add_argument(
        "--requests",
        type=Path,
        default=Path("/mnt/drive/remote_outcomes/csda/vantor_request.csv"),
    )
    ap.add_argument(
        "--clouds",
        type=Path,
        default=Path("/mnt/drive/remote_outcomes/csda/vantor_browse"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--web-prefix", default="data/siaya-full")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--samples-per-strip", type=int, default=4)
    ap.add_argument("--tile-limit", type=int, default=0)
    ap.add_argument("--bounds", type=float, nargs=4)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    if args.samples_per_strip < 2:
        ap.error("Need training and held-out calibration areas")
    try:
        with torch.inference_mode():
            run(args)
    except BlockingIOError:
        raise SystemExit("Another builder holds the output lock")
    except Exception as exc:
        if args.out.exists():
            save_json(args.out / "error.json", {"error": str(exc), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
