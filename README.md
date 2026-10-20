# DinoSats

**Jump to: [Using the app](#using-the-app) | [How it works](#how-it-works) | [Run it locally](#run-it-locally) | [Make your own scene](#make-your-own-scene) | [Configuration](#configuration) | [Hosting](#hosting) | [Credits](#credits) | [Citation](#citation) | [License](#license)**

The DinoSats app lets you search aerial imagery by typing a description of what you are looking for. For example, if you type *runway*, *marina*, *solar panels* or *aircraft carrier*, the app shows where those things are across 380 km² of San Diego. Everything runs in your web browser. The app compares precomputed [DINOv3](https://github.com/facebookresearch/dinov3) embeddings of the imagery with an embedding of your prompt, which is computed by [dino.txt](https://arxiv.org/abs/2412.16334), a text model that was trained to align with DINOv3.

**You can try the app at [calebrob.com/dinosats](https://calebrob.com/dinosats/).**

<p align="center">
  <a href="https://calebrob.com/dinosats/"><img src="web/teaser.jpg" alt="DinoSats teaser: a heatmap for the prompt beach over San Diego, highlighting the ocean beaches of Coronado and the Silver Strand" width="100%"></a>
</p>

The image above shows the heatmap for the prompt *beach*, which highlights the ocean beaches of Coronado and the Silver Strand.

## Using the app

- **Search.** Type a prompt in the search box, or click one of the example prompts below it. Pressing `/` moves the cursor to the search box. About 900 common prompts are precomputed. Any other prompt needs the 664 MB text encoder, which the app downloads the first time you use it and the browser normally caches for later visits.
- **Top matches.** When you switch on Top matches, the app lists the 10 highest-scoring places, and clicking one zooms the map to it. The list is chosen from a coarse version of the whole scene, with patches about 20 m across, so small objects may only appear after you zoom in.
- **Describe a location.** If you hold `Shift` and click on the map, the app shows the words whose embeddings best match that location.
- **Display settings.** The Display card lets you switch between the relative score and raw similarity, and change the threshold, the opacity and the background prompts. Pressing `Space` hides or shows the heatmap.

## How it works

DINOv3 is an image model from Meta. It turns each patch of imagery, about 5 m across, into a vector: a list of numbers that describes what the patch looks like. Its companion text model, dino.txt, was trained to turn text into vectors of the same kind, so the vector for a patch of runway is similar to the vector for the word *runway*. The app scores each patch by how similar its vector is to the vector for your prompt.

- **Embedding the imagery ahead of time.** We embed 2022 [NAIP](https://planetarycomputer.microsoft.com/dataset/naip) aerial imagery, which has 0.6 m pixels, with DINOv3, and we compress each patch's vector to 64 bytes. The San Diego scene has 16.5 million patches, which are stored in a 1.1 GB Cloud-Optimized GeoTIFF.
- **Encoding the prompt in your browser.** The text model runs on your own device with onnxruntime-web, so there is no backend server, and the site consists only of static files.
- **Loading the embeddings.** The map loads a coarse version of the embeddings first, so the app is usable within a few seconds, and then downloads higher-resolution blocks as you zoom in.
- **Scoring against background prompts.** Each patch's score is how much better it matches your prompt than a set of generic background prompts, such as *water*, *road* and *buildings*. These relative scores make the heatmap much easier to read than the raw similarity.

DinoSats is zero-shot: dino.txt was trained on web photos, not aerial imagery, and we have not fine-tuned it. It works best on things that are large and distinctive when seen from above, such as runways, golf courses, marinas and sports fields, and it works less well on small or ambiguous objects. It is a quick way to explore a scene, not a replacement for a trained detector.

## Run it locally

To run the app on your own machine with the same hosted data as the demo, you need Node.js and Python 3. Clone the repo, build the app and start the local server:

```bash
git clone https://github.com/calebrob6/dinosats.git
cd dinosats
npm install
npm run build
python3 serve.py
```

Then open http://localhost:8766/ in your browser. Browsers only allow the model cache and multi-threading on `localhost` or over https, so if you run the server on a remote machine, forward the port with `ssh -L 8766:localhost:8766 <host>` and open the app through the forwarded port.

## Make your own scene

To build a scene for a different area, start from the clone in [Run it locally](#run-it-locally). You also need a CUDA GPU, about 34 GB of RAM for a scene the size of San Diego, Python 3.12, [azcopy](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10), and access to the gated [DINOv3 weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) on Hugging Face.

First, install [uv](https://docs.astral.sh/uv/) and create the locked Python 3.12 environment.
`pyproject.toml` declares the pipeline dependencies and `uv.lock` pins their exact versions.
The DINOv3 source is cloned separately because the scripts import it directly from `dinov3/`.

Set up the environment and download the models:

```bash
uv sync --locked
git clone https://github.com/facebookresearch/dinov3.git
git -C dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751

# Create a gitignored .env containing HF_TOKEN=hf_...
uv run --env-file .env hf auth whoami
uv run --env-file .env hf download facebook/dinov3-vitl16-pretrain-lvd1689m model.safetensors --local-dir weights/hf-dinov3-vitl16-lvd1689m
uv run --env-file .env hf download PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder \
    dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth --local-dir weights
curl -L -o weights/bpe_simple_vocab_16e6.txt.gz https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz
```

Check the installation before downloading imagery:

```bash
uv lock --check
uv pip check
uv run python scripts/embed.py --help
uv run python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
uv run --env-file .env hf auth whoami
azcopy --version
```

`uv run --env-file .env` supplies `HF_TOKEN` to Hugging Face without a CLI login or
`python-dotenv`. The token must have read access to the gated DINOv3 repository.
The root `.env` is gitignored; keep the token out of tracked files.
A CUDA-enabled PyTorch wheel still needs a working NVIDIA driver. If CUDA is unavailable,
the embedding script falls back to CPU, which is useful for small smoke checks but slow for
an entire scene. `azcopy` is a separate system executable, not a Python dependency.
Use `--device cuda:0 --batch_size 1` as a starting point on a single GPU; the demo's default
1024-pixel windows become 2048 pixels with `--resize_factor 2` and need substantial VRAM.

The dino.txt weights on Hugging Face are a mirror of the file that Meta distributes through the [dinov3 repo](https://github.com/facebookresearch/dinov3#pretrained-heads---zero-shot-tasks-with-dinotxt).

Next, build the scene. The commands below build the San Diego scene. For a different area, replace the NAIP quarter quad IDs and the file names with your own.

```bash
# 1. Download the NAIP quarter quads.
uv run python scripts/download_naip.py 3211715_sw 3211715_se 3211716_sw 3211723_nw 3211723_ne 3211724_nw 3211723_sw 3211723_se 3211724_sw --year 2022 --state ca

# 2. Mosaic them into one GeoTIFF for embedding and one RGB GeoTIFF for display.
mkdir -p web/data
uv run python scripts/make_mosaic.py data/naip/*.tif --out data/naip_2022_san_diego_3x3.tif --web_out web/data/naip_2022_san_diego_3x3_rgb.tif

# 3. Embed the mosaic with DINOv3. This is the slowest step: it takes about an hour on two GPUs. To use one GPU, pass --device cuda:0.
uv run python scripts/embed.py --input_fn data/naip_2022_san_diego_3x3.tif --output_fn web/data/naip_2022_san_diego_3x3_dinotxt_pca64_2x_s256.tif --resize_factor 2 --device cuda:0,cuda:1

# 4. Precompute the embeddings of the common prompts.
uv run python scripts/embed_vocab.py

# 5. Export the text encoder to ONNX for the browser.
uv run python scripts/export_text_encoder.py

# 6. Copy the config and replace the hosted URLs with local paths. If your file names differ, edit them in local.json.
sed -E 's#https://[^"]*/dinosats/##' web/config.json > web/data/local.json
python3 serve.py
```

Finally, open http://localhost:8766/?config=data/local.json to see your scene.

A quad ID such as `3211723_ne` means the north-east quarter of quad 23 in the 1° cell at 32° N, 117° W. Quads are numbered 1 to 64, row by row, starting from the north-west corner of the cell. Every script accepts `--help`, which lists its options.

## Local Siaya pilot

The local Vantor pilot has separate pre (2013-03-02, WV02) and endline
(2016-08-29, GE01) acquisitions covering the same 256 m square. Start it with:

```bash
uv run python serve.py --port 8787 --config data/siaya/pre.json
```

Open http://localhost:8787/. The Acquisition selector switches epochs and retains
the current query. The strip ID, sensor and acquisition date are shown below it.
The pilot assets must have been generated first; they are deliberately gitignored.
Vantor imagery and derived renders stay within the project under the CSDA agreement.

`scripts/build_siaya.py` reads a bounded window from each existing RGB strip VRT;
it never modifies the source imagery or `../remote_outcomes`. This pilot requires
complete valid RGB coverage in both acquisitions and rejects mapped clouds within
100 m of its bounds. It does not detect cloud shadow or correct registration,
seasonal differences, or strip brightness. No cross-date imagery blending is used.

The local input manifest is `data/siaya/pilot.json`. It records `bounds`
(`[left, bottom, right, top]` in metres), `crs` (`EPSG:32636`), and two `sources`.
Each source specifies `epoch` (`pre` or `endline`), `catalog_id`, `date`, `sensor`,
`rgb` (the existing RGB VRT path), and `clouds` (Vantor GeoJSON path in EPSG:4326).
Keep manifests with project coordinates in the ignored `data/` directory.

To reproduce into a fresh output directory:

```bash
uv run python scripts/build_siaya.py --manifest data/siaya/pilot.json \
    --out-dir web/data/siaya-rerun --device cuda:0
uv run python serve.py --port 8787 --config data/siaya-rerun/pre.json
```

The default pilot uses 256-pixel windows, 128-pixel stride, 2× upsampling (4 m
output spacing), a joint 128-component PCA, shared uint8 quantization ranges, and
128 positional directions. These smaller windows suit the T1000 8 GB; inference
uses fp32 on GPUs without native bf16. On the T1000, measured fp16 inference was
about five times slower. CPU execution is available with `--device cpu`.
If an agent sandbox hides NVIDIA devices, run in a GPU-enabled execution context;
that does not require changing this environment or reinstalling the driver.

Outputs include RGB and embedding COGs, a shared PCA/basis, precomputed local
prompts, acquisition configs, full float16 reference features, and `fidelity.json`.
Fidelity compares final position-corrected text scores and top-5% token overlap
against the full features. It measures compression loss, not detection accuracy.
Bounds must share an embedding-grid origin, fit within 2048 × 2048 source pixels,
and contain no nodata. This is a pilot builder, not a whole-strip processing pipeline.

New prompts use the original demo's text ONNX model, downloaded locally to
`web/models/text_encoder.onnx`; the CLIP tokenizer is
`web/models/bpe_simple_vocab_16e6.txt`. These are shared by both acquisitions.
To recreate those files without exporting the text encoder again:

```bash
mkdir -p web/models
curl -fL -o web/models/text_encoder.onnx \
    https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/calebrob6/geospatialml/dinosats/models/text_encoder.onnx
uv run python -c "import gzip; from pathlib import Path; Path('web/models/bpe_simple_vocab_16e6.txt').write_bytes(gzip.decompress(Path('weights/bpe_simple_vocab_16e6.txt.gz').read_bytes()))"
```

The downloaded ONNX SHA256 is
`f30e29cd552f53e9432260d0103d66c17433739f30e2b9f91409058c7c811c52`.
Common prompts load immediately; a new prompt loads the 664 MB model into the browser.
The original San Diego configuration remains available at `/?config=config.json`.

## Full Siaya pre/endline map

`scripts/build_siaya_full.py` processes the nine RGB strip VRTs listed in
`data/siaya/inventory.json`. Source imagery stays in place. Outputs need substantial
disk space; use the data drive:

```bash
uv run python scripts/build_siaya_full.py \
    --out /mnt/drive/dinosats/siaya_full
```

Run the same command to resume. `plan.json` records the source inventory, cloud
masks, settings, and code fingerprint; changed inputs require a new output folder.
`status.json` reports progress, `error.json` records failures, and each completed
tile is an atomic checkpoint. `--plan-only` lists the work without loading the
model; `--tile-limit 1` stops after one tile for a smoke check.

The two epochs use a common 4 m grid and a shared 128-component PCA/uint8
quantizer calibrated across all nine strips. One sampled area per strip is held
out for `fidelity.json`; this measures compression loss, not search accuracy.
Overlapping windows and tile halos reduce edge artifacts.

Within each epoch, lower off-nadir imagery supplies valid tokens first. Other
strips fill remaining coverage. The RGB mosaic uses the same source as each
embedding. Nodata and clouds buffered by 100 m are excluded; cloud shadows are
not detected. Tokens require all source pixels to be valid and at least one
window with 75% valid context. Missing context is mean-filled. Coverage therefore
excludes some cutline edges and is not the complete rectangular strip footprint.

Final outputs include imagery, embeddings, coverage and acquisition COGs, and
pre/endline configs. Invalid areas have no search scores. Shift-click shows the
source sensor, date and catalog ID; overview source labels are representative.
The endline mosaic combines dates from June 2016 through January 2017. Neither
the shared grid nor the composite corrects residual image registration errors.

After assembly, expose the output folder locally:

```bash
ln -s /mnt/drive/dinosats/siaya_full web/data/siaya-full
uv run python serve.py --port 8787 --config data/siaya/pre.json
```

Open `http://localhost:8787/?config=data/siaya-full/pre.json` and switch epochs
with the acquisition menu. The root URL continues to show the pilot during
processing. The full configs appear only after their COGs are assembled.

### Moving the run to HPC

Transfer these inputs separately from git:

- `remote_outcomes/vantor/ard/`: processed TOA GeoTIFFs and RGB/strip VRTs;
  preserve their relative directory layout.
- `remote_outcomes/csda/vantor_request.csv` and `vantor_browse/*.cloud.json`.
- This checkout's `data/siaya/inventory.json`; update each strip's `rgb` path
  for the HPC filesystem. Copy `data/siaya/pilot.json` only if rebuilding the pilot.
- `weights/`, or download the weights using the setup instructions above.

The embedding step reads these processed inputs directly. Raw NASA deliveries
are needed to repeat preprocessing in `remote_outcomes`.

Create the environment with `uv sync --locked` and install the pinned DINOv3
checkout described above. Verify CUDA inside a GPU allocation, then run:

```bash
uv run python scripts/build_siaya_full.py \
    --inventory /path/to/inventory.json \
    --requests /path/to/csda/vantor_request.csv \
    --clouds /path/to/csda/vantor_browse \
    --out /path/to/new/siaya_full --device cuda:0
```

The full builder currently uses one GPU. Use a fresh output directory when
relocating inputs: the resume fingerprint includes paths and VRT timestamps.
After processing, the browser also needs the local text model and tokenizer
described in the pilot section. These assets are excluded from git.

### Sherlock with NGC / Apptainer

The container recipe is awaiting cluster validation. It follows the working
`remote_outcomes` build: NGC 26.02, node-local extraction, and a staged image
published only after smoke checks pass. Builds request eight CPUs and 64 GB.

The host's glibc 2.17 cannot install the locked ONNX Runtime wheel, which needs
glibc 2.28 or newer. Use `containers/ngc.def` for a separate embedding runtime:
NGC `pytorch:26.02-py3` supplies Python 3.12, PyTorch 2.11 and CUDA 13.1.1.
`uv` installs extra packages in an in-image environment that inherits NGC's
packages. Torch/TorchVision versions are protected, and NumPy is capped below
2 to match the working reference. Do not run the workstation `uv sync` inside
this image: it would replace NVIDIA's stack.
The resolved package list is saved at `/opt/dinosats/runtime-packages.txt`.

Build from this repository's root on an Apptainer build host. All Sherlock jobs,
including CPU builds, use `serc`. Put temporary files on node-local
storage and the finished SIF under `$GROUP_HOME/$USER`. Allow substantial
temporary space for NGC's unpacked image.

```bash
mkdir -p "$SCRATCH/dinosats-logs"
DINOSATS_GPU_TEST=1 sbatch --gpus=1 --constraint='GPU_GEN:AMP|GPU_GEN:HPR' \
    --output="$SCRATCH/dinosats-logs/build-%j.log" scripts/build_ngc.sbatch
```

This single allocation builds the final image, checks native libraries, then
benchmarks synthetic DINOv3 inference on its GPU. For a CPU-only build, omit
`DINOSATS_GPU_TEST`, `--gpus`, and `--constraint`.

If the build host does not support fakeroot, build on a machine that does and
transfer the SIF. The recipe includes the pinned DINOv3 source, but weights and
imagery remain external. This profile covers embedding, not ONNX text export;
copy the existing browser model assets separately.

`scripts/siaya.sbatch` defaults to `$GROUP_HOME/containers/dinosats.sif`, requests
one A100 or H100 in `serc`, and checks CUDA before processing.
Recent NGC images do not support the V100s also present in `serc`.
Driver compatibility must pass the preflight on the allocated node.
NGC includes CUDA compatibility libraries; the GPU preflight verifies the
combination of container libraries and the allocated node's host driver.

```bash
mkdir -p "$SCRATCH/dinosats-logs"
sbatch --output="$SCRATCH/dinosats-logs/siaya-%j.log" scripts/siaya.sbatch
```

The job defaults to `$SCRATCH/inventory.json`, `$SCRATCH/vantor_request.csv`,
and `$SCRATCH/vantor_browse`, and writes outputs to `$SCRATCH/siaya_full`
(override with `DINOSATS_OUT` if needed). The inventory's RGB paths must point to the
transferred strip VRTs under `$SCRATCH/ard`. Pass `--inventory`, `--requests`,
or `--clouds` after the script name to override these defaults.

[Sherlock exposes compute-node
filesystems inside containers by default](https://www.sherlock.stanford.edu/docs/software/containers/apptainer/).
No explicit bindings are needed. Submit from the repository root: the job runs
the checked-out embedding script using the container's Python and libraries.
The existing NGC 26.02 image includes `hf` / `huggingface_hub` 1.4.1; no rebuild
is needed for this loader update.

Put `HF_TOKEN=hf_...` in the HPC repository's gitignored `.env`. The job uses the
image's `uv` to load this file with the existing container Python; it does not
sync or install the workstation environment. Alternatively, export `HF_TOKEN`
before submission or run `apptainer exec --cleanenv "$GROUP_HOME/containers/dinosats.sif" hf auth login`
once. Tokens are not printed or included in the image. Existing `weights/` files
are reused; missing files are downloaded through `huggingface_hub` at pinned
revisions. The backbone comes from Meta's gated repository; the dino.txt head
and tokenizer come from the same PIA-SPACE-LAB mirror used above.
Downloads are cached under `${HF_HUB_CACHE:-$SCRATCH/huggingface/hub}`.

Keep permanent copies of inputs and finished outputs outside Sherlock's purged
scratch storage. Re-submit the same command to resume after the time limit.

Inside an existing GPU allocation, the optional synthetic benchmark tests the
ViT-L/16 architecture without downloading weights:

```bash
apptainer exec --cleanenv --nv "$GROUP_HOME/containers/dinosats.sif" \
    python /opt/dinosats/scripts/check_runtime.py --benchmark
```

This measures model inference only; it does not estimate the full pipeline's
imagery I/O, PCA, compositing, or output assembly time.

## Configuration

The app reads its settings from [`web/config.json`](web/config.json), which lists the scene's files, the text model, the background prompts and the example prompts. Each path can be relative to `web/` or a full URL.

<details>
<summary>All configuration fields and URL options</summary>

| Field | What it is |
|:--|:--|
| `scene.name`, `scene.subtitle` | Place name and imagery label shown in the app |
| `scene.rgb` | RGB imagery COG, from `make_mosaic.py --web_out` |
| `scene.embeddings`, `scene.pca` | Embedding COG and its `.pca.json`, from `embed.py` |
| `scene.posbasis`, `scene.posnorm` | Optional, from `embed.py`: removes DINOv3's positional bias, a faint repeating pattern on uniform areas like open water |
| `scene.validity`, `scene.sources`, `scene.source_catalog` | Optional aligned coverage/source COGs and acquisition metadata for composite scenes |
| `model`, `vocab` | ONNX text encoder and CLIP tokenizer vocabulary |
| `background` | Background prompts for the relative score |
| `examples` | Example prompts under the search box |
| `point_vocab` | Optional precomputed prompts, from `embed_vocab.py` |

The app also accepts these URL options: `?config=<url>` loads a different config file, `?debug=true` shows developer controls, `?debias=0` keeps the positional bias, and `?ep=wasm` runs the text model on WebAssembly instead of WebGPU.

</details>

## Hosting

The app consists only of static files, so you can host the code, the data and the model on any cloud object store or web server. The server needs to use https and support HTTP Range requests, because the app reads the GeoTIFFs one block at a time. If the data is on a different origin from the app, the server also needs to send CORS headers that expose `Content-Range`. The code for the demo is deployed to GitHub Pages by [`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to `main`.

## Credits

- The imagery is from [NAIP](https://planetarycomputer.microsoft.com/dataset/naip) (USDA), accessed through the Microsoft Planetary Computer, and is in the public domain.
- The [DINOv3 and dino.txt](https://github.com/facebookresearch/dinov3) models are from Meta and are released under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- The CLIP BPE tokenizer vocabulary is redistributed by Meta under its [MIT license](https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE).
- The method for removing the positional bias is from [INSID3](https://github.com/visinf/INSID3).
- Thanks to [Blaz Rolih](https://huggingface.co/blaz-r) for introducing me to the dino.txt text encoder and for the tip about removing DINOv3's positional bias.

## Citation

If you use DinoSats, please cite it along with [DINOv3](https://github.com/facebookresearch/dinov3#citing-dinov3) and [dino.txt](https://arxiv.org/abs/2412.16334):

```bibtex
@misc{robinson2026dinosats,
  author       = {Robinson, Caleb},
  title        = {{DinoSats}: open-vocabulary search over aerial imagery in the browser},
  year         = {2026},
  howpublished = {\url{https://github.com/calebrob6/dinosats}}
}
```

## License

The code is MIT licensed (see [`LICENSE`](LICENSE)). The DINOv3 weights and everything derived from them, including the hosted embeddings and text encoder, are under the DINOv3 License.
