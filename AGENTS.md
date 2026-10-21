# DinoSats project workflow

## Scope and entry points

- The current goal is the full Siaya pre/endline interactive search map using
  processed Vantor imagery. `scripts/build_siaya_full.py` precomputes embeddings;
  the browser scores new text queries against them.
- `scripts/siaya.sbatch` runs the full pipeline on Sherlock. Submit from the
  repository root with `sbatch scripts/siaya.sbatch`; defaults belong in the
  script, not in a list of exports the user must remember.
- The pipeline uses RGB imagery at 0.5 m/pixel and produces 4 m embedding tokens.
  It shares PCA and quantization across three pre and six endline strips.
  Calibration, positional correction, and vocabulary embeddings are generated
  for the run; do not reuse pilot calibration accidentally.
- `scripts/build_siaya.py` is the pilot builder. `serve.py --port 8787` serves
  the local map. Keep the full-run and pilot assets distinct.

## Sherlock and containers

- Access Sherlock with `ssh sherlock`. Read `/etc/agents/AGENTS.md` and the
  relevant topic files at the start of each remote session and after compaction.
  Use a login shell when commands need `$SCRATCH`, `$GROUP_HOME`, or `sh_part`.
- Always use the `serc` partition, including CPU builds. Never use `normal`.
  Prefer the user's existing allocation. When the user is running jobs, provide
  commands rather than submitting additional verification jobs automatically.
- Builds, ML imports, inference, and GPU tests belong on compute nodes.
  Wait at least 60 seconds between scheduler checks. Pending `(None)` alone
  does not justify cancellation or duplicate submissions.
- The working image path is `$GROUP_HOME/containers/dinosats.sif`. The recipe is
  `containers/ngc.def`; `scripts/build_ngc.sbatch` builds it. Set `DINOSATS_IMAGE`
  explicitly when rebuilding this path: the build script's default differs.
- Follow `../remote_outcomes/infrastructure/build_sif.sbatch` as the build
  reference, keeping that project unchanged. Use 64 GB RAM, node-local temporary
  extraction, persistent download cache, and publish the staged SIF only after
  checks pass. Build and GPU-test in one allocation when both are needed.
- Inside an existing allocation, use `bash scripts/build_ngc.sbatch` from the
  repo root. `#SBATCH` directives do not allocate resources when run with bash.
  `DINOSATS_GPU_TEST=1` enables the final image's GPU benchmark.
- Use `--nv` for GPU access, including `nvidia-smi` inside the container.
  Preserve Slurm's `CUDA_VISIBLE_DEVICES` across `--cleanenv`. This image targets
  A100/H100-class GPUs; do not assume V100 compatibility.
- [Sherlock mounts compute-node filesystems by default](https://www.sherlock.stanford.edu/docs/software/containers/apptainer/).
  Do not add redundant data bindings. The job executes checked-out repo code
  with the image's Python, so code-only updates do not require image rebuilds.
- Preserve NGC's Torch/CUDA stack. Do not run the workstation `uv sync` inside
  the image. Container additions are in `containers/ngc-requirements.txt`.

## Inputs, credentials, and outputs

The established HPC layout is below; verify existence before claiming readiness.

| Purpose | Path |
| --- | --- |
| Processed GeoTIFFs and strip VRTs | `$SCRATCH/ard/` |
| Nine-strip inventory | `$SCRATCH/inventory.json` |
| Acquisition metadata | `$SCRATCH/vantor_request.csv` |
| Cloud masks | `$SCRATCH/vantor_browse/` |
| Full-run outputs and checkpoints | `$SCRATCH/siaya_full/` |
| Hugging Face model cache | `$SCRATCH/huggingface/hub/` |
| Hugging Face Xet cache | `$SCRATCH/huggingface/xet/` |

- `data/` is gitignored. The required full-run input from it is
  `data/siaya/inventory.json`, relocated to the scratch path above with updated
  `rgb` paths. Check nested VRT source references too. The pilot manifest and
  `data/siaya/cpu64/` outputs are not full-run prerequisites.
- Source imagery stays unchanged. Raw NASA deliveries are not embedding inputs.
  Keep permanent copies outside purged scratch. Changed input paths or code
  fingerprints can require a fresh output directory rather than resuming.
- `scripts/dinotxt.py` reuses local weights or downloads pinned HF revisions.
  The backbone is Meta's official DINOv3 ViT-L/16 LVD-1689M. The dino.txt head
  and tokenizer use the existing PIA-SPACE-LAB mirror; it is not a different
  backbone. Do not substitute the SAT backbone: text alignment differs.
- The image was verified to include `hf` / `huggingface_hub` 1.4.1. The job
  loads `HF_TOKEN` from the HPC repo's `.env` using `uv --env-file`, with the
  existing container Python. Never print tokens, commit `.env`, or bake secrets
  into images. No weights binding or `DINOSATS_WEIGHTS` export is needed.
- Browser assets under `web/models/` are separate from embedding inputs and are
  gitignored. Check them before serving the map, not as a GPU-job prerequisite.
- The local full embedding job was stopped and its output deleted at the user's
  request. Focus on HPC; do not restart local production inference implicitly.

## Validation and handoff

- Distinguish image metadata inspection, CPU imports, CUDA kernel execution,
  synthetic benchmarks, and inference with real weights. Do not report a queued
  check or `torch.cuda.is_available()` alone as successful GPU validation.
- Base ETAs on measured windows/second and remaining work. Candidate tiles can
  be empty; synthetic GPU timings exclude imagery I/O, PCA, and map assembly.
- Keep scripts simple. Validate shell edits with `bash -n` and changes with
  `git diff --check`; use focused smoke checks rather than broad test scaffolds.
- When pushing requested HPC changes, confirm the push completed and give the
  user the commit and minimal pull/run commands. Do not edit `.README.md.swp`.

## Coding conventions

Here are some instructions when writing code:

- We use `uv` for Python environments. Make sure you always use it to install
  packages

- When writing something intended for human consumption, (comment, commit
  message, reply to prompt) use as few words as possible. Pick every word
  meticulously to reduce the volume to a strict minimum. Be down to the point.
  Less is more.

- Avoid superlatives and praise. Stop telling me I am absolutely right. Give me
  the cold hard truth.

- Avoid magic numbers and strings by extracting recurring or meaningful values
  into descriptive constants or enums. Keep self-explanatory, one-off
  values inline to avoid clutter. If a value comes from a spec (e.g. HTTP 200
  OK), use a constant regardless.

- Keep function names short. Less than 30 characters.

- Let the reader of the code breathe. Add empty lines between logical blocks of
  code.

- Add a small, to the point, comment to explain *what* the block does and
  *why*. Use examples when possible. Propose ASCII drawings to explain complete
  systems.

- Program to levels of abstraction. Lower-level mechanics (e.g., raw hardware
  I/O, sector parsing, direct socket streams) must be encapsulated in
  a dedicated driver/abstraction layer. Expose clean, high-level APIs to the
  rest of the application so calling code works with domain concepts, not raw
  implementation details.

- Don't touch blocks of code unrelated to the feature you implement. e.g. Don't
  add comments to a block of code if you did not create it or modify it. As
  much as possible try to minimize the number of changed lines when
  implementing a feature.

- Strictly adhere to the layered boundary hierarchy: each layer may only
  communicate with its immediate neighbor directly below it. Never "punch
  holes" through layers (e.g., controllers or UI components must never directly
  call database queries, raw hardware drivers, or low-level network clients;
  always route through the intermediate service/abstraction layer).

- Follow Google's [Python coding guide](https://google.github.io/styleguide/pyguide.html)
  and make sure to use `ruff` at the end of an editing session. We can use the
  default `ruff` settings or whatever adjusts to the Google guide

When you write a commit message, follow these 7 rules:
 - Rule 1: Separate the subject line from the body with a single blank line.
 - Rule 2: Limit the subject line to 50 characters (72 is the absolute hard limit).
 - Rule 3: Capitalize the first letter of the subject line.
 - Rule 4: Do not end the subject line with a period.
 - Rule 5: Use the imperative mood in the subject line (e.g., "Fix bug," "Add
   feature," not "Fixed" or "Adds"). Test formula: It must complete the
   sentence: "If applied, this commit will [your subject line here]".
 - Rule 6: Wrap the body text manually at 72 characters to prevent Git
   formatting issues.
 - Rule 7: Use the body to explain what and why vs. how. Assume the code
   explains the how; the message must explain the context and reasoning.

- If the prompt indicates that a bug is being fixed, don't write the fix right
  away. First write the test. Observe it failing. Then write the fix. And
  observe the test passing.
