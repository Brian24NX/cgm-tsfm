# Running on the lab server (cpsl-mds) — migration guide

Repo: **`Brian24NX/cgm-tsfm`** (private). You're on the `cpsl-mds` lab server so training runs on GPU.

> ## ✅ Already built & validated on `cpsl-mds` (2026-07-06)
> The numbered setup below has **already been run on this server** — you normally don't need to redo it. To start working, just activate the env:
> ```bash
> export WS=/data_1_8TB_ssd/brian_workspace
> source ~/.bashrc                      # sets WS/HF_HOME/… + sources conda (block already added)
> conda activate "$WS/envs/cgm"
> # …or skip activation entirely and call the env directly:
> "$WS/envs/cgm/bin/python" -m cgm_tsfm.run_demo --encoder mock
> ```
> **Installed (all on the SSD):** Miniforge → `$WS/miniforge3`; prefix env → `$WS/envs/cgm` (Python 3.11.15); `torch 2.6.0+cu124`, `chronos-forecasting 2.3.1`, `lightning 2.6.5`, numpy/pandas/scikit-learn/scipy. **GPU:** NVIDIA **RTX 6000 Ada** (49 GB); driver reports CUDA ≤13.1 (backward-compatible with the cu124 wheels).
> **Validated (2026-07-06):** both `run_demo --encoder mock` (offline) **and** `run_headtohead --encoder chronos --with-arm-b --device cuda` pass end-to-end — the latter loads real Chronos-Bolt on the GPU (512-d embeddings) and trains Arm B over grouped folds. (Synthetic numbers are a plumbing check, not evidence.)
> Redo the numbered steps below only to rebuild from scratch or provision a different machine.

## ⚠️ STORAGE POLICY (Liuyi) — read first
On `cpsl-mds`, put **everything heavy** — the repo, the conda env, model/HuggingFace downloads, the data, and caches — under:

```
/data_1_8TB_ssd/brian_workspace          # the big SSD (/dev/sda1, ~870 GB free)
```

**Do NOT** put heavy things under `/home/brian` — it's on `/`, a smaller shared spinning disk (~300 GB free, used by the whole lab). The conda env + model downloads alone are several GB, so this matters. Every command below keeps things on the SSD.

> Nothing was "lost" by cloning the repo: `/data/` was empty (we don't have the real CSVs yet) and `.cache/` (embeddings + logs) regenerates on first run. Data and caches are **never** in git — data because it's patient data (privacy), caches because they rebuild.

## 1. Put the repo on the SSD
```bash
WS=/data_1_8TB_ssd/brian_workspace
mkdir -p "$WS"

# If you already cloned it under /home, just MOVE it onto the SSD:
mv ~/cgm-tsfm "$WS"/ 2>/dev/null

# …otherwise clone it fresh onto the SSD:
cd "$WS"
gh auth login                         # once, if needed; or use an SSH key
gh repo clone Brian24NX/cgm-tsfm

cd "$WS/cgm-tsfm"
```

## 2. Send all caches to the SSD (so /home never fills up)
Add to `~/.bashrc` so it persists across logins, then `source ~/.bashrc`. **This block is already in `~/.bashrc` on `cpsl-mds`** (added 2026-07-06); it also sources Miniforge (step 3) so `conda` works in every login shell:
```bash
export WS=/data_1_8TB_ssd/brian_workspace
export HF_HOME="$WS/hf_cache"            # HuggingFace model downloads (Chronos) — can be GBs
export PIP_CACHE_DIR="$WS/pip_cache"
export CONDA_PKGS_DIRS="$WS/conda_pkgs"  # conda package cache

# Miniforge on the SSD → enables `conda` + `conda activate` in interactive shells:
if [ -f "$WS/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$WS/miniforge3/etc/profile.d/conda.sh"
fi
```
> Ubuntu's default `~/.bashrc` **returns early for non-interactive shells**, so `ssh host <cmd>` and plain scripts won't pick these up. In a script, either `source` the block explicitly or just call the env by absolute path (`"$WS/envs/cgm/bin/python"`).

## 3. Conda (Miniforge) + a PREFIX env — all on the SSD
**`cpsl-mds` has no conda preinstalled**, so first install Miniforge *onto the SSD* (this keeps base conda off `/home` too). Miniforge is the conda-forge, BSD-licensed distro — the `-b` batch install has no prompts and no Anaconda ToS gate:
```bash
curl -fL -o "$WS/Miniforge3-Linux-x86_64.sh" \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash "$WS/Miniforge3-Linux-x86_64.sh" -b -p "$WS/miniforge3"   # -b -p → non-interactive, onto the SSD
rm "$WS/Miniforge3-Linux-x86_64.sh"
```
Then make a **prefix** env inside the workspace (a normal `conda create -n <name>` would land under `/home`):
```bash
"$WS/miniforge3/bin/conda" create -y -p "$WS/envs/cgm" python=3.11
conda activate "$WS/envs/cgm"          # needs step 2's conda.sh sourced (open a new shell / source ~/.bashrc)
```
> In a non-interactive shell where `conda activate` isn't available, just call the env directly:
> `"$WS/envs/cgm/bin/python"` / `"$WS/envs/cgm/bin/pip"`. The install commands below assume one of the two.

## 4. Install PyTorch (GPU) + the rest
Check the server's CUDA with `nvidia-smi`, then install a matching torch wheel **first**. On `cpsl-mds` the driver reports CUDA ≤13.1 (backward-compatible), so the well-tested **cu124** wheel is a safe choice — this is what's installed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"   # want True
```
> - If a wheel download dies with `SSL: SSLV3_ALERT_HANDSHAKE_FAILURE` to `download-r2.pytorch.org`, it's a transient Cloudflare/CDN blip — just re-run the `pip install` (add `--retries 10`). The env's TLS stack is fine (system `curl` and the env's Python both handshake with that host).
> - `pip install -r requirements.txt` sees `torch>=2.4` already satisfied and leaves the GPU build in place — re-run the CUDA check above afterwards to confirm it wasn't swapped for a CPU wheel.

## 5. Data (when you get it from Liuyi)
Real CSVs are **not** in git (patient data). Copy them **directly to the server** (scp / lab share) — **not** through GitHub — into the repo's data folder (which is on the SSD because the repo is):
```
$WS/cgm-tsfm/data/Merged_glucose_data/
    Cohort1_scores_merged_with_glucose.csv
    Cohort2_scores_with_glucose.csv
```
Never commit or paste patient data. Not needed for the synthetic runs below.

## 6. Run (on the GPU)
```bash
python -m cgm_tsfm.run_demo --encoder mock                                          # offline smoke test
python -m cgm_tsfm.run_headtohead --encoder chronos --with-arm-b --device cuda      # GPU, synthetic
# real runs, once Step 5's CSVs are in place:
python -m cgm_tsfm.run_sweep      --kind all --real --device cuda
python -m cgm_tsfm.run_headtohead --encoder chronos --real --with-arm-b --pca 32 --device cuda
```
Add `--device cuda` to any command to use the GPU (default is `cpu`). The pipeline's embedding cache (`.cache/` inside the repo) sits on the SSD too. Results land in `results/*.md`.

## Notes
- Optional laptop-only files (grant PDF, papers, Liuyi's `.docx`) were intentionally kept out of git; copy them over separately only if you actually want them on the server.
- For long training jobs use the lab's scheduler (SLURM `sbatch`/`srun`) rather than a login node — ask Liuyi for the convention.
- Caches already point at the SSD (`HF_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS`), so first-run Chronos downloads and pip installs won't touch `/home`.
