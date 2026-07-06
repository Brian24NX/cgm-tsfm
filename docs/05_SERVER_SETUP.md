# Running on the lab server (cpsl-mds) — migration guide

Repo: **`Brian24NX/cgm-tsfm`** (private). You're on the `cpsl-mds` lab server so training runs on GPU.

## ⚠️ STORAGE POLICY (Liuyi) — read first
On `cpsl-mds`, put **everything heavy** — the repo, the conda env, model/HuggingFace downloads, the data, and caches — under:

```
/data_1_8TB_ssd/brian_workspace          # the big SSD, ~1 TB free
```

**Do NOT** put heavy things under `/home/brian` (≤300 GB, shared by everyone). The conda env + model downloads alone are several GB, so this matters. Every command below keeps things on the SSD.

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
Add to `~/.bashrc` so it persists across logins, then `source ~/.bashrc`:
```bash
export WS=/data_1_8TB_ssd/brian_workspace
export HF_HOME="$WS/hf_cache"            # HuggingFace model downloads (Chronos) — can be GBs
export PIP_CACHE_DIR="$WS/pip_cache"
export CONDA_PKGS_DIRS="$WS/conda_pkgs"  # conda package cache
```

## 3. Conda environment — on the SSD via a PREFIX path
A normal `conda create -n <name>` env lands under `/home`. Use a **prefix** env inside the workspace instead so it lives on the SSD:
```bash
conda create -y -p "$WS/envs/cgm" python=3.11
conda activate "$WS/envs/cgm"
```

## 4. Install PyTorch (GPU) + the rest
Check the server's CUDA with `nvidia-smi`, then install a matching torch wheel **first** (example = CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"   # want True
```

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
