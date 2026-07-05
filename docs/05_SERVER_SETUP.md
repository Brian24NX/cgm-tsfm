# Running on the lab server (migration guide)

You're moving off the laptop so training/compute runs on the lab server. This is the full pull-and-run recipe. Repo: **`Brian24NX/cgm-tsfm`** (private).

> The `~/Desktop/mod-actigraphy-advanced/.venv/...` interpreter path in some older examples was my **laptop's** environment — ignore it on the server. On the server you make your own venv (Step 2) and just use `python`.

## 1. Get the code (clone the private repo)
The repo is private, so the server needs to authenticate to GitHub. Easiest is the GitHub CLI:

```bash
gh auth login                     # interactive: pick GitHub.com → HTTPS → paste device code
gh repo clone Brian24NX/cgm-tsfm
cd cgm-tsfm
```

No `gh` on the server? Use SSH instead — add an SSH key to your GitHub account (Settings → SSH keys), then:
```bash
git clone git@github.com:Brian24NX/cgm-tsfm.git && cd cgm-tsfm
```

Later, to pull updates you push from the laptop: `git pull`.

## 2. Python environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
```
**If the server has GPUs (it should — that's the point):** install a CUDA build of PyTorch that matches the server's CUDA **before** the rest. Check the CUDA version with `nvidia-smi`, then grab the matching wheel from https://pytorch.org (example for CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Then the rest:
```bash
pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"   # want True
```

## 3. Model weights (Chronos) download
The first real run downloads the Chronos checkpoint from HuggingFace (needs internet). If compute nodes have no internet, run it once on a login node, or point the cache at a shared location:
```bash
export HF_HOME=$HOME/.cache/huggingface     # optional; persists downloads across jobs
```

## 4. Put the study data (when you have it)
The real CSVs are **not** in the repo (gitignored for privacy). Transfer them to the server yourself (e.g. `scp`, or the lab file share) into:
```
cgm-tsfm/data/Merged_glucose_data/
    Cohort1_scores_merged_with_glucose.csv
    Cohort2_scores_with_glucose.csv
```
Not needed for synthetic runs. (Never commit or paste patient data.)

## 5. Run
```bash
# offline smoke test (no data, no download) — confirms the install works
python -m cgm_tsfm.run_demo --encoder mock

# real Chronos on GPU, synthetic data (proves the GPU path end-to-end)
python -m cgm_tsfm.run_headtohead --encoder chronos --with-arm-b --device cuda

# THE REAL RUN — once Step 4's CSVs are in place
python -m cgm_tsfm.run_sweep      --kind all --real --device cuda
python -m cgm_tsfm.run_headtohead --encoder chronos --real --with-arm-b --pca 32 --device cuda
```
Add `--device cuda` to any command to use the GPU (default is `cpu`). Results land in `results/*.md`.

## Notes
- `.cache/` (cached embeddings) and `/data/` are **not** in git — embeddings regenerate on first run; data you provide.
- For long training jobs, launch under the lab's scheduler (e.g. `sbatch`/`srun` for SLURM) rather than interactively — ask Liuyi for the lab's convention.
