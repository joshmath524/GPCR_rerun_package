# ESMDynamic — 68 GPCR panel (local run, no Colab)

Please predict **dynamic contact maps** for 68 Class A GPCR sequences using **ESMDynamic**.

Paper / code: https://github.com/ShuklaGroup/esmdynamic  
Data / weights: https://databank.illinois.edu/datasets/IDB-0355017  
DOI: https://doi.org/10.1038/s41467-026-76361-2

---

## What is in this folder

| File | Use |
|------|-----|
| `gpcr_68_panel.csv` | **Input** — columns `id`, `sequence` (68 receptors) |
| `gpcr_68_panel.fasta` | Same sequences (optional) |
| `gpcr_68_panel_metadata.csv` | Receptor name ↔ UniProt accession ↔ length |

---

## Hardware needed

- **NVIDIA GPU + CUDA** strongly preferred (8–16+ GB VRAM ideal)
- ~20 GB free disk
- CPU-only is possible but very slow for 68 proteins

---

## Setup (recommended: Docker)

```bash
git clone https://github.com/ShuklaGroup/esmdynamic.git
cd esmdynamic
docker build -t esmdynamic .
docker run --rm -it --gpus all -v "$PWD":/workspace -v /path/to/this_folder:/data esmdynamic
```

Inside the container (adjust paths):

```bash
run_esmdynamic -h
```

### Alternative: Conda (see repo README)

Follow https://github.com/ShuklaGroup/esmdynamic — Conda + CUDA PyTorch + OpenFold + `pip install git+https://github.com/ShuklaGroup/esmdynamic.git`

---

## Weights

ESMDynamic needs `esmdynamic_model_weights_V2.pt` from the Illinois Data Bank.

If auto-download fails with **HTTP 403**:

1. Download `esmdynamic_model_weights_V2.pt` in a browser from  
   https://databank.illinois.edu/datasets/IDB-0355017  
2. Copy it to the Torch hub cache as `esmdynamic.pt`:

**Linux / macOS**
```bash
mkdir -p ~/.cache/torch/hub/checkpoints
cp /path/to/esmdynamic_model_weights_V2.pt ~/.cache/torch/hub/checkpoints/esmdynamic.pt
```

**Windows (PowerShell)**
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.cache\torch\hub\checkpoints"
Copy-Item "C:\path\to\esmdynamic_model_weights_V2.pt" "$env:USERPROFILE\.cache\torch\hub\checkpoints\esmdynamic.pt"
```

(ESMFold weights are also required; Docker/setup usually fetches those.)

---

## Run command

From a machine with the env installed:

```bash
run_esmdynamic \
  --csv /path/to/gpcr_68_panel.csv \
  --output_dir /path/to/ESMDynamic_68_receptors_out \
  --device cuda \
  --save_txt \
  --save_raw_pt \
  --low_memory \
  --batch_size 1
```

Optional (more memory use):
```bash
  --save_png --save_html
```

If you hit GPU OOM, try:
```bash
  --chunk_size 32
```
or run one receptor at a time (split the CSV).

---

## Outputs to send back

Please zip and return the full `ESMDynamic_68_receptors_out/` folder, especially:

- `*.txt` / `*.csv` tables  
- `*.pt` raw prediction bundles  
- any logs if something failed  

**Note:** ESMDynamic does **not** write Excel (`.xlsx`). CSV/TXT is fine; we can convert later.

---

## Expected runtime

Roughly **1–4+ hours** for all 68 on a decent NVIDIA GPU (T4/RTX-class), longer if memory-constrained.

Thank you!
