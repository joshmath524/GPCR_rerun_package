# ESMDynamic for the 68 GPCR panel

## What is ready locally

| File | Purpose |
|------|---------|
| `gpcr_68_panel.fasta` | Sequences for all 68 receptors |
| `gpcr_68_panel.csv` | Same data (id, sequence) for `run_esmdynamic` |
| `gpcr_68_panel_metadata.csv` | Receptor ↔ UniProt accession ↔ length |
| `ESMDynamic_GPCR68_Colab.ipynb` | Colab notebook to run inference |

All 68 sequences downloaded from UniProt (length 300–590 aa; under Colab T4 ~900 aa limit).

## Why Colab (not this laptop)

Your PC has AMD Vega 8 (~1 GB) and limited free disk. ESMDynamic needs an NVIDIA GPU. Use Google Colab.

## What Cursor cannot do

Cursor **cannot** log into your Google account or press Run in Colab for you. You must open the notebook once.

## Steps (you do this — ~5 min setup, ~1–4 h runtime)

1. Open https://colab.research.google.com/
2. **File → Upload notebook** → choose  
   `ESMDynamic_68_receptors/ESMDynamic_GPCR68_Colab.ipynb`
3. **Runtime → Change runtime type → GPU (T4)**
4. Run cells in order:
   - Check GPU
   - Install (~3–5 min)
   - Upload `gpcr_68_panel.csv` from this folder
   - (Optional) turn on Drive saving
   - Run inference
   - Download the zip
5. Extract the zip into this `ESMDynamic_68_receptors/` folder

Tip: set `USE_DRIVE = True` in cell 4 so results survive if Colab disconnects.

## Official single-protein Colab (backup)

https://colab.research.google.com/github/ShuklaGroup/esmdynamic/blob/main/examples/esmdynamic/esmdynamic.ipynb

## Citation

Kleiman et al., *Nat Commun* (2026). ESMDynamic.  
https://doi.org/10.1038/s41467-026-76361-2
