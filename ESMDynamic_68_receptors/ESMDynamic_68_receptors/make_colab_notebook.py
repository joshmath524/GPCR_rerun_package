import json
from pathlib import Path

nb_path = Path(
    r"c:\Users\joshmatchem\Music\GPCRtryagain - Delete - Copy\ESMDynamic_68_receptors\ESMDynamic_GPCR68_Colab.ipynb"
)

cells = []


def md(s: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in s.splitlines()],
        }
    )


def code(s: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in s.splitlines()],
        }
    )


md(
    """# ESMDynamic — 68 GPCR panel

Predict dynamic contact maps for the manuscript **68-receptor** Class A GPCR panel.

### How to open
1. Go to https://colab.research.google.com/
2. File → Upload notebook → select `ESMDynamic_GPCR68_Colab.ipynb`
3. Runtime → Change runtime type → **GPU (T4)**
4. Run cells in order. Upload `gpcr_68_panel.csv` when asked.

### Note on weights
Illinois Data Bank often returns **HTTP 403** for automatic downloads. Cell 5 places weights in the torch cache or asks you to upload `esmdynamic_model_weights_V2.pt` (~121 MB).
"""
)

code(
    """#@title 1) Check GPU
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: No GPU. Enable Runtime → Change runtime type → GPU.')
"""
)

code(
    """#@title 2) Install ESMDynamic (~3-5 min)
%%time
import os, time
model_name = 'esmfold.model'
os.chdir('/content')

if not os.path.isfile(model_name):
    os.system('apt-get install aria2 -qq')
    os.system(f'aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/{model_name} &')
    os.system('aria2c -q -x 16 --user-agent=\"Mozilla/5.0\" https://databank.illinois.edu/datafiles/7odsk/download --out esmdynamic.pt &')

    if not os.path.isfile('finished_install'):
        print('installing libs...')
        os.system('pip install -q mdtraj omegaconf pytorch_lightning biopython ml_collections einops py3Dmol modelcif')
        os.system('pip install -q git+https://github.com/NVIDIA/dllogger.git')
        print('installing openfold...')
        os.system('pip install -q git+https://github.com/sokrypton/openfold.git')
        print('installing esmdynamic...')
        os.system('pip install -q git+https://github.com/ShuklaGroup/esmdynamic.git')
        os.system('touch finished_install')

    while not os.path.isfile(model_name):
        time.sleep(5)
        while os.path.isfile(f'{model_name}.aria2'):
            print('downloading ESMFold params...')
            time.sleep(5)

for _ in range(60):
    if os.path.isfile('esmdynamic.pt') and not os.path.isfile('esmdynamic.pt.aria2'):
        break
    time.sleep(5)

print('Install done.')
print('ESMFold:', os.path.isfile(model_name))
sz = os.path.getsize('esmdynamic.pt') if os.path.isfile('esmdynamic.pt') else 0
print('ESMDynamic weights file size bytes:', sz)
if sz < 50_000_000:
    print('NOTE: weights file missing/too small. Cell 5 will ask you to upload it.')
"""
)

code(
    """#@title 3) Upload gpcr_68_panel.csv
from google.colab import files
import os, shutil
import pandas as pd

uploaded = files.upload()  # choose gpcr_68_panel.csv
assert any(n.endswith('.csv') for n in uploaded), 'Please upload gpcr_68_panel.csv'
src = [n for n in uploaded if n.endswith('.csv')][0]
os.makedirs('/content/input', exist_ok=True)
shutil.move(src, '/content/input/gpcr_68_panel.csv')
print('Saved to /content/input/gpcr_68_panel.csv')

df = pd.read_csv('/content/input/gpcr_68_panel.csv')
print(df.head())
print('n_receptors =', len(df))
print('length range:', int(df['sequence'].str.len().min()), '-', int(df['sequence'].str.len().max()))
assert len(df) == 68, f'Expected 68 rows, got {len(df)}'
"""
)

code(
    """#@title 4) Optional — save outputs to Google Drive
USE_DRIVE = False  #@param {type:'boolean'}
DRIVE_OUT = '/content/drive/MyDrive/ESMDynamic_68_receptors'

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    os.makedirs(DRIVE_OUT, exist_ok=True)
    print('Will write to', DRIVE_OUT)
else:
    print('Outputs stay in /content/ESMDynamic_68_receptors')
"""
)

code(
    """#@title 5) Fix weights (403 bypass) + run all 68 receptors
import os, shutil, subprocess
from pathlib import Path
import torch

# Illinois Data Bank blocks torch.hub downloads (HTTP 403).
# Place weights in torch hub cache so run_esmdynamic skips the download.
CACHE = Path('/root/.cache/torch/hub/checkpoints')
CACHE.mkdir(parents=True, exist_ok=True)
dest = CACHE / 'esmdynamic.pt'

def ok(p, mb=50):
    return os.path.isfile(p) and os.path.getsize(p) >= mb * 1024 * 1024

if not ok(dest):
    for src in ['/content/esmdynamic.pt', '/content/esmdynamic_model_weights_V2.pt']:
        if ok(src):
            shutil.copy(src, dest)
            print('Copied', src, '->', dest)
            break

if not ok(dest):
    print('Retrying Illinois Data Bank download via aria2...')
    os.system('aria2c -x 16 -s 16 --allow-overwrite=true --user-agent=\"Mozilla/5.0\" -d /content -o esmdynamic.pt \"https://databank.illinois.edu/datafiles/7odsk/download\"')
    if ok('/content/esmdynamic.pt'):
        shutil.copy('/content/esmdynamic.pt', dest)

if not ok(dest):
    print('AUTO DOWNLOAD BLOCKED (403).')
    print('On your PC open: https://databank.illinois.edu/datasets/IDB-0355017')
    print('Download esmdynamic_model_weights_V2.pt (~121 MB), then upload it below.')
    from google.colab import files
    up = files.upload()
    src = list(up.keys())[0]
    shutil.move(src, '/content/esmdynamic.pt')
    assert ok('/content/esmdynamic.pt'), 'Uploaded file looks too small / corrupt'
    shutil.copy('/content/esmdynamic.pt', dest)

print('Weights ready:', dest, 'MB=', round(dest.stat().st_size / 1e6, 1))

out_dir = '/content/drive/MyDrive/ESMDynamic_68_receptors' if ('USE_DRIVE' in globals() and USE_DRIVE) else '/content/ESMDynamic_68_receptors'
os.makedirs(out_dir, exist_ok=True)

cmd = [
    'run_esmdynamic',
    '--csv', '/content/input/gpcr_68_panel.csv',
    '--output_dir', out_dir,
    '--device', 'cuda' if torch.cuda.is_available() else 'cpu',
    '--save_txt',
    '--save_png',
    '--save_raw_pt',
    '--low_memory',
    '--batch_size', '1',
]
print('Running:', ' '.join(cmd))
print('Expect ~1-4 hours on free T4. Keep the tab awake.')
proc = subprocess.run(cmd, capture_output=True, text=True)
print((proc.stdout or '')[-8000:])
print((proc.stderr or '')[-8000:])
print('returncode', proc.returncode)
print('Files written:', len(os.listdir(out_dir)))
print(sorted(os.listdir(out_dir))[:40])
"""
)

code(
    """#@title 6) Zip and download results
import os, shutil
from google.colab import files

out_dir = '/content/drive/MyDrive/ESMDynamic_68_receptors' if ('USE_DRIVE' in globals() and USE_DRIVE) else '/content/ESMDynamic_68_receptors'
zip_base = '/content/ESMDynamic_68_receptors_results'
shutil.make_archive(zip_base, 'zip', out_dir)
zip_path = zip_base + '.zip'
print('Created', zip_path, 'size_MB', round(os.path.getsize(zip_path) / 1e6, 1))
files.download(zip_path)
print('Extract the zip into: .../GPCRtryagain - Delete - Copy/ESMDynamic_68_receptors/')
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote", nb_path)
