# Warhead + ESMDynamic scaffold evaluation

Combines:

1. Table-2 `warhead_*` count columns (`../warhead_scaffold_eval/cache/`)
2. Pocket + interaction receptor features
3. Pocket-binned ESMDynamic contact features (shell/TM–ECL/sulfur) + warhead×ESMD interactions
   (`cache/esmdynamic_receptor_features.csv`)

Hyperparameters match the baseline scaffold scripts (no Optuna).

```bash
# from package root
python run_warhead_esmd_scaffold.py
```

Or per model:

```bash
python ML_code/warhead_esmd_scaffold_eval/scaffold_eval_warhead_esmd.py --model rf
```
