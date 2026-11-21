# Warhead + ESMDynamic scaffold evaluation

Combines:

1. Curated warhead ligand features (`../warhead_scaffold_eval/cache/`)
2. Pocket + interaction receptor features
3. ESMDynamic dynamic-contact receptor features (`cache/esmdynamic_receptor_features.csv`)

Hyperparameters match the baseline scaffold scripts (no Optuna).

```bash
# from package root
python run_warhead_esmd_scaffold.py
```

Or per model:

```bash
python ML_code/warhead_esmd_scaffold_eval/scaffold_eval_warhead_esmd.py --model rf
```
