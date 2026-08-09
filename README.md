# GRASPE — GRaph-based Adaptive SParse Encoder

Reference implementation of **GRASPE**, a lightweight graph-based framework for **Irregular Multivariate
Time Series Forecasting (IMTS)**.

The model operates directly on raw observations: no imputation, no alignment to a regular grid, no patching.
Each observed measurement `(value, timestamp, channel)` becomes a node of a **sparse per-sample graph**;
edges connect temporally consecutive observations of the same channel (*intra-channel*) plus a minimal set of
*bridge* edges across channels. A lightweight GNN (GAT/GCN) propagates context over the graph, an
attention-based **Query Pooling** readout produces one fixed-size vector per variable, and a sparse decoder
maps it to forecasts at arbitrary future timestamps.

Benchmarks: **PhysioNet**, **MIMIC-III**, **Human Activity**, **USHCN**.

Pipeline (see `src/layers/grape.py`):

```
(v, t_in, t_out, c, m) → observation/time/channel encoders → sparse graph construction
                       → GNN → Query Pooling → time-conditioned decoder → ŷ(t_out)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Built on PyTorch 2.8 + CUDA 12.8, `torch-geometric`, `pytorch-lightning`. All commands below are meant to be
run from the repository root with the `-m` flag, so that `src` is importable.

## Data

Datasets live in `data/<name>/{raw,processed}`.

| Dataset | Key | Download |
|---|---|---|
| PhysioNet Challenge 2012 | `physionet` | automatic (`download=True`) |
| Human Activity (UCI) | `activity` | automatic |
| USHCN | `ushcn` | place raw files in `data/ushcn/raw` |
| MIMIC-III | `mimic` | credentialed access; preprocess with `src/dataset/prepare_mimic_data.py` (after the GRU-ODE-Bayes preprocessing notebooks) |

## Repository structure

```
src/
  config.py                 # Parameters: single place for all hyperparameters, YAML load/save
  layers/
    grape.py                # GRASPE model (Grape) + QueryPooling; GrapeDgm variant
    utils.py                # build_graph_from_mask_v2 (sparse graph construction), select_gnn
    decoder.py              # decoders: simple, film, gated, filmSwiglu, gru, crossAttn, INR
  dataset/
    parse_datasets.py       # dataloaders + collate for irregular batches
    physionet.py, mimic.py, ushcn.py, person_activity.py
    prepare_mimic_data.py   # MIMIC-III preprocessing
    evaluation.py, utils.py, eda.py
  training/
    training_module.py      # model wrapper, loss (masked MSE), metrics
    train.py                # train/test loops, early stopping, LR scheduling
  scripts/
    run_train.py            # single run
    run_grid_search.py      # multi-seed grid search
  utils/
    utils.py                # model/datamodule factory, metric aggregation
registry/
  configurations/           
  logs/                     
  experiments/              
  checkpoints/
data/                       # raw + processed datasets
```

## Running

### Single training run

Edit the defaults in `src/config.py` (`dataset_name`, `model`, `hid_dim`, `gnn_name`, `decoder_name`,
`inner_mode`, …), then:

```bash
python -m src.scripts.run_train
```

### Grid search (multi-seed)

Edit `search_space`, `global_config` and `seed_list` in `main()` of `src/scripts/run_grid_search.py`:

```bash
python -m src.scripts.run_grid_search
```

Every combination is trained on all seeds; mean/std of val/test MSE, RMSE and MAE are appended to
`registry/logs/<timestamp>/log.txt`, together with the search space and the static parameters used.



