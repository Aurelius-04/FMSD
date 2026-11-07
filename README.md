# FMSD-Net

Federated Multimodal Sensor Decision Network for heterogeneous greenhouse sensor clients. The implementation follows the manuscript modules: the Heterogeneous Multimodal Sensor Encoder (HMSE), Non-IID-Aware Federated Aggregation (NFA), and the Sensor-to-Decision Prediction Module (SDPM).

Local clients keep environmental sequences, soil sequences, canopy images, and structured management inputs on site. The coordinator receives parameter updates, a four-dimensional descriptor, and the mean and diagonal covariance of the fused representation. Reported runs use NFA-Fixed. NFA-Learned is available as the scoring-network formulation and is not the default.

## Environment

The code targets Python 3.10 and PyTorch 2.2 or newer. On this Mac the existing `py310` environment is sufficient. On Linux the same environment uses CUDA when `torch.cuda.is_available()` is true. Device selection is `auto`: CUDA, then Apple MPS, then CPU.

```bash
conda activate py310
pip install -r requirements.txt
```

## Layout

```text
configs/          configuration
dataset/          client records, not included in this repository
src/data/         repair, scaling, and loaders
src/models/       HMSE, NFA, SDPM, FMSD-Net
src/losses/       multi-task objective
src/engine/       federated training and evaluation
src/utils/
tools/            entry points
outputs/          checkpoints and metrics
```

## Data

Place one NumPy archive per client in `dataset/`:

```text
dataset/changping.npz
dataset/baoding.npz
dataset/weifang.npz
```

Each archive stores `environmental` with shape `(N, 336, 4)`, `soil` with shape `(N, 336, 3)`, `canopy` with shape `(N, 3, 224, 224)`, `structured` with shape `(N, 5)`, plus `price`, `yield`, `cost`, `net_return`, `season`, `cycle_id`, and `modality_observed`. The 2022 season is the training partition, 2023 is validation, and 2024 is the hold-out test. Windows that share a `cycle_id` share one compartment-season target. The loader repairs missing sensor steps, replaces outliers, and fits z-score statistics on the local 2022 partition. Raw facility records are not in this repository; they can be requested from the corresponding author as stated in the manuscript.

## Run

Shorter run:

```bash
python tools/train.py --config configs/default.yaml
python tools/evaluate.py --config configs/default.yaml --checkpoint outputs/last.pt
```

Manuscript budget (100 communication rounds, 5 local epochs, batch size 32):

```bash
python tools/train.py --config configs/manuscript.yaml
```

Useful overrides:

```bash
python tools/train.py --device cpu --rounds 2 --local-epochs 1 --batch-size 2
python tools/train.py --aggregation nfa_learned --rounds 1
```

Metrics are written to `outputs/metrics.json`. `outputs/last.pt` is the final global model. `outputs/best.pt` is the lowest validation-loss round. Net-return metrics are reported at window level and at compartment-season level. Decision scores use the manuscript thresholds and map to Expand Inputs / Investment, Maintain Current Inputs, or Reduce Risk / Cut Inputs.
