# FMSD-Net

Federated Multimodal Sensor Decision Network for heterogeneous greenhouse sensor clients. The implementation follows the manuscript modules: the Heterogeneous Multimodal Sensor Encoder (HMSE), Non-IID-Aware Federated Aggregation (NFA), and the Sensor-to-Decision Prediction Module (SDPM).

Local clients keep environmental sequences, soil sequences, canopy images, and structured management inputs on site. The coordinator receives parameter updates, a four-dimensional descriptor, and the mean and diagonal covariance of the fused representation. Reported runs use NFA-Fixed. NFA-Learned is available as the scoring-network formulation and is not the default.

## Environment

Python 3.10 is required. PyTorch 2.2 or newer is installed from `requirements.txt`.

```bash
conda create -n fmsd python=3.10
conda activate fmsd
pip install -r requirements.txt
```

The default device setting is `auto`: CUDA is used when available, then Apple MPS, then CPU. Pass `--device` to select one explicitly.

## Layout

```text
configs/          configuration
src/data/         repair, scaling, and loaders
src/models/       HMSE, NFA, SDPM, FMSD-Net
src/losses/       multi-task objective
src/engine/       federated training and evaluation
src/utils/
tools/            entry points
outputs/          checkpoints and metrics
```

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
