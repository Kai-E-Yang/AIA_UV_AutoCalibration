# Model card: AIA 1600/1700 Å auto-calibration CNN (refined recipe, seed 1)

**File:** `autocal_uv_1600_1700_refined_final.pt` (SHA-256 `75ce5851fc436931…`), a copy of `../run/results/autocal_uv_1600_1700_refined_final.pt`.

## Use

- **What it estimates.** The relative sensitivity α of AIA 1600 Å and 1700 Å (1 = SDOML-v2's corrected level) from one pair of full-disk frames taken within about 12 min of each other.
- **How to correct.** Divide the observed frame by α̂ to correct it.
- **Precomputed curve.** `results/released_alpha_daily.csv` and `results/released_alpha_monthly.csv` (in `final_report/`) give α̂ for the 3,826 daily pairs of 2010–2020, and by month with its scatter and seed spread.

## Input contract

The model only gives meaningful α̂ for input prepared exactly this way:

1. **Geometry and units.** SDOML-v2: Sun rescaled to 976″, 4.8″ per pixel, 512², DN/s after exposure normalisation, as observed (an SDOML-v2 frame × its `DEG_COR`).
2. **Channel order.** [1600, 1700].
3. **Subsample.** `[::2]` to 256².
4. **Scale.** ÷ 125 (1600 Å) and ÷ 1750 (1700 Å).
5. **Apodize.** Zero outside r = 100 px on the 256 grid (pixel centres at arange(256) − 128 + 0.5).

The reported numbers use frames mirrored left–right (`torch.flip(x, [2])`), as in the evaluation. For new data, mirroring changes α̂ only slightly (the tutorial prints both); do not mix the two in one study.

## Output

- α̂ per channel, from a linear head.
- Training covered α from 0.01 to 1.25; outside that range the model extrapolates.

## Training

- **Data.** Months January–July of 2010–2020 (2,176 pairs) from SDOML-v2, dimmed by random α.
- **Optimisation.** 200 epochs; Adam with warm-up and a stepped learning rate.
- **Checkpoint.** Epoch 135 (`best.pt`; `final.pt` holds the same weights), chosen by the lowest test-month (August–October) MSE.
- **Code.** `../run/02b_train_autocal_uv_refined.ipynb`; seed 1; 37 min on an Apple-silicon GPU.

## Performance (held-out months, November–December 2010–2020, 660 pairs)

- **Synthetic dimming.** MAE 0.0070.
- **Recovery of `DEG_COR`.** MAE 0.0094 (1600 Å) and 0.0072 (1700 Å); bias +0.0034 and +0.0008.
- **Seed spread** (`DEG_COR` MAE). Across seeds 1–3: 0.0089–0.0095 (1600 Å) and 0.0072–0.0078 (1700 Å).
- **Calibration** (α sweep, 96 frames from all months). Largest bias inside the official range: 0.005 and 0.003.
- **Month by month** (all months, `results/released_alpha_monthly.csv`). RMS of the monthly residual 0.009 (1600 Å) and 0.006 (1700 Å).

## Limits

- **Coverage.** Not tested after 2020.
- **Not independent of the official curve.** The `DEG_COR` comparison is a recovery test, not an independent check of the official curve. The recipe was chosen with held-out and `DEG_COR` results in view.
- **Single-frame scatter.** Single frames scatter by about 0.015 (1600 Å) and 0.009 (1700 Å) at α = 0.9.

## Load

Run from `final_report/`; the paths are relative to it.

```python
import torch
from sdoml_uv_dataset import Autocalibration6
ck = torch.load("models/autocal_uv_1600_1700_refined_final.pt", map_location="cpu", weights_only=True)
model = Autocalibration6(tuple(ck["input_shape"]), output_dim=2, head=ck["head"], output_scale=ck["output_scale"])
model.load_state_dict(ck["model"]); model.eval()
```

**Fingerprint.** For `torch.rand((1, 2, 256, 256), generator=torch.Generator().manual_seed(0)) * 0.5` the output is about [0.07831, 0.09314]; the tutorial checks this.

**Credit.** Method: Dos Santos et al. 2021, A&A 648, A53, ported from `sdo-autocal_pub` (DOI 10.5281/zenodo.4434743). Data: SDOML-v2 (Galvez et al. 2019). Model: Kai Yang, PhD (SETI Institute).
