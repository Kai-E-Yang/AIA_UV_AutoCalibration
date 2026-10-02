"""
sdoml_uv_dataset.py
===================

Dataset + model for AIA auto-calibration on the SDOML-v2 1600 A / 1700 A cubes
produced by ``run/01_download_sdomlv2_uv.ipynb``.

This module is a *faithful port* of the pipeline used in

    Dos Santos et al. 2021, A&A 648, A53
    "Multi-Channel Auto-Calibration for the Atmospheric Imaging Assembly
     using Machine Learning"   (arXiv:2012.14023)

as implemented in ``sdo-autocal_pub-master`` and pinned by
``config/autocal_paper_config.yaml``.  Every deviation from that reference is
called out in a ``# PAPER:`` comment next to the code.

Reference chain
---------------
* ``src/sdo/datasets/sdo_dataset.py``       -> loading, subsample, scale, apodize
* ``src/sdo/datasets/dimmed_sdo_dataset.py``-> synthetic dimming, test flip
* ``src/sdo/models/autocalibration_models.py`` -> ``Autocalibration6``
* ``src/sdo/pipelines/autocalibration_pipeline.py`` -> MSE loss, tolerance metric
* ``src/sdo/io.py``                          -> ``sdo_scale`` / ``AUNIT_BYCH``

Changelog 2026-09-02 (debugging the NaN at epoch 4/5)
-----------------------------------------------------
* Scale constants: the paper's ``AUNIT_BYCH`` are in SDOML **v1** units.  v1
  makes its 512x512 frame by *summing* 2x2 blocks of the 1024x1024 synoptic
  image (Galvez et al. 2019, Sec. 3.1 step 4); SDOML-v2 makes it by *averaging*
  the same blocks (``SDOMLv2/aia_fits_to_zarr.py``: ``downscale_local_mean``).
  Hence  v1 = 4 x v2  and the paper's constants in v2 units are
  ``PAPER_AUNIT_BYCH_V2 = AUNIT_BYCH / 4``.  Using the raw frame *mean* instead
  (35 / 434, what ``scaling_constants.json`` holds) feeds the network inputs
  ~4x larger than the paper did and, with Adam at lr 1e-3 on a 93 312-wide
  fully-connected layer, that is enough to push a sigmoid output to logit -60
  on the first steps, after which that channel is dead for good.  See
  ``DEBUG_REPORT_02.md``.
* ``scale_map`` is no longer read silently from ``scaling_constants.json``;
  the caller passes it explicitly (default: ``PAPER_AUNIT_BYCH_V2``).
* Apodization radius default is the reference's literal 200 px at 512
  (``PAPER_APODIZE_RADIUS_PX_512``).  The physical limb in the v2 frames is
  976"/4.8" = 203.3 px (measured: 203.5-204.5 px) and is available as
  ``R_SUN_PX_512`` -- one config line to switch.
* ``__getitem__`` raises if a frame is non-finite, so a corrupt sample can
  never turn into a NaN weight update silently.
* ``discover_years`` / ``scan_cubes`` helpers so the training notebook does not
  depend on a stale ``manifest.json``.

Changelog 2026-09-29 (refined training, 02b)
--------------------------------------------
* ``Autocalibration6(..., head="sigmoid", output_scale=1.0)``: optional output head.
  ``head="linear"`` returns the FC output itself (no sigmoid) -- a deliberate
  deviation from the reference used by ``02b_train_autocal_uv_refined.ipynb``;
  ``output_scale`` multiplies the output (``output_scale * sigmoid(z)`` for the
  sigmoid head).  The defaults reproduce the reference exactly (same parameters,
  same state_dict, bit-identical outputs).  Why: ``ANALYSIS_1700_BIAS.md``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# PAPER: sdo.io.AUNIT_BYCH -- "approximate mean of the channels across the full
# time period", read off Fig. 3 of Galvez et al. 2019 for SDOML *v1*.
PAPER_AUNIT_BYCH: Dict[str, float] = {
    "1600": 500.0,
    "1700": 7000.0,
    "0094": 10.0,
    "0131": 80.0,
    "0171": 2000.0,
    "0193": 3000.0,
    "0211": 1000.0,
    "0304": 500.0,
    "0335": 80.0,
}

# The same constants expressed in SDOML-v2 units.
#   v1 512x512 = SUM  of 2x2 blocks of the 1024 synoptic frame  (Galvez 2019 Sec 3.1, step 4)
#   v2 512x512 = MEAN of 2x2 blocks of the 1024 synoptic frame  (aia_fits_to_zarr.py)
#   =>  value_v1 = 4 * value_v2
V1_TO_V2_UNIT_FACTOR = 4.0
PAPER_AUNIT_BYCH_V2: Dict[str, float] = {
    k: v / V1_TO_V2_UNIT_FACTOR for k, v in PAPER_AUNIT_BYCH.items()
}

# Solar radius in pixels on the native SDOML-v2 512x512 grid.
#   aia_fits_to_zarr.py rescales every image so the Sun subtends trgtAS = 976"
#   on a grid of 4.8"/px  ->  976 / 4.8 = 203.33 px.  Measured on the frames on
#   disk: half-intensity limb at 203.5-204.5 px.
R_SUN_PX_512 = 976.0 / 4.8  # = 203.333...

# PAPER: sdo_dataset.py masks  dist < 200 / subsample  ("Radius of sun at 1 AU
# is 200*4.8 arcsec").  That is 0.984 R_sun on SDOML data (v1 and v2 both fix
# the Sun at 976").  We follow the reference literally by default.
PAPER_APODIZE_RADIUS_PX_512 = 200.0


# --------------------------------------------------------------------------
# Channel-name helpers
# --------------------------------------------------------------------------

def canonical_channel(ch) -> str:
    """'1600A' | '1600' | 1600  ->  '1600'   (zero-padded to 4 for EUV)."""
    s = str(ch).strip().upper()
    if s.endswith("A"):
        s = s[:-1]
    if s.isdigit() and len(s) < 4:
        s = s.zfill(4)
    return s


def zarr_channel(ch) -> str:
    """'1600' -> '1600A'  (the key used in the SDOML-v2 zarr store / filenames)."""
    s = str(ch).strip().upper()
    if not s.endswith("A"):
        s = s + "A"
    return s.lstrip("0") if s[0] == "0" else s


# --------------------------------------------------------------------------
# Train / test / holdout month split  (verbatim from the reference)
# --------------------------------------------------------------------------

def find_months(test: bool = False,
                test_ratio: float = 0.3,
                mnt_step: int = 1,
                holdout: bool = False) -> List[int]:
    """
    PAPER: exact port of ``SDO_Dataset.find_months``.

    November and December are *always* held out.  Of the remaining months
    1..10, the first (1 - test_ratio) fraction are training months and the
    last test_ratio fraction are test months.

    With the paper settings (mnt_step=1, test_ratio=0.3):
        train   -> [1, 2, 3, 4, 5, 6, 7]
        test    -> [8, 9, 10]
        holdout -> [11, 12]

    Note this splits *within* every year, so train and test months are never
    adjacent in time except across the 7/8 boundary.  That is the whole point:
    Galvez et al. 2019 Sec. 4 warns that random splits of SDO data are
    "overly optimistic" because consecutive frames are nearly identical.
    """
    if holdout:
        return [11, 12]
    months = np.arange(1, 11, mnt_step)
    if test:
        n = int(len(months) * test_ratio)
        return months[-n:].tolist()
    n = int(len(months) * (1 - test_ratio))
    return months[:n].tolist()


# --------------------------------------------------------------------------
# Data discovery / integrity helpers
# --------------------------------------------------------------------------

def _cube_path(data_root: str, prefix: str, ch: str, year: int, size_tag: str) -> str:
    base = f"{prefix}_{ch}_{year}_{size_tag}.npy"
    cand = os.path.join(data_root, str(year), base)
    return cand if os.path.exists(cand) else os.path.join(data_root, base)


def discover_years(data_root: str,
                   channels: Sequence[str] = ("1600A", "1700A"),
                   size_tag: str = "512x512") -> List[int]:
    """
    Years for which *every* requested channel has an image cube on disk.
    Do not trust ``manifest.json`` for this: it is rewritten by every run of
    the download notebook and only lists the years of that run.
    """
    chans = [zarr_channel(c) for c in channels]
    years = []
    for name in sorted(os.listdir(data_root)):
        if not re.fullmatch(r"\d{4}", name):
            continue
        y = int(name)
        if all(os.path.exists(_cube_path(data_root, "sub_image", c, y, size_tag)) for c in chans):
            years.append(y)
    return years


def scan_cubes(data_root: str,
               years: Sequence[int],
               channels: Sequence[str] = ("1600A", "1700A"),
               size_tag: str = "512x512",
               chunk: int = 64) -> "pd.DataFrame":
    """
    Full pass over every frame of every requested cube.  Returns one row per
    (year, channel) with counts of non-finite / negative pixels and value
    statistics.  Cheap (memory-mapped, chunked) and worth running once before
    training: a single NaN pixel in a single frame is enough to poison the
    weights on the first epoch that samples it.
    """
    import pandas as pd
    rows = []
    for y in years:
        for c in [zarr_channel(ch) for ch in channels]:
            p = _cube_path(data_root, "sub_image", c, y, size_tag)
            a = np.load(p, mmap_mode="r")
            n = a.shape[0]
            n_nan = n_inf = n_neg = 0
            frames_bad = 0
            fmax = np.empty(n); fmean = np.empty(n)
            for s in range(0, n, chunk):
                blk = np.asarray(a[s:s + chunk], dtype=np.float32)
                nan = np.isnan(blk); inf = np.isinf(blk)
                n_nan += int(nan.sum()); n_inf += int(inf.sum())
                n_neg += int((blk < 0).sum())
                frames_bad += int((nan | inf).any(axis=(1, 2)).sum())
                fmax[s:s + chunk] = np.nanmax(np.where(inf, np.nan, blk), axis=(1, 2))
                fmean[s:s + chunk] = np.nanmean(np.where(inf, np.nan, blk), axis=(1, 2))
            deg = np.load(_cube_path(data_root, "sub_degcor", c, y, size_tag))
            rows.append(dict(year=y, channel=c, n_frames=n, shape=str(a.shape), dtype=str(a.dtype),
                             frames_nonfinite=frames_bad, px_nan=n_nan, px_inf=n_inf, px_negative=n_neg,
                             frame_mean_med=float(np.nanmedian(fmean)),
                             frame_mean_min=float(np.nanmin(fmean)), frame_mean_max=float(np.nanmax(fmean)),
                             frame_max_med=float(np.nanmedian(fmax)), frame_max_max=float(np.nanmax(fmax)),
                             degcor_min=float(deg.min()), degcor_max=float(deg.max())))
    return pd.DataFrame(rows)


def measured_scale_map(data_root: str,
                       years: Sequence[int],
                       channels: Sequence[str] = ("1600A", "1700A"),
                       size_tag: str = "512x512",
                       max_frames_per_year: int = 120) -> Dict[str, float]:
    """
    'Median over time of the full-frame mean' for each channel, over *all* the
    given years -- the definition 01_download uses for scaling_constants.json,
    but not restricted to the last downloaded year.  Informational; the
    default training scale is PAPER_AUNIT_BYCH_V2.
    """
    out = {}
    for ch in channels:
        c = zarr_channel(ch)
        vals = []
        for y in years:
            a = np.load(_cube_path(data_root, "sub_image", c, y, size_tag), mmap_mode="r")
            step = max(1, a.shape[0] // max_frames_per_year)
            vals.append(np.asarray(a[::step], dtype=np.float32).mean(axis=(1, 2)))
        out[canonical_channel(c)] = float(np.median(np.concatenate(vals)))
    return out


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

@dataclass
class _YearBlock:
    year: int
    images: Dict[str, np.ndarray]          # channel -> memmap [N, H, W]
    months: np.ndarray                     # [N] int
    times: np.ndarray                      # [N] datetime64[s]
    degcor: Dict[str, np.ndarray]          # channel -> [N] float
    exptime: Dict[str, np.ndarray]         # channel -> [N] float
    keep: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


class SdomlUVDataset(Dataset):
    """
    Returns ``(dimmed, alpha, clean)`` exactly like ``DimmedSDO_Dataset``.

    Per-item pipeline, in the reference's order:

        1. read native 512x512 frame for each channel
        2. subsample by slicing  ``img[::s, ::s]``          # PAPER: sdo_dataset
        3. divide by the per-channel scale constant          # PAPER: sdo_scale
        4. optional per-image min-max normalisation (off)    # PAPER: normalization=0
        5. apodize -- zero everything outside the solar limb  # PAPER: apodize=true
        6. optional threshold_black                          # PAPER: false
        7. optional horizontal flip (test set only)          # PAPER: true for test
        8. alpha ~ U(min_alpha, max_alpha) per channel; dimmed = clean * alpha

    Parameters
    ----------
    data_root : str
        ``run/data`` -- the directory written by 01_download_sdomlv2_uv.ipynb.
    channels : sequence
        e.g. ``["1600A", "1700A"]``.  Order is preserved and defines the
        network's channel order.
    years : sequence of int
    months : sequence of int
        Use :func:`find_months` to get the paper's split.
    subsample : int
        512 // subsample = network input size.  PAPER: 2 -> 256x256.
    scale_map : dict
        channel -> divisor.  Default ``PAPER_AUNIT_BYCH_V2`` (the paper's
        constants in v2 units).  Never read from disk implicitly.
    apodize_radius_px_512 : float
        Mask radius at 512x512.  Default: the reference's 200 px.
    """

    def __init__(
        self,
        data_root: str,
        channels: Sequence[str] = ("1600A", "1700A"),
        years: Sequence[int] = (2010,),
        months: Optional[Sequence[int]] = None,
        subsample: int = 2,
        scaling: bool = True,
        scale_map: Optional[Dict[str, float]] = None,
        normalization: int = 0,
        apodize: bool = True,
        apodize_radius_px_512: float = PAPER_APODIZE_RADIUS_PX_512,
        min_alpha: float = 0.01,
        max_alpha: float = 1.0,
        threshold_black: bool = False,
        threshold_black_value: float = 0.09,
        flip_images: bool = False,
        synthetic_dimming: bool = True,
        use_real_degradation: bool = False,
        size_tag: str = "512x512",
        return_metadata: bool = False,
        seed: Optional[int] = None,
        check_finite: bool = True,
    ):
        super().__init__()

        self.data_root = data_root
        self.channels = [zarr_channel(c) for c in channels]
        self.years = [int(y) for y in years]
        self.months = None if months is None else sorted(int(m) for m in months)
        self.subsample = int(subsample)
        self.scaling = scaling
        self.normalization = int(normalization)
        self.apodize = apodize
        self.apodize_radius_px_512 = float(apodize_radius_px_512)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.threshold_black = threshold_black
        self.threshold_black_value = float(threshold_black_value)
        self.flip_images = flip_images
        self.synthetic_dimming = synthetic_dimming
        self.use_real_degradation = use_real_degradation
        self.size_tag = size_tag
        self.return_metadata = return_metadata
        self.check_finite = check_finite

        if not (0.0 < self.min_alpha <= self.max_alpha):
            raise ValueError("require 0 < min_alpha <= max_alpha")
        if use_real_degradation and synthetic_dimming:
            raise ValueError(
                "use_real_degradation=True is for evaluation only; set "
                "synthetic_dimming=False so alpha comes from DEG_COR."
            )

        self.scale_map = ({canonical_channel(k): float(v) for k, v in scale_map.items()}
                          if scale_map is not None else dict(PAPER_AUNIT_BYCH_V2))
        if self.scaling:
            missing = [c for c in self.channels if canonical_channel(c) not in self.scale_map]
            if missing:
                raise KeyError(f"scale_map has no entry for {missing}")

        self._rng = np.random.default_rng(seed) if seed is not None else None

        self.blocks: List[_YearBlock] = []
        self.index_map: List[Tuple[int, int]] = []
        self._mask: Optional[np.ndarray] = None
        self._load()

    # -- loading ----------------------------------------------------------

    def _path(self, prefix: str, ch: str, year: int) -> str:
        return _cube_path(self.data_root, prefix, ch, year, self.size_tag)

    def _load(self) -> None:
        for year in self.years:
            images, degcor, exptime = {}, {}, {}
            lengths = []
            for ch in self.channels:
                img_p = self._path("sub_image", ch, year)
                if not os.path.exists(img_p):
                    raise FileNotFoundError(
                        f"{img_p} not found -- run 01_download_sdomlv2_uv.ipynb "
                        f"for year {year} first."
                    )
                images[ch] = np.load(img_p, mmap_mode="r")
                if images[ch].ndim != 3:
                    raise ValueError(f"{img_p} must be [N, H, W], got {images[ch].shape}")
                if images[ch].shape[1] != images[ch].shape[2]:
                    raise ValueError(f"{img_p}: frames must be square, got {images[ch].shape}")
                lengths.append(images[ch].shape[0])

                d_p = self._path("sub_degcor", ch, year)
                degcor[ch] = (np.load(d_p).astype(np.float64)
                              if os.path.exists(d_p)
                              else np.ones(lengths[-1], dtype=np.float64))
                e_p = self._path("sub_expt", ch, year)
                exptime[ch] = (np.load(e_p).astype(np.float64)
                               if os.path.exists(e_p)
                               else np.full(lengths[-1], np.nan))

            if len(set(lengths)) != 1:
                raise ValueError(
                    f"channel cubes disagree in length for {year}: "
                    f"{dict(zip(self.channels, lengths))}. The download notebook "
                    f"writes co-registered cubes, so this means a partial write."
                )
            n = lengths[0]

            t_p = self._path("sub_tobs", self.channels[0], year)
            times = np.asarray(np.load(t_p, allow_pickle=True)).reshape(n, -1)[:, 0]
            times = np.array([_to_datetime64(v) for v in times], dtype="datetime64[s]")
            months = times.astype("datetime64[M]").astype(int) % 12 + 1

            keep = np.arange(n)
            if self.months is not None:
                keep = keep[np.isin(months[keep], self.months)]
            keep = keep[np.argsort(times[keep], kind="stable")]

            blk = _YearBlock(year=year, images=images, months=months, times=times,
                             degcor=degcor, exptime=exptime, keep=keep)
            self.blocks.append(blk)
            b = len(self.blocks) - 1
            self.index_map.extend((b, int(i)) for i in keep)

        if not self.index_map:
            raise RuntimeError(
                f"No samples for years={self.years} months={self.months}."
            )

        native = self.blocks[0].images[self.channels[0]].shape[1]
        self.native_size = int(native)
        self.size = self.native_size // self.subsample
        if self.apodize:
            # PAPER: mask radius is (200 / subsample) on the *subsampled* grid
            self._mask = _disk_mask(self.size,
                                    self.apodize_radius_px_512 * (self.native_size / 512.0) / self.subsample)

    # -- torch API --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index_map)

    def timestamps(self) -> np.ndarray:
        return np.array([self.blocks[b].times[i] for b, i in self.index_map],
                        dtype="datetime64[s]")

    def years_of_samples(self) -> np.ndarray:
        return np.array([self.blocks[b].year for b, _ in self.index_map], dtype=int)

    def get_metadata(self, idx: int) -> dict:
        b, i = self.index_map[idx]
        blk = self.blocks[b]
        return {
            "year": blk.year,
            "local_index": i,
            "t_obs": str(blk.times[i]),
            "month": int(blk.months[i]),
            "deg_cor": {c: float(blk.degcor[c][i]) for c in self.channels},
            "exptime": {c: float(blk.exptime[c][i]) for c in self.channels},
        }

    def __getitem__(self, idx: int):
        b, i = self.index_map[idx]
        blk = self.blocks[b]
        s = self.subsample

        # 1-2. read + subsample by slicing.  PAPER: temp[::s, ::s]
        img = np.stack(
            [np.asarray(blk.images[c][i][::s, ::s], dtype=np.float32)
             for c in self.channels],
            axis=0,
        )

        if self.check_finite and not np.isfinite(img).all():
            raise ValueError(
                f"non-finite pixels in year {blk.year} frame {i} ({blk.times[i]}) "
                f"-- refusing to feed it to the network"
            )

        # 3. per-channel scaling.  PAPER: sdo_scale -> img / AUNIT_BYCH[ch]
        if self.scaling:
            for k, c in enumerate(self.channels):
                img[k] /= float(self.scale_map[canonical_channel(c)])

        # 4. per-image normalisation.  PAPER: normalization = 0 (off)
        if self.normalization == 1:
            for k in range(img.shape[0]):
                lo, hi = float(img[k].min()), float(img[k].max())
                img[k] = (img[k] - lo) / (hi - lo) if hi > lo else 0.0

        # 5. apodize -- mask everything beyond the solar limb
        if self._mask is not None:
            img *= self._mask[None, :, :]

        clean = torch.from_numpy(img)

        # 6. threshold_black.  PAPER: applied AFTER scaling (value 0.09 is in
        #    scaled units), and disabled in autocal_paper_config.yaml
        if self.threshold_black:
            clean[clean <= self.threshold_black_value] = 0.0

        # 7. flip.  PAPER: flip_test_images=true -- the test set is mirrored so
        #    stray/dead pixels cannot be memorised across the split.
        if self.flip_images:
            clean = torch.flip(clean, [2])

        # 8. dimming
        if self.synthetic_dimming:
            alpha = self._sample_alpha()
        elif self.use_real_degradation:
            # Evaluation mode: SDOMLv2 stores raw/(exptime*DEG_COR), i.e. already
            # corrected.  Multiplying by DEG_COR reproduces the *as-observed*
            # degraded image, and the ground-truth alpha is DEG_COR itself.
            alpha = torch.tensor(
                [float(blk.degcor[c][i]) for c in self.channels],
                dtype=torch.float32,
            )
        else:
            alpha = torch.ones(len(self.channels), dtype=torch.float32)

        dimmed = clean * alpha[:, None, None]

        if self.return_metadata:
            return dimmed, alpha, clean, self.get_metadata(idx)
        return dimmed, alpha, clean

    # -- alpha ------------------------------------------------------------

    def _sample_alpha(self) -> torch.Tensor:
        """
        PAPER: ``DimmedSDO_Dataset`` draws ``max_alpha * rand(C)`` and *rejects
        the whole vector* while any component < min_alpha.  Conditioning a
        vector of i.i.d. U(0, max) on "all components >= min" leaves the
        components i.i.d. U(min, max), so the direct draw below is
        distributionally identical -- and does not loop.
        """
        c = len(self.channels)
        if self._rng is not None:
            u = torch.from_numpy(self._rng.random(c).astype(np.float32))
        else:
            u = torch.rand(c)
        return self.min_alpha + (self.max_alpha - self.min_alpha) * u


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _to_datetime64(v) -> np.datetime64:
    if isinstance(v, np.datetime64):
        return v.astype("datetime64[s]")
    s = v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
    s = s.strip()
    if s.endswith("_TAI"):                        # HMI style 2010.05.01_00:12:04_TAI
        s = s[:-4]
    s = s.replace("Z", "")
    if "_" in s and "T" not in s:                 # 2010.05.01_00:12:04
        d, t = s.split("_", 1)
        s = d.replace(".", "-") + "T" + t
    return np.datetime64(s, "s")


def _disk_mask(size: int, radius_px: float) -> np.ndarray:
    """
    PAPER: sdo_dataset.py builds the same grid --
        x = arange(N) - N/2 + 0.5,  keep where sqrt(x^2 + y^2) < radius
    """
    x = np.arange(size, dtype=np.float32) - size / 2.0 + 0.5
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return (np.sqrt(xx * xx + yy * yy) < radius_px).astype(np.float32)


# --------------------------------------------------------------------------
# Model -- PAPER: model-version 6  ==  Autocalibration6
# --------------------------------------------------------------------------

class Autocalibration6(nn.Module):
    """
    Verbatim port of ``sdo.models.autocalibration_models.Autocalibration6``,
    the architecture selected by ``autocal_paper_config.yaml``
    (``model-version: 6``).

        Conv(C->64, k3) - ReLU - MaxPool(3)
        Conv(64->128, k3) - ReLU - MaxPool(3)
        flatten - Linear(-> C) - Sigmoid

    Note there is exactly ONE fully-connected layer.  ``Autocalibration1``
    (two FC layers, and no ReLU after the second conv) is a *different*,
    earlier model that the paper did not use.

    ``head`` / ``output_scale`` (not in the reference; defaults = reference):
        "sigmoid"  alpha_hat = output_scale * sigmoid(FC)   (reference: scale 1)
        "linear"   alpha_hat = output_scale * FC            (02b; see ANALYSIS_1700_BIAS.md)
    Neither option adds parameters, so every checkpoint loads into either head.
    """

    HEADS = ("sigmoid", "linear")

    def __init__(self, input_shape, output_dim: int, head: str = "sigmoid",
                 output_scale: float = 1.0):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (C, H, W)")
        if head not in self.HEADS:
            raise ValueError(f"head must be one of {self.HEADS}, got {head!r}")
        self.head = head
        self.output_scale = float(output_scale)
        c = int(input_shape[0])
        self._conv2d1 = nn.Conv2d(c, 64, kernel_size=3)
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)
        self._conv2d2 = nn.Conv2d(64, 128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)
        with torch.no_grad():
            self._cnn_output_dim = self._cnn(
                torch.zeros(*input_shape).unsqueeze(0)
            ).nelement()
        self._fc = nn.Linear(self._cnn_output_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)
        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)
        return x

    def logits(self, x):
        """Output of the FC layer before the head (pre-sigmoid for the sigmoid head)."""
        n = x.shape[0]
        return self._fc(self._cnn(x).view(n, -1))

    def activate(self, z):
        """The output head applied to FC outputs ``z``."""
        out = torch.sigmoid(z) if self.head == "sigmoid" else z
        return out if self.output_scale == 1.0 else self.output_scale * out

    def forward(self, x):
        return self.activate(self.logits(x))


def forward_chunked(model: nn.Module, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """
    Evaluate ``model`` on ``x`` in slices of ``chunk`` samples and concatenate.

    Mathematically identical to ``model(x)`` (the network has no batch-dependent
    layers), but keeps every intermediate tensor small.  With the paper's
    ``batch-size-test: 256`` at 256x256 the first conv activation alone is
    256 x 64 x 254 x 254 x 4 B = 4.2 GB, which is above the 4 GB MPSNDArray
    limit of Apple's Metal backend (pytorch/pytorch#149325) -- the source of
    the inf/garbage test loss seen at epoch 1.
    """
    outs = [model(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)]
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------
# Metric -- PAPER: AutocalibrationPipeline.calculate_primary_metric
# --------------------------------------------------------------------------

def binary_success(pred: torch.Tensor,
                   target: torch.Tensor,
                   tolerance: float = 0.05) -> float:
    """Fraction of (sample, channel) entries with |pred - target| < tolerance."""
    return float((torch.abs(pred - target) < tolerance).float().mean())


# --------------------------------------------------------------------------
# Device self-test
# --------------------------------------------------------------------------

def device_self_test(model: nn.Module, x: torch.Tensor, device, chunk: int = 32,
                     atol: float = 1e-3) -> dict:
    """
    Guard against silent numerical corruption on non-CUDA accelerators.

    1. Round-trip a random tensor through ``device`` and compare bit-for-bit.
    2. Run the model on ``x`` on CPU and on ``device`` (chunked) and compare.

    Returns a dict with the discrepancies; raises RuntimeError if either check
    fails.  Known failure modes this catches on Apple MPS:
      * ``.to("mps", non_blocking=True)`` returning garbage (pytorch/pytorch#139550)
      * ops on tensors > 4 GB / 2**31 elements returning wrong results (#149325)
    """
    dev = torch.device(device)
    r = torch.randn(3, 5, 257, 131)
    back = r.to(dev).cpu()
    if not torch.equal(r, back):
        raise RuntimeError(f"tensor round-trip through {dev} is not exact")

    model_cpu = model.to("cpu").eval()
    with torch.no_grad():
        ref = model_cpu(x)
        model_dev = model.to(dev).eval()
        got = forward_chunked(model_dev, x.to(dev), chunk).cpu()
    model.to(dev)
    if not torch.isfinite(got).all():
        raise RuntimeError(f"model forward on {dev} produced non-finite values")
    diff = float((ref - got).abs().max())
    if diff > atol:
        raise RuntimeError(
            f"model forward on {dev} differs from CPU by {diff:.3e} (> {atol}). "
            f"Do not train on this device; set device='cpu'."
        )
    return {"device": str(dev), "max_abs_diff_vs_cpu": diff, "n": int(x.shape[0])}
