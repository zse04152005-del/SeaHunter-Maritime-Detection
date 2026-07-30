# Third-party notices

This document records third-party components retained during the M0 baseline-freeze stage. It is not legal advice; release owners must perform a formal license review before redistribution.

## Ultralytics

- Component: temporary source copy under `yolo_source/ultralytics/`
- Upstream: <https://github.com/ultralytics/ultralytics>
- Pinned version: `8.3.234`
- Declared license in source: AGPL-3.0
- Project use: loading and reproducing the original SeaHunter detector, custom model parsing, training, tracking, and export
- Local modifications: SPDConv, EMA, MultiSEAM registration, SeaHunter model YAML files, and CIoU/NWD bbox loss blending

The repository's root MIT license applies to original SeaHunter-VIS code and does not override Ultralytics licensing terms. Before closed-source or commercial distribution, choose one of the following and record the decision:

1. obtain an appropriate Ultralytics commercial license;
2. publish and operate the combined work in a manner compliant with AGPL-3.0; or
3. migrate the production detector to a license-compatible framework and independently verify the migrated model.

## SeaDronesSee

- Component: dataset configuration and expected class taxonomy
- Project use: training and evaluation only; dataset files are not included in this repository
- Action required: verify and record the dataset release terms before sharing derived datasets or annotations

## TrackEval

- Component: optional TrackEval Python dependency pinned to commit
  `12c8791b303e0a0b50f753af204249e622d0281a`
- Upstream: <https://github.com/JonathonLuiten/TrackEval>
- Declared license: MIT
- Project use: standard HOTA, DetA, AssA, CLEAR MOT, and identity metric evaluation
- Local modifications: none; SeaHunter-VIS prepares normalized class-agnostic MOTChallenge inputs, restores the
  deprecated NumPy scalar aliases required by this pinned revision at the import boundary, and parses the returned
  standard metric dictionaries

## Model weights

`weights/seahunter_best.pt` was produced by the original SeaHunter project using the modified Ultralytics code. Distribution of weights must be reviewed together with the training framework, source dataset terms, and intended product use.
