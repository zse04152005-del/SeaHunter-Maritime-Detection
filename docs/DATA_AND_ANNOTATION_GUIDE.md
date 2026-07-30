# Maritime data and annotation guide

## Identity and split policy

Every frame record must carry `sample_id`, `video_id`, `voyage_id`, UTC capture time, location, weather, degradation
source/intensity, image dimensions, and split. Split complete videos and voyages before extracting frames. A video,
voyage, or same date/location group must never cross train, validation, and frozen test partitions. Near-duplicate
clips, synthetic derivatives, and relabelled exports inherit the source group.

## Detection and track annotation

- Classes are `swimmer`, `boat`, `jetski`, `life_saving_appliances`, and `buoy`; uncertain categories go to review.
- Boxes enclose visible pixels tightly. Do not hallucinate the full extent of an occluded target.
- `track_id` is unique within one video and remains stable across short occlusion. A confirmed scene exit and later
  unrelated entry receives a new ID.
- Occlusion is `none`, `partial` (<50% hidden), `heavy` (>=50% hidden), or `full` for an auditable interpolation gap.
- Event labels reference the configured zone/rule version and distinguish true alert, false positive, and unsure.

## Weather, sea state, and degradation

Use `normal`, `low_light`, `fog`, `glare`, and `high_wave`. Intensity is 0 (none), 1 (mild), 2 (material), or 3
(severe). `degradation_source` is always `real` or `synthetic`; synthetic variants retain their real source ID.
Do not infer weather from model errors. When conditions change, segment the video and record the boundary.

## Hard negatives and quality control

Empty frames may carry `wave`, `wake`, `bird`, `foam`, or `reflection`. Review all small boxes, class changes,
track splits/merges, full occlusions, event disagreements, and at least a stratified 10% of remaining annotations.
Run `seahunter-data-audit ... --require-gates` before training and attach the report to the dataset version.
