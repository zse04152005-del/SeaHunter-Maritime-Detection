# SeaHunter maritime dataset card

Status: template; no production dataset is bundled in this public repository.

The target corpus is single-UAV visible-light maritime video covering multiple voyages, locations, times, heights,
weather conditions, and sea states. Raw imagery may contain people and precise locations and therefore requires an
approved retention, access, redaction, and deletion policy. DVC or lakeFS commits identify immutable data versions;
credentials and raw media remain outside Git.

Required release statistics are samples/videos/voyages per split, class and pixel-size distribution, weather and
degradation matrix, hard-negative counts, annotation-review rate, leakage audit, and known geographic/demographic
limitations. The example manifest validates contracts only and is not training or acceptance data.
