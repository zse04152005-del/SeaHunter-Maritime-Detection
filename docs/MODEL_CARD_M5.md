# M5 teacher-student model card

Status: engineering baseline; real-data benefit is not established.

The teacher-student plan uses an EMA teacher, three deterministic seeds, weather-aware confidence gates, teacher /
student class agreement, temporal IoU, and minimum tracklet length. It logs Git commit, data version, configuration
hash, seed, metrics, and artifacts to MLflow. It must be compared with the fully supervised baseline on a frozen,
leakage-free test set by weather and pixel-size bucket.

Promotion requires a reproducible gain with no material increase in false alarms/hour. Do not deploy a model based
on pseudo-label acceptance rate or synthetic tests alone. Known risks include confirmation bias, missed tiny targets,
weather threshold overfitting, geographic shift, and hard-negative under-coverage.
