# SeaHunter-VIS operator guide

SeaHunter-VIS is decision support, not an autonomous rescue or navigation authority. Confirm the selected source,
camera calibration, telemetry quality, zone version, model package, storage capacity, and health status before a
mission. An absolute range or location is valid only when the UI reports accepted calibration, synchronized telemetry,
matching altitude datum, downward sea intersection, and sufficient quality; otherwise use relative trend only.

Treat an alert as a prompt for operator verification. Review class, track history, observed versus inferred points,
range quality, TTC, rule, zone, and evidence. Acknowledge receipt separately from marking confirmed, false positive,
or unsure. False positives enter the hard-sample queue. Never erase evidence to hide a bad alert.

If the service reports reduced ROI, lower resolution, software decode, stale telemetry, high temperature, memory
pressure, or timeout, follow the site escalation procedure. In safe-stop mode, use the approved fallback watch and do
not claim automated coverage. The system continues local persistence during network loss when storage is healthy.
