# V1 release checklist

- [ ] Frozen real maritime test set and M3 truth set accepted.
- [ ] Weather/location/time/altitude/sea-state sea-trial matrix accepted.
- [ ] Operator blind tests and issue retests accepted.
- [ ] Static/dynamic ONNX and target TensorRT consistency accepted.
- [ ] NVDEC, CUDA streams, preallocation, FPS/P95/power/thermal/VRAM accepted on the target device.
- [ ] Live fault matrix, 8-hour performance, and 72-hour stability accepted.
- [ ] Safety, privacy, license, and operations reviews independently approved.
- [ ] Signed model package, engine, edge image, documentation, and sea-trial hashes recorded.
- [ ] Canary upgrade and automatic rollback drill accepted.
- [ ] `freeze_release_manifest` produces the reviewed immutable V1 manifest.

No unchecked item may be waived by changing this file alone; update the governing approval evidence and machine gate.
