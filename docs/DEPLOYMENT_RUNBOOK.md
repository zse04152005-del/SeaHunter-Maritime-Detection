# Edge deployment, upgrade, rollback, and fault runbook

## Build and package

1. Export both frozen static shape and approved dynamic batch ONNX graphs and run the deterministic PyTorch/ONNX
   consistency check. Record shapes, opset, Git commit, data version, and errors.
2. On the target NVIDIA build host, use the reviewed `TensorRTBuildSpec` arguments to build FP16. INT8 is permitted
   only when an immutable, real-only calibration set covers every weather condition at the frozen minimum count.
3. Re-run output consistency and real replay metrics. A successful engine build alone is not acceptance.
4. Assemble the ONNX/engine, labels, preprocessing configuration, metrics, calibration reference, and complete model
   metadata into a signed package. HMAC is available for isolated site testing; production should inject an
   asymmetric HSM/KMS signer through the `PackageSigner` interface.

## Deploy and roll back

Stage a verified package without changing the stable slot. Start a bounded canary, poll the local health endpoint,
and promote only after the configured consecutive healthy samples. Any package verification, startup, output,
latency, memory, temperature, or event-path failure reports unhealthy and retains the previous stable version.
Release state is atomically persisted so a process restart resumes the same decision.

## Runtime degradation

The sequence is normal, reduced ROI, lower resolution, then safe stop for critical temperature or repeated timeout.
Healthy recovery uses hysteresis. Local event persistence remains enabled during network loss. Never silently label
software decode as NVDEC or CPU/ONNX fallback as TensorRT.

## Acceptance evidence

Capture device model/serial, JetPack/driver/CUDA/TensorRT/DeepStream versions, engine and package hashes, calibration
version, stream properties, FPS/P50/P95/P99, power, temperature, VRAM, memory slope, dropped frames, reconnects,
timeouts, and fault evidence. Run 8 hours for performance and 72 hours for stability. Exercise network loss, stream
loss, corrupt frames, clock drift, and process restart. Unit-test time compression is not a soak-test substitute.
