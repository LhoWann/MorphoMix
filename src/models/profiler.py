"""Model complexity profiling with torchinfo."""
from typing import Dict, Any, Tuple
import torch.nn as nn
from torchinfo import summary


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> Dict[str, Any]:
    model.eval()
    s = summary(model, input_size=input_shape, verbose=0)

    total_params = s.total_params
    total_macs = s.total_mult_adds
    gflops = (total_macs * 2) / 1e9
    gmacs = total_macs / 1e9
    size_mb = s.to_megabytes(s.total_param_bytes)

    return {
        "params": f"{total_params:,}",
        "params_raw": total_params,
        "macs": f"{gmacs:.3f} GMACs",
        "flops": f"{gflops:.3f} GFLOPs",
        "size_mb": f"{size_mb:.2f} MB",
        "summary": str(s)
    }


def benchmark_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    warmup: int = 20,
    iterations: int = 100
) -> Dict[str, Any]:
    import time
    import torch
    import numpy as np

    model.eval()
    dev = torch.device(device)
    model.to(dev)
    dummy_input = torch.randn(*input_shape, device=dev)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "device": device.upper(),
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "fps": fps,
        "latency_str": f"{mean_ms:.2f} +/- {std_ms:.2f} ms",
        "fps_str": f"{fps:.1f} FPS"
    }
