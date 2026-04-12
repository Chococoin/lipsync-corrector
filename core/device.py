from __future__ import annotations


def get_onnx_providers() -> list[str]:
    """Return ordered list of ONNX Runtime execution providers available on this machine."""
    import onnxruntime
    available = onnxruntime.get_available_providers()
    preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    result = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in result:
        result.append("CPUExecutionProvider")
    return result


def get_torch_device() -> str:
    """Return the best available PyTorch device string: 'mps', 'cuda', or 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
