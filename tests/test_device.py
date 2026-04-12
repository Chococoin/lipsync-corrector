from core.device import get_onnx_providers, get_torch_device


def test_get_onnx_providers_returns_list():
    providers = get_onnx_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1


def test_get_onnx_providers_always_ends_with_cpu():
    providers = get_onnx_providers()
    assert providers[-1] == "CPUExecutionProvider"


def test_get_torch_device_returns_known_string():
    device = get_torch_device()
    assert device in ("mps", "cuda", "cpu")
