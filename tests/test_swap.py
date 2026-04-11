from swap import select_providers


def test_select_providers_prefers_coreml_when_available():
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert select_providers(available) == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_select_providers_falls_back_to_cpu_when_coreml_missing():
    available = ["CPUExecutionProvider"]
    assert select_providers(available) == ["CPUExecutionProvider"]


def test_select_providers_always_includes_cpu_as_last_resort():
    available = ["CoreMLExecutionProvider"]
    result = select_providers(available)
    assert result[-1] == "CPUExecutionProvider"
