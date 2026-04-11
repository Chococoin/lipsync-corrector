import pytest

from swap import parse_args, select_providers


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


def test_parse_args_accepts_required_arguments():
    args = parse_args([
        "--face", "a.jpg",
        "--video", "b.mp4",
        "--output", "c.mp4",
    ])
    assert args.face == "a.jpg"
    assert args.video == "b.mp4"
    assert args.output == "c.mp4"
    assert args.max_seconds is None


def test_parse_args_accepts_max_seconds():
    args = parse_args([
        "--face", "a.jpg",
        "--video", "b.mp4",
        "--output", "c.mp4",
        "--max-seconds", "15",
    ])
    assert args.max_seconds == 15


def test_parse_args_rejects_missing_required():
    with pytest.raises(SystemExit):
        parse_args(["--face", "a.jpg"])
