import pytest

from cli.main import parse_args, main


class TestParseArgs:
    def test_accepts_required_args(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4"])
        assert args.video == "in.mp4"
        assert args.output == "out.mp4"

    def test_rejects_missing_video(self):
        with pytest.raises(SystemExit):
            parse_args(["--output", "out.mp4"])

    def test_rejects_missing_output(self):
        with pytest.raises(SystemExit):
            parse_args(["--video", "in.mp4"])

    def test_accepts_audio_arg(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--audio", "dub.wav"])
        assert args.audio == "dub.wav"

    def test_audio_defaults_to_none(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4"])
        assert args.audio is None

    def test_debug_tracking_flag_accepted(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--debug-tracking"])
        assert args.debug_tracking is True

    def test_debug_tracking_default_false(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4"])
        assert args.debug_tracking is False


class TestMainPassthrough:
    def test_passthrough_preserves_frame_count(self, tmp_video, tmp_path):
        output = tmp_path / "output.mp4"
        result = main(["--video", str(tmp_video), "--output", str(output)])
        assert result == 0
        assert output.exists()
        from core.video_io import VideoReader
        with VideoReader(output) as reader:
            assert reader.frame_count == 10

    def test_passthrough_preserves_audio(self, tmp_video_with_audio, tmp_path):
        output = tmp_path / "output.mp4"
        result = main(["--video", str(tmp_video_with_audio), "--output", str(output)])
        assert result == 0
        from core.video_io import has_audio_stream
        assert has_audio_stream(output) is True

    def test_returns_1_for_missing_video(self, tmp_path):
        result = main(["--video", str(tmp_path / "nope.mp4"), "--output", str(tmp_path / "out.mp4")])
        assert result == 1
