import numpy as np
import pytest

from core.video_io import VideoReader


class TestVideoReader:
    def test_opens_valid_video(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            assert reader.fps == pytest.approx(10.0, abs=0.5)
            assert reader.width == 64
            assert reader.height == 64
            assert reader.frame_count == 10

    def test_duration(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            assert reader.duration == pytest.approx(1.0, abs=0.2)

    def test_iterates_all_frames(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            frames = list(reader)
            assert len(frames) == 10
            assert all(f.shape == (64, 64, 3) for f in frames)
            assert all(f.dtype == np.uint8 for f in frames)

    def test_frames_have_different_content(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            frames = list(reader)
            first = frames[0]
            last = frames[-1]
            assert not np.array_equal(first, last)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoReader(tmp_path / "nonexistent.mp4")

    def test_len(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            assert len(reader) == 10

    def test_context_manager_releases(self, tmp_video):
        reader = VideoReader(tmp_video)
        reader.close()
        frames = list(reader)
        assert frames == []
