import numpy as np
import pytest

from core.video_io import VideoReader, VideoWriter


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


class TestVideoWriter:
    def test_writes_frames(self, tmp_path):
        output = tmp_path / "output.mp4"
        with VideoWriter(output, fps=10.0, width=64, height=64) as writer:
            for i in range(5):
                frame = np.full((64, 64, 3), (i * 50, 0, 0), dtype=np.uint8)
                writer.write(frame)
            assert writer.frames_written == 5
        assert output.exists()
        assert output.stat().st_size > 0

    def test_written_video_is_readable(self, tmp_path):
        output = tmp_path / "roundtrip.mp4"
        original_frames = []
        for i in range(5):
            original_frames.append(np.full((64, 64, 3), (i * 50, 0, 0), dtype=np.uint8))

        with VideoWriter(output, fps=10.0, width=64, height=64) as writer:
            for f in original_frames:
                writer.write(f)

        with VideoReader(output) as reader:
            read_frames = list(reader)
            assert len(read_frames) == 5

    def test_raises_on_invalid_path(self, tmp_path):
        bad_path = tmp_path / "nonexistent_dir" / "output.mp4"
        with pytest.raises(RuntimeError):
            VideoWriter(bad_path, fps=10.0, width=64, height=64)
