import pytest
from src.srt_telemetry import parse_srt, altitude_at_timestamp

# Real format, confirmed 2026-08-22 by extracting the embedded subtitle
# stream from data/Intersection_Merged.MP4 via ffmpeg (docs/decisions.md
# Decision 4). Values below are copied from the real extraction's first two
# frames, not fabricated.
SAMPLE_SRT = """1
00:00:00,000 --> 00:00:00,033
FrameCnt: 0 2026-08-21 17:38:52.463
[iso: 130] [shutter: 1/120.0] [fnum: 2.8] [ev: 0] [color_md: default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 18.566227] [longitude: 73.771846] [rel_alt: 70.472 abs_alt: 607.273] [gb_yaw: -125.5 gb_pitch: -63.1 gb_roll: 0.0]

2
00:00:00,033 --> 00:00:00,066
FrameCnt: 1 2026-08-21 17:38:52.497
[iso: 130] [shutter: 1/120.0] [fnum: 2.8] [ev: 0] [color_md: default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 18.566227] [longitude: 73.771846] [rel_alt: 70.473 abs_alt: 607.274] [gb_yaw: -125.5 gb_pitch: -63.1 gb_roll: 0.0]

"""


def test_parse_srt_extracts_frames(tmp_path):
    srt_file = tmp_path / "sample.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    frames = parse_srt(str(srt_file))

    assert len(frames) == 2
    assert frames[0].frame_idx == 0
    assert frames[0].timestamp_ms == 0
    assert frames[0].latitude == pytest.approx(18.566227)
    assert frames[0].longitude == pytest.approx(73.771846)
    assert frames[0].rel_altitude_m == pytest.approx(70.472)
    assert frames[0].abs_altitude_m == pytest.approx(607.273)
    assert frames[1].frame_idx == 1
    assert frames[1].timestamp_ms == 33


def test_parse_srt_raises_on_unparseable_file(tmp_path):
    srt_file = tmp_path / "empty.srt"
    srt_file.write_text("not a valid srt file at all", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_srt(str(srt_file))


def test_altitude_at_timestamp_returns_closest(tmp_path):
    srt_file = tmp_path / "sample.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")
    frames = parse_srt(str(srt_file))

    assert altitude_at_timestamp(frames, timestamp_ms=0) == pytest.approx(70.472)
    assert altitude_at_timestamp(frames, timestamp_ms=33) == pytest.approx(70.473)
    assert altitude_at_timestamp(frames, timestamp_ms=15) in (70.472, 70.473)
