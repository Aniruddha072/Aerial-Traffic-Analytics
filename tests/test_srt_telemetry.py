import pytest
from src.srt_telemetry import parse_srt, altitude_at_timestamp

SAMPLE_SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="36">SrtCnt : 1, DiffTime : 33ms
2026-08-22 10:00:00.000
[iso: 100] [shutter: 1/1000.0] [fnum: 2.8] [ev: 0] [ct: 5500] [color_md: default] [focal_len: 24.00] [latitude: 12.345678] [longitude: 77.123456] [rel_alt: 50.000 abs_alt: 500.000] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="36">SrtCnt : 2, DiffTime : 33ms
2026-08-22 10:00:00.033
[iso: 100] [shutter: 1/1000.0] [fnum: 2.8] [ev: 0] [ct: 5500] [color_md: default] [focal_len: 24.00] [latitude: 12.345680] [longitude: 77.123458] [rel_alt: 50.500 abs_alt: 500.500] </font>

"""


def test_parse_srt_extracts_frames(tmp_path):
    srt_file = tmp_path / "sample.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    frames = parse_srt(str(srt_file))

    assert len(frames) == 2
    assert frames[0].frame_idx == 1
    assert frames[0].timestamp_ms == 0
    assert frames[0].latitude == pytest.approx(12.345678)
    assert frames[0].longitude == pytest.approx(77.123456)
    assert frames[0].rel_altitude_m == pytest.approx(50.000)
    assert frames[0].abs_altitude_m == pytest.approx(500.000)
    assert frames[1].frame_idx == 2
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

    assert altitude_at_timestamp(frames, timestamp_ms=0) == pytest.approx(50.000)
    assert altitude_at_timestamp(frames, timestamp_ms=33) == pytest.approx(50.500)
    assert altitude_at_timestamp(frames, timestamp_ms=15) in (50.000, 50.500)
