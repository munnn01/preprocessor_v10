from pathlib import Path

from tools.parse_tegrastats import parse_power_mw, summarize


def test_parse_current_vdd_in() -> None:
    rail, values = parse_power_mw("RAM 123/100 VDD_IN 7421mW/8050mW VDD_CPU_GPU_CV 1000mW/1200mW\n")
    assert rail == "VDD_IN"
    assert values == [7421.0]


def test_summarize_legacy_rail(tmp_path: Path) -> None:
    source = tmp_path / "tegrastats.log"
    source.write_text("POM_5V_IN 5000mW/5500mW\nPOM_5V_IN 7000mW/7100mW\n")
    report = summarize(source)
    assert report["rail"] == "POM_5V_IN"
    assert report["samples"] == 2
    assert report["power_w"]["mean"] == 6.0
