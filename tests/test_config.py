from pathlib import Path

from pcmno.config import apply_overrides, load_config


def test_configs_load():
    root = Path(__file__).resolve().parents[1]
    for name in ("pns_paper.yaml", "lc_paper.yaml", "pns_smoke.yaml", "lc_smoke.yaml"):
        cfg = load_config(root / "configs" / name)
        assert cfg["dataset"]["name"] in {"pns", "lc"}


def test_nested_override():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "pns_smoke.yaml")
    updated = apply_overrides(cfg, ["training.epochs=3", "runtime.device=cpu"])
    assert updated["training"]["epochs"] == 3
    assert updated["runtime"]["device"] == "cpu"
