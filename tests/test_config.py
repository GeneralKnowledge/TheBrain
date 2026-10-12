from pathlib import Path

import pytest

from cogrt import config as config_mod

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_valid() -> None:
    cfg = config_mod.load()
    assert cfg.model.backend == "mock"
    assert cfg.model.device == "cpu"


@pytest.mark.parametrize("name", ["mock", "laptop-cpu", "laptop-gpu", "research-transformers", "remote"])
def test_shipped_configs_load(name: str) -> None:
    cfg = config_mod.load(ROOT / "configs" / f"{name}.toml")
    assert cfg.data_path.is_absolute()


def test_unknown_key_rejected() -> None:
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(overrides={"scheduler": {"intervall_s": 3}})


def test_unknown_section_rejected() -> None:
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(overrides={"schedular": {}})


def test_type_checked() -> None:
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(overrides={"scheduler": {"max_cycles_per_hour": "many"}})


def test_remote_requires_network_permission() -> None:
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(overrides={"model": {"backend": "remote", "remote_url": "http://x"}})


def test_confidence_bounds() -> None:
    with pytest.raises(config_mod.ConfigError):
        config_mod.load(overrides={"cycle": {"initial_hypothesis_confidence": 0.9,
                                             "max_hypothesis_confidence": 0.5}})
