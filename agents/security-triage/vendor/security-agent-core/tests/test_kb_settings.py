"""Covers knowledge-base settings resolution.

Two things matter here. The precedence order (argument > env > config.yaml > default) is what
`secagent kb build --device mps` and `KB_PROFILE=standard` rely on. And `load_kb_settings` must never
raise: `chunking.ChunkStrategy` reads it at class-definition time, so an exception would break
every import of every curation client.
"""
import pytest

from security_agent.knowledge_base.settings import DEFAULT_PROFILES, KbSettings, load_kb_settings


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An isolated project root, with the KB_* env cleared."""
    for name in ("KB_DIR", "KB_PROFILE", "KB_DEVICE", "KB_EMBEDDING_MODEL", "KB_RERANKER_MODEL",
                 "KB_BATCH_SIZE", "KB_ST_BATCH_SIZE", "KB_OFFLINE", "NVD_LOOKBACK_DAYS",
                 "NVD_MIN_CVSS"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _write_config(root, body: str):
    (root / "config.yaml").write_text(body)


def test_defaults_apply_when_there_is_no_config(project):
    settings = load_kb_settings(project)
    assert settings == KbSettings()
    assert settings.device == "cpu"
    assert settings.embedding_dim == 384
    assert settings.sources("dev") == ["cwe"]


def test_config_yaml_is_read(project):
    _write_config(project, "knowledge_base:\n  profile: standard\n  device: mps\n  nvd_min_cvss: 9.0\n")
    settings = load_kb_settings(project)
    assert settings.profile == "standard"
    assert settings.device == "mps"
    assert settings.nvd_min_cvss == 9.0
    # Untouched keys keep their defaults.
    assert settings.embedding_model == KbSettings().embedding_model


def test_env_beats_config(project, monkeypatch):
    _write_config(project, "knowledge_base:\n  profile: standard\n  device: mps\n")
    monkeypatch.setenv("KB_DEVICE", "cpu")
    monkeypatch.setenv("KB_PROFILE", "standard")
    monkeypatch.setenv("NVD_LOOKBACK_DAYS", "30")
    settings = load_kb_settings(project)
    assert settings.device == "cpu"
    assert settings.profile == "standard"
    assert settings.nvd_lookback_days == 30


def test_arguments_beat_env(project, monkeypatch):
    monkeypatch.setenv("KB_PROFILE", "standard")
    assert load_kb_settings(project, profile="full").profile == "full"
    # None arguments are ignored, so argparse defaults can be passed straight through.
    assert load_kb_settings(project, profile=None).profile == "standard"


def test_unparseable_env_values_are_ignored(project, monkeypatch):
    monkeypatch.setenv("KB_BATCH_SIZE", "lots")
    monkeypatch.setenv("NVD_MIN_CVSS", "high")
    settings = load_kb_settings(project)
    assert settings.batch_size == KbSettings().batch_size
    assert settings.nvd_min_cvss == KbSettings().nvd_min_cvss


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True),
                                            ("0", False), ("false", False)])
def test_boolean_env_parsing(project, monkeypatch, value, expected):
    monkeypatch.setenv("KB_OFFLINE", value)
    assert load_kb_settings(project).offline is expected


def test_a_corrupt_config_falls_back_to_defaults_without_raising(project):
    """Must not raise: chunking reads this at import time."""
    _write_config(project, "knowledge_base: [this, is, not, a, mapping]\n")
    assert load_kb_settings(project) == KbSettings()

    _write_config(project, "knowledge_base:\n  profile: standard\n   bad indent: {\n")
    assert load_kb_settings(project) == KbSettings()

    _write_config(project, "knowledge_base:\n  no_such_setting: 1\n  profile: standard\n")
    assert load_kb_settings(project).profile == "standard"  # unknown keys are ignored


def test_a_partial_profile_map_keeps_the_built_in_profiles(project):
    """A YAML override must not delete `dev`, which CI and the integration test build against."""
    _write_config(project, "knowledge_base:\n  profiles:\n    tiny: [cwe]\n")
    settings = load_kb_settings(project)
    assert settings.sources("tiny") == ["cwe"]
    assert settings.sources("full") == DEFAULT_PROFILES["full"]


def test_an_unknown_profile_is_a_clear_error(project):
    with pytest.raises(ValueError, match="Unknown knowledge-base profile"):
        load_kb_settings(project).sources("gigantic")


def test_paths_resolve_against_the_project_root():
    settings = KbSettings()
    from security_agent import paths

    assert settings.resolve("data/kb") == paths.PROJECT_ROOT / "data" / "kb"
    assert settings.resolve("/absolute/kb").as_posix() == "/absolute/kb"


def test_the_committed_config_matches_the_dataclass_defaults():
    """config.yaml and KbSettings must agree, or `kb build` and the seed snippet drift apart."""
    settings = load_kb_settings()
    defaults = KbSettings()
    assert settings.embedding_model == defaults.embedding_model
    assert settings.embedding_dim == defaults.embedding_dim
    assert settings.query_prefix == defaults.query_prefix
    assert settings.reranker_model == defaults.reranker_model
    assert settings.device == defaults.device
    assert settings.profiles == DEFAULT_PROFILES
