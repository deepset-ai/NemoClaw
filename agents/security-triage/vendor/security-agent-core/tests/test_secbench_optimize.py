"""The SEC-bench target agent must be optimizable offline: the committed seed serializes and the
meta-agent's structured patches apply to it (prompt, step budget, tool descriptions).

The seed YAML (`seeds/secbench.yaml`) is the single source of truth; the tool catalog is
reconstructed from it (not from Python constants)."""

import yaml

from security_agent import paths
from security_agent.optimize import config_patch
from security_agent.optimize.config_patch import PatchPolicy
from security_agent.optimize.tool_catalog import get_catalog
from security_agent.optimize.validate import validate_config

_POLICY = PatchPolicy(approved_models=("gpt-5.4-mini",), max_agent_steps_range=(5, 45))


def _seed() -> dict:
    """The committed SEC-bench seed agent."""
    return yaml.safe_load((paths.PROJECT_ROOT / "seeds" / "secbench.yaml").read_text())


def test_secbench_seed_is_valid():
    cfg = _seed()
    assert validate_config(cfg).ok
    assert sorted(config_patch.enabled_tool_names(cfg)) == [
        "debug_crash",
        "edit_file",
        "find_symbol",
        "read_file",
        "run_shell",
        "search_security_kb",
    ]


def test_kb_tool_survives_a_disable_enable_round_trip():
    """The KB tool is a knob the meta-agent may turn off and back on. Re-enabling must restore it
    fully wired — a re-added tool that lost its JSON schema or its pinned model revisions would
    either confuse the LLM or make `load_kb` refuse the store mid-benchmark."""
    catalog = get_catalog(_seed())

    cfg = config_patch.apply_patch(
        _seed(), [{"op": "disable_tool", "name": "search_security_kb"}], catalog, _POLICY
    )
    assert "search_security_kb" not in config_patch.enabled_tool_names(cfg)
    assert validate_config(cfg).ok

    cfg = config_patch.apply_patch(
        cfg, [{"op": "enable_tool", "name": "search_security_kb"}], catalog, _POLICY
    )
    assert "search_security_kb" in config_patch.enabled_tool_names(cfg)
    assert validate_config(cfg).ok

    tool = next(
        t for t in cfg["init_parameters"]["tools"] if t["data"]["name"] == "search_security_kb"
    )
    assert set(tool["data"]["parameters"]["properties"]) == {"query"}
    params = tool["data"]["component"]["init_parameters"]
    assert params["embedding_revision"] and params["reranker_revision"]
    assert params["query_prefix"]
    # Re-adding the tool without the sub-agent's own knobs would silently fall back to the
    # component defaults, changing what a later iteration is actually scoring.
    assert params["llm_model"] and params["max_agent_steps"]


def test_catalog_comes_from_the_seed():
    cfg = _seed()
    catalog = get_catalog(cfg)
    # The catalog IS the seed's tools, fully wired (so enable_tool can re-add an identical entry).
    assert sorted(catalog) == [
        "debug_crash", "edit_file", "find_symbol", "read_file", "run_shell", "search_security_kb",
    ]
    assert all(catalog[name].description for name in catalog)


def test_patch_system_prompt_and_steps():
    cfg = _seed()
    catalog = get_catalog(cfg)
    patched = config_patch.apply_patch(
        cfg,
        [
            {"op": "set", "path": "system_prompt", "value": "Fix the bug carefully."},
            {"op": "set", "path": "max_agent_steps", "value": 25},
        ],
        catalog,
        _POLICY,
    )
    assert patched["init_parameters"]["system_prompt"] == "Fix the bug carefully."
    assert patched["init_parameters"]["max_agent_steps"] == 25
    assert validate_config(patched).ok


def test_patch_max_steps_out_of_range_rejected():
    cfg = _seed()
    catalog = get_catalog(cfg)
    try:
        config_patch.apply_patch(cfg, [{"op": "set", "path": "max_agent_steps", "value": 99}], catalog, _POLICY)
        assert False, "expected PatchError"
    except config_patch.PatchError:
        pass


def test_patch_max_output_tokens_is_validated():
    """The OpenAI Responses generator uses `max_output_tokens`; it must be bounds/type-checked, and
    the stale Anthropic `max_tokens` kwarg must be rejected."""
    cfg = _seed()
    catalog = get_catalog(cfg)

    def gen_kwargs(c):
        return c["init_parameters"]["chat_generator"]["init_parameters"]["generation_kwargs"]

    ok = config_patch.apply_patch(
        cfg,
        [{"op": "set", "path": "chat_generator.generation_kwargs.max_output_tokens", "value": 8000}],
        catalog,
        _POLICY,
    )
    assert gen_kwargs(ok)["max_output_tokens"] == 8000

    # Out-of-bounds and wrong-type must be rejected (the earlier bug let these through unvalidated).
    for bad in (999999999, "lots"):
        try:
            config_patch.apply_patch(
                cfg,
                [{"op": "set", "path": "chat_generator.generation_kwargs.max_output_tokens", "value": bad}],
                catalog,
                _POLICY,
            )
            assert False, f"expected PatchError for max_output_tokens={bad!r}"
        except config_patch.PatchError:
            pass

    # The stale Anthropic kwarg is no longer a valid path.
    try:
        config_patch.apply_patch(
            cfg,
            [{"op": "set", "path": "chat_generator.generation_kwargs.max_tokens", "value": 8000}],
            catalog,
            _POLICY,
        )
        assert False, "expected PatchError for the removed max_tokens key"
    except config_patch.PatchError:
        pass


def test_set_tool_description_and_toggle():
    cfg = _seed()
    catalog = get_catalog(cfg)

    described = config_patch.apply_patch(
        cfg,
        [{"op": "set_tool_description", "name": "run_shell", "value": "Build before you finish."}],
        catalog,
        _POLICY,
    )
    entry = next(t for t in described["init_parameters"]["tools"] if t["data"]["name"] == "run_shell")
    assert entry["data"]["description"] == "Build before you finish."

    disabled = config_patch.apply_patch(cfg, [{"op": "disable_tool", "name": "edit_file"}], catalog, _POLICY)
    assert "edit_file" not in config_patch.enabled_tool_names(disabled)
    re_enabled = config_patch.apply_patch(disabled, [{"op": "enable_tool", "name": "edit_file"}], catalog, _POLICY)
    assert "edit_file" in config_patch.enabled_tool_names(re_enabled)
    assert validate_config(re_enabled).ok
