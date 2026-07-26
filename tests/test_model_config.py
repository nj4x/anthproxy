"""Tests for anthproxy/model_config.py."""
import json
import os

import pytest

from anthproxy import model_config


# ---------------------------------------------------------------------------
# Defaults — no config file present
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_bedrock_aliases_populated(self):
        aliases = model_config.model_aliases('bedrock')
        assert aliases['sonnet'] == 'anthropic.claude-sonnet-4-5-20250929-v1:0'
        assert aliases['opus'] == 'anthropic.claude-opus-4-8'
        assert aliases['claude-sonnet-4-6'] == 'anthropic.claude-sonnet-4-6'
        # haiku short alias required for auto model routing classifier calls
        assert aliases['haiku'] == 'anthropic.claude-haiku-4-5-20251001-v1:0'

    def test_openrouter_aliases_populated(self):
        aliases = model_config.model_aliases('openrouter')
        assert aliases['sonnet'] == 'z-ai/glm-5.2'
        assert aliases['opus'] == 'moonshotai/kimi-k3'
        assert aliases['haiku'] == 'deepseek/deepseek-v4-flash'

    def test_codex_aliases_populated(self):
        aliases = model_config.model_aliases('codex')
        assert aliases['sonnet'] == 'gpt-5.6-terra'
        assert aliases['opus'] == 'gpt-5.6-sol'
        assert aliases['haiku'] == 'gpt-5.6-luna'

    def test_anthropic_aliases_populated(self):
        aliases = model_config.model_aliases('anthropic')
        assert aliases['opus'] == 'claude-opus-5'
        assert aliases['sonnet'] == 'claude-sonnet-4-6'
        assert aliases['haiku'] == 'claude-haiku-4-5-20251001'

    def test_local_has_default(self):
        aliases = model_config.model_aliases('local')
        assert aliases['default'] == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'

    def test_inference_profile_models_is_frozenset(self):
        ipm = model_config.inference_profile_models()
        assert isinstance(ipm, frozenset)
        assert 'anthropic.claude-sonnet-4-6' in ipm
        assert 'anthropic.claude-opus-4-8' in ipm

    def test_model_pricing_returns_tuples(self):
        pricing = model_config.model_pricing()
        assert isinstance(pricing, dict)
        for tier in ('opus', 'sonnet', 'haiku'):
            assert tier in pricing
            vals = pricing[tier]
            assert isinstance(vals, tuple)
            assert len(vals) == 4
            assert all(isinstance(v, float) for v in vals)
        assert pricing['opus'] == (5.0, 25.0, 0.50, 6.25)
        assert pricing['sonnet'] == (3.0, 15.0, 0.30, 3.75)
        assert pricing['haiku'] == (1.0, 5.0, 0.10, 1.25)

    def test_model_labels_populated(self):
        labels = model_config.model_labels()
        assert labels['opus'] == 'Opus'
        assert labels['sonnet'] == 'Sonnet'
        assert labels['haiku'] == 'Haiku'
        assert labels['other'] == 'Other'

    def test_unknown_backend_returns_empty_dict(self):
        assert model_config.model_aliases('nonexistent') == {}


# ---------------------------------------------------------------------------
# File override / merge
# ---------------------------------------------------------------------------

class TestFileOverride:
    def test_alias_override_replaces_key(self, tmp_path):
        cfg_path = tmp_path / 'cfg.json'
        cfg_path.write_text(json.dumps({
            'model_aliases': {
                'anthropic': {'sonnet': 'my-custom-sonnet'},
            },
        }))
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        aliases = model_config.model_aliases('anthropic')
        assert aliases['sonnet'] == 'my-custom-sonnet'
        # Other keys from defaults must survive the merge
        assert aliases['opus'] == 'claude-opus-5'

    def test_alias_extend_adds_new_key(self, tmp_path):
        cfg_path = tmp_path / 'cfg.json'
        cfg_path.write_text(json.dumps({
            'model_aliases': {
                'codex': {'my-alias': 'my-model'},
            },
        }))
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        aliases = model_config.model_aliases('codex')
        assert aliases['my-alias'] == 'my-model'
        assert aliases['sonnet'] == 'gpt-5.6-terra'   # default still present

    def test_local_default_overridable(self, tmp_path):
        cfg_path = tmp_path / 'cfg.json'
        cfg_path.write_text(json.dumps({
            'model_aliases': {
                'local': {'default': 'llama-3.2'},
            },
        }))
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        aliases = model_config.model_aliases('local')
        assert aliases['default'] == 'llama-3.2'

    def test_inference_profile_list_replaced(self, tmp_path):
        cfg_path = tmp_path / 'cfg.json'
        cfg_path.write_text(json.dumps({
            'bedrock_inference_profile_models': ['custom.model-x'],
        }))
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        ipm = model_config.inference_profile_models()
        assert ipm == frozenset({'custom.model-x'})

    def test_pricing_override(self, tmp_path):
        cfg_path = tmp_path / 'cfg.json'
        cfg_path.write_text(json.dumps({
            'model_pricing': {'opus': [9.9, 99.9, 0.9, 12.0]},
        }))
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        pricing = model_config.model_pricing()
        assert pricing['opus'] == (9.9, 99.9, 0.9, 12.0)
        assert pricing['sonnet'] == (3.0, 15.0, 0.30, 3.75)   # default survives


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

class TestMalformedFile:
    def test_malformed_json_falls_back_to_defaults(self, tmp_path):
        cfg_path = tmp_path / 'cfg.json'
        cfg_path.write_text('{ this is not valid json')
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        # Must not raise; returns defaults
        aliases = model_config.model_aliases('anthropic')
        assert aliases['opus'] == 'claude-opus-5'

    def test_non_dict_root_falls_back_to_defaults(self, tmp_path):
        cfg_path = tmp_path / 'cfg.json'
        cfg_path.write_text('[1, 2, 3]')
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        aliases = model_config.model_aliases('anthropic')
        assert aliases['opus'] == 'claude-opus-5'


# ---------------------------------------------------------------------------
# Cache / ensure_file
# ---------------------------------------------------------------------------

class TestCacheAndEnsureFile:
    def test_load_is_cached(self):
        first = model_config.load()
        second = model_config.load()
        assert first is second

    def test_reset_clears_cache(self):
        first = model_config.load()
        model_config.reset()
        second = model_config.load()
        assert first is not second

    def test_ensure_file_writes_defaults(self, tmp_path):
        cfg_path = tmp_path / 'subdir' / 'config.json'
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        assert not cfg_path.exists()
        model_config.ensure_file()
        assert cfg_path.exists()

        data = json.loads(cfg_path.read_text())
        assert 'model_aliases' in data
        assert 'bedrock' in data['model_aliases']
        assert 'local' in data['model_aliases']
        assert data['model_aliases']['local']['default'] == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'

    def test_ensure_file_does_not_overwrite_existing(self, tmp_path):
        cfg_path = tmp_path / 'config.json'
        cfg_path.write_text('{"custom": true}')
        model_config.reset()
        os.environ['ANTHPROXY_CONFIG'] = str(cfg_path)

        model_config.ensure_file()
        data = json.loads(cfg_path.read_text())
        assert 'custom' in data   # original content preserved


# ---------------------------------------------------------------------------
# Config / parse_args — auto model routing fields
# ---------------------------------------------------------------------------

class TestAutoModelRoutingConfig:
    def test_default_auto_model_routing_is_false(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing is False

    def test_default_classifier_model_is_haiku(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_classifier_model == 'haiku'

    def test_cli_flag_enables_routing(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing'])
        assert cfg.auto_model_routing is True

    def test_cli_flag_disables_routing(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing', '--no-auto-model-routing'])
        assert cfg.auto_model_routing is False

    def test_cli_classifier_model_override(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-classifier-model', 'sonnet'])
        assert cfg.auto_model_routing_classifier_model == 'sonnet'

    def test_env_enables_routing(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_AUTO_MODEL_ROUTING', '1')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing is True

    def test_env_disables_routing(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_AUTO_MODEL_ROUTING', '0')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing is False

    def test_env_classifier_model_override(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_AUTO_MODEL_ROUTING_CLASSIFIER_MODEL', 'opus')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_classifier_model == 'opus'

    def test_default_long_context_threshold_is_150000(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_long_context_threshold == 150_000

    def test_cli_long_context_threshold_override(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-long-context-threshold', '150000'])
        assert cfg.auto_model_routing_long_context_threshold == 150_000

    def test_env_long_context_threshold_override(self, monkeypatch):
        monkeypatch.setenv(
            'ANTHPROXY_AUTO_MODEL_ROUTING_LONG_CONTEXT_THRESHOLD', '120000')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_long_context_threshold == 120_000

    def test_long_context_threshold_zero_allowed(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-long-context-threshold', '0'])
        assert cfg.auto_model_routing_long_context_threshold == 0

    def test_negative_long_context_threshold_rejected(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args(['--auto-model-routing-long-context-threshold', '-1'])

    def test_default_affirmation_inherit_is_on(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_affirmation_inherit is True

    def test_cli_flag_disables_affirmation_inherit(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--no-auto-model-routing-affirmation-inherit'])
        assert cfg.auto_model_routing_affirmation_inherit is False

    def test_cli_flag_enables_affirmation_inherit(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-affirmation-inherit'])
        assert cfg.auto_model_routing_affirmation_inherit is True

    def test_env_disables_affirmation_inherit(self, monkeypatch):
        monkeypatch.setenv(
            'ANTHPROXY_AUTO_MODEL_ROUTING_AFFIRMATION_INHERIT', '0')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_affirmation_inherit is False

    def test_cli_overrides_env_affirmation_inherit(self, monkeypatch):
        monkeypatch.setenv(
            'ANTHPROXY_AUTO_MODEL_ROUTING_AFFIRMATION_INHERIT', '0')
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-affirmation-inherit'])
        assert cfg.auto_model_routing_affirmation_inherit is True

    def test_bedrock_haiku_alias_resolves_correctly(self):
        """Bedrock haiku short alias must resolve to the correct Bedrock model ID."""
        from anthproxy.bedrock.mapper import normalize_model_id
        assert normalize_model_id('haiku') == 'anthropic.claude-haiku-4-5-20251001-v1:0'

    # -----------------------------------------------------------------
    # --auto-model-routing-classification
    # -----------------------------------------------------------------

    def test_classification_default(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_classification == {
            'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'opus',
        }

    def test_classification_all_three_override(self):
        from anthproxy.config import parse_args
        cfg = parse_args([
            '--auto-model-routing-classification',
            'trivial:fable,standard:opus,deep:fable',
        ])
        assert cfg.auto_model_routing_classification == {
            'trivial': 'fable', 'standard': 'opus', 'deep': 'fable',
        }

    def test_classification_partial_override_standard(self):
        from anthproxy.config import parse_args
        cfg = parse_args([
            '--auto-model-routing-classification', 'standard:opus',
        ])
        assert cfg.auto_model_routing_classification == {
            'trivial': 'haiku', 'standard': 'opus', 'deep': 'opus',
        }

    def test_classification_partial_override_deep(self):
        from anthproxy.config import parse_args
        cfg = parse_args([
            '--auto-model-routing-classification', 'deep:fable',
        ])
        assert cfg.auto_model_routing_classification == {
            'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable',
        }

    def test_classification_unknown_label_rejects(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args([
                '--auto-model-routing-classification', 'unknown:fable',
            ])

    def test_classification_empty_model_rejects(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args([
                '--auto-model-routing-classification', 'standard:',
            ])

    def test_classification_malformed_pair_rejects(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args([
                '--auto-model-routing-classification',
                'standard_no_colon_fable',
            ])

    def test_classification_env_var(self, monkeypatch):
        monkeypatch.setenv(
            'ANTHPROXY_AUTO_MODEL_ROUTING_CLASSIFICATION', 'standard:opus')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_classification == {
            'trivial': 'haiku', 'standard': 'opus', 'deep': 'opus',
        }

    def test_classification_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv(
            'ANTHPROXY_AUTO_MODEL_ROUTING_CLASSIFICATION', 'standard:fable')
        from anthproxy.config import parse_args
        cfg = parse_args([
            '--auto-model-routing-classification', 'standard:opus',
        ])
        assert cfg.auto_model_routing_classification == {
            'trivial': 'haiku', 'standard': 'opus', 'deep': 'opus',
        }

    # -----------------------------------------------------------------
    # --auto-model-routing-long
    # -----------------------------------------------------------------

    def test_long_default(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_long == 'off'

    def test_long_model_override(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-long', 'fable'])
        assert cfg.auto_model_routing_long == 'fable'

    def test_long_special_off(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-long', 'off'])
        assert cfg.auto_model_routing_long == 'off'

    def test_long_env_var(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_AUTO_MODEL_ROUTING_LONG', 'fable')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.auto_model_routing_long == 'fable'

    def test_long_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_AUTO_MODEL_ROUTING_LONG', 'opus')
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing-long', 'fable'])
        assert cfg.auto_model_routing_long == 'fable'
