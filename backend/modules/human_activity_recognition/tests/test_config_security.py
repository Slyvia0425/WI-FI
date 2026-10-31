from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.modules.human_activity_recognition.config import load_config
from backend.modules.human_activity_recognition.errors import (
    AuthorizationError,
    ConfigurationError,
    InputValidationError,
)
from backend.modules.human_activity_recognition.security import AccessPolicy, resolve_under


class ConfigSecurityTest(unittest.TestCase):
    def test_config_is_deterministic_and_env_driven(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"HAR_DATA_ROOT": str(root / "data"), "HAR_OUTPUT_ROOT": str(root / "out")}
            first = load_config(env=env)
            second = load_config(env=env)
            self.assertEqual(first.config_hash, second.config_hash)
            self.assertEqual(first.paths.output_root, (root / "out").resolve())
            self.assertEqual(first.secret_env_names["remote_token"], first.runtime.remote_token_env)

    def test_invalid_mode_is_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                Path("backend/modules/human_activity_recognition/config.yaml")
                .read_text()
                .replace("mode: local", "mode: invalid")
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(InputValidationError):
                resolve_under(root, "../escape")

    def test_role_policy_denies_viewer_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = AccessPolicy.build(roles=["viewer"], readable_roots=[root], writable_roots=[root])
            with self.assertRaises(AuthorizationError):
                policy.authorize_read("viewer", root / "raw.h5")
            admin = AccessPolicy.build(roles=["system"], readable_roots=[root], writable_roots=[root])
            self.assertEqual(admin.authorize_write("system", root / "out.json"), (root / "out.json").resolve())
