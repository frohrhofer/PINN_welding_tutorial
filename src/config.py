# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Loading of the YAML configuration files."""

from pathlib import Path

import yaml


def load_config(
        file: str | Path,
        config_update: dict | None = None,
    ) -> dict:
    """Loads a YAML configuration file and optionally updates it.

    Args:
        file (str | Path): Path to the YAML file.
        config_update (dict, None): Top-level entries that overwrite the
            loaded configuration (e.g. for parameter studies).

    Returns:
        config (dict): Configuration dictionary.
    """
    with open(file) as f:
        config = yaml.safe_load(f)

    if config_update:
        config.update(config_update)

    return config
