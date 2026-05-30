"""Shared Hydra helpers for taba experiment scripts.

All experiment entrypoints read their configuration from the repository-level
``configs/`` directory. Because the scripts live at varying depths under
``taba/scripts/``, we expose an absolute path to that directory so every script
can use the same ``config_path`` regardless of where it lives.
"""

from pathlib import Path

# taba/taba/_hydra.py -> taba/taba -> taba (repo root) -> taba/configs
CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "configs"))
