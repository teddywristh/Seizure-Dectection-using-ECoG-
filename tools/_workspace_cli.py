from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ds003029_eda.paths import get_paths, resolve_workspace_root


REPORT_FAMILIES = ("timeseries", "ml", "dl")


def resolve_workspace_paths(workspace_root: str | None):
    resolved_root = resolve_workspace_root(workspace_root)
    return get_paths(resolved_root)


def run_python_script(script_name: str, *script_args: str) -> None:
    script_path = REPO_ROOT / "tools" / script_name
    command = [sys.executable, str(script_path), *script_args]
    print("$ " + " ".join(shlex.quote(token) for token in command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(int(completed.returncode))


def load_presets(config_path: str | Path) -> tuple[Path, dict[str, object]]:
    resolved_path, payload = load_config_payload(config_path)
    presets = payload.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid preset config at {resolved_path.as_posix()}: expected an object under 'presets'.")
    return resolved_path, presets


def load_config_payload(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved_path = resolve_repo_path(config_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config at {resolved_path.as_posix()}: expected a top-level JSON object.")
    return resolved_path, payload


def load_family_report_config(config_path: str | Path) -> tuple[Path, list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    resolved_path, payload = load_config_payload(config_path)
    family_order = payload.get("family_order", list(REPORT_FAMILIES))
    family_payloads = payload.get("families", {})
    all_payload = payload.get("all", {})

    if not isinstance(family_order, list) or not all(isinstance(item, str) for item in family_order):
        raise ValueError(f"Invalid report config at {resolved_path.as_posix()}: family_order must be a list of strings.")
    if not isinstance(family_payloads, dict):
        raise ValueError(f"Invalid report config at {resolved_path.as_posix()}: families must be an object.")
    if not isinstance(all_payload, dict):
        raise ValueError(f"Invalid report config at {resolved_path.as_posix()}: all must be an object.")

    resolved_families: dict[str, dict[str, Any]] = {}
    for family in family_order:
        if family not in REPORT_FAMILIES:
            raise ValueError(f"Unsupported report family '{family}' in {resolved_path.as_posix()}.")
        family_spec = family_payloads.get(family)
        if not isinstance(family_spec, dict):
            raise ValueError(
                f"Invalid report config at {resolved_path.as_posix()}: missing object for family '{family}'."
            )
        resolved_families[family] = family_spec

    return resolved_path, family_order, resolved_families, all_payload


def resolve_preset_payload(config_path: str | Path, preset: str) -> tuple[Path, dict[str, Any]]:
    resolved_path, payload = load_config_payload(config_path)
    defaults = payload.get("defaults", {})
    presets = payload.get("presets", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Invalid config at {resolved_path.as_posix()}: expected 'defaults' to be an object.")
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid config at {resolved_path.as_posix()}: expected 'presets' to be an object.")
    if preset not in presets:
        raise KeyError(f"Preset '{preset}' was not found in {resolved_path.as_posix()}.")

    merged = dict(defaults)
    preset_payload = presets[preset]
    if not isinstance(preset_payload, dict):
        raise ValueError(f"Invalid preset '{preset}' in {resolved_path.as_posix()}: expected an object.")
    merged.update(preset_payload)
    return resolved_path, merged


def resolve_repo_path(path_value: str | Path) -> Path:
    resolved_path = Path(path_value)
    if not resolved_path.is_absolute():
        resolved_path = (REPO_ROOT / resolved_path).resolve()
    return resolved_path


def append_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def append_optional_sequence(command: list[str], flag: str, values: Sequence[object] | None) -> None:
    if not values:
        return
    command.append(flag)
    command.extend(str(value) for value in values)