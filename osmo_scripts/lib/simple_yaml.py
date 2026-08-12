# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small YAML subset reader for OSMO preset files.

This parser intentionally supports only the subset used by
``osmo_scripts/presets/*.yaml`` so OSMO helpers do not need a PyYAML dependency
before the IsaacLab environment is initialized.
"""

from __future__ import annotations

from pathlib import Path


def _parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value == "true":
        return True
    if value == "false":
        return False
    if value in ("null", "None"):
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Expected key-value pair, got: {text!r}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _prepare_lines(path: Path) -> list[tuple[int, str]]:
    prepared = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        prepared.append((indent, raw_line.strip()))
    return prepared


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int):
    if index >= len(lines) or lines[index][0] < indent:
        return {}, index

    if lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int):
    result = []
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent != indent or not text.startswith("- "):
            break

        item_text = text[2:].strip()
        index += 1
        if item_text == "":
            item, index = _parse_block(lines, index, indent + 2)
        elif ":" in item_text:
            key, value = _split_key_value(item_text)
            item = {}
            if value:
                item[key] = _parse_scalar(value)
            else:
                item[key], index = _parse_block(lines, index, indent + 2)
            if index < len(lines) and lines[index][0] == indent + 2 and not lines[index][1].startswith("- "):
                extra, index = _parse_mapping(lines, index, indent + 2)
                item.update(extra)
        else:
            item = _parse_scalar(item_text)
        result.append(item)
    return result, index


def _parse_mapping(lines: list[tuple[int, str]], index: int, indent: int):
    result = {}
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent != indent or text.startswith("- "):
            break

        key, value = _split_key_value(text)
        index += 1
        if value:
            result[key] = _parse_scalar(value)
        else:
            result[key], index = _parse_block(lines, index, indent + 2)
    return result, index


def load_yaml(path: str | Path):
    """Load a small YAML subset from a file."""
    lines = _prepare_lines(Path(path))
    if not lines:
        return {}
    data, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError(f"Could not parse all lines in {path}. Stopped at line item {index}.")
    return data
