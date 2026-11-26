"""External port registry support (--ports-file and {ports_root})."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conformance.paths import REPO
from conformance.runner.__main__ import build_parser, load_ports, resolve_cmd


def test_load_ports_from_external_file(tmp_path: Path):
    registry = tmp_path / "ports.json"
    registry.write_text(json.dumps({
        "_comment": "external registry",
        "go": {"cmd": ["{ports_root}/go/bin/rdocs-conform"]},
        "python": {"cmd": ["{python}", "-m", "conformance.refcli"]},
    }), encoding="utf-8")

    ports = load_ports(registry)
    assert set(ports) == {"go", "python"}
    assert "_comment" not in ports


def test_load_ports_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_ports(tmp_path / "does-not-exist.json")


def test_resolve_cmd_expands_ports_root_and_defaults(tmp_path: Path):
    spec = {"cmd": ["{python}", "{repo}/tool", "{ports_root}/go/bin/rdocs-conform"]}

    default_cmd = resolve_cmd(spec)
    assert default_cmd == [sys.executable, f"{REPO}/tool", f"{REPO / 'ports'}/go/bin/rdocs-conform"]

    external_cmd = resolve_cmd(spec, ports_root=tmp_path)
    assert external_cmd == [sys.executable, f"{REPO}/tool", f"{tmp_path.resolve()}/go/bin/rdocs-conform"]


def test_cli_parser_accepts_ports_file_on_list_and_run():
    parser = build_parser()
    list_args = parser.parse_args(["list", "--ports-file", "custom/ports.json"])
    assert list_args.ports_file == "custom/ports.json"

    run_args = parser.parse_args(["run", "--port", "go", "--ports-file", "custom/ports.json"])
    assert run_args.port == "go"
    assert run_args.ports_file == "custom/ports.json"
