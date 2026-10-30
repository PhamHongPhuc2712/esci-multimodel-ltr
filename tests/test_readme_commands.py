"""Every `python -m esci_rerank.<module>` line in the README, against its module.

A reproduction that follows the README types these lines. A flag the module
no longer accepts, or a module that no longer imports, is the kind of thing
nothing else in the suite would notice, because the suite calls functions
and the README calls commands. Each command is run with --help and every
flag the line uses must appear in the help text. Marked `slow`: each run
imports the module, and the heavy ones import torch.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = (Path("README.md"),)
_FENCE = re.compile(r"```bash\n(.*?)```", re.S)
_FLAG = re.compile(r"(?<!\S)(--[a-z][a-z0-9-]*)")


def commands_in(path: Path) -> list[str]:
    """Each `python -m esci_rerank.<module> ...` command in the file's bash fences.

    Backslash continuations are folded, trailing `# comments` dropped, and a
    `for ... ; do A; B; done` line is split into its commands.
    """
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for block in _FENCE.findall(text):
        joined = block.replace("\\\n", " ")
        for raw in joined.splitlines():
            line = raw.split(" #", 1)[0].split("\t#", 1)[0].strip()
            if line.startswith("#"):
                continue
            for part in re.split(r"\s*(?:&&|;)\s*", line):
                part = re.sub(r"\s+", " ", part).strip()
                part = re.sub(r"^(?:do|then) ", "", part)
                if part.startswith("python -m esci_rerank."):
                    out.append(part)
    return out


def flags_of(command: str) -> set[str]:
    return set(_FLAG.findall(command))


def module_of(command: str) -> str:
    return shlex.split(command)[2]


_COMMANDS = sorted({c for doc in DOCS for c in commands_in(doc)})


def test_the_docs_list_commands():
    assert len(_COMMANDS) > 20


def test_commands_are_found_through_loops_and_continuations(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        "```bash\n"
        "for s in train test; do python -m esci_rerank.a --split $s; python -m esci_rerank.b --x 1; done\n"
        "python -m esci_rerank.c --long-flag 2 \\\n    --other 3   # note\n"
        "python -m esci_rerank.d --a && python -m esci_rerank.e --b\n"
        "```\n",
        encoding="utf-8",
    )
    found = commands_in(doc)
    assert "python -m esci_rerank.a --split $s" in found
    assert "python -m esci_rerank.b --x 1" in found
    assert "python -m esci_rerank.c --long-flag 2 --other 3" in found
    assert {"python -m esci_rerank.d --a", "python -m esci_rerank.e --b"} <= set(found)
    assert flags_of("python -m esci_rerank.c --long-flag 2 --other 3") == {"--long-flag", "--other"}


@pytest.mark.slow
@pytest.mark.parametrize("command", _COMMANDS)
def test_the_module_accepts_every_flag_the_docs_use(command):
    module = module_of(command)
    sub = ""
    # Some modules have subcommands (the evaluation CLI, the LLM trainers);
    # help for the subcommand is where its flags live. A module without
    # subcommands still prints its help when a positional word comes first.
    parts = shlex.split(command)
    if len(parts) > 3 and not parts[3].startswith("-"):
        sub = parts[3]
    argv = [sys.executable, "-m", module, *([sub] if sub else []), "--help"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"{' '.join(argv)} failed:\n{result.stderr[-2000:]}"
    missing = sorted(f for f in flags_of(command) if f not in result.stdout)
    assert not missing, f"{command}\n  flags not in `{module} --help`: {missing}"
