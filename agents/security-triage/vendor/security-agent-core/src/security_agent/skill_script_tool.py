"""A generic execution surface for Haystack SkillToolset skills.

Haystack's `SkillToolset` is read-only by design: `load_skill` returns a skill's
instructions plus a manifest of bundled files, but nothing can run them. This
module supplies the missing half as ONE reusable tool: pass the same skills
directory that backs the `FileSystemSkillStore`, and any skill following the
"bundled scripts + optional venv/requirements.txt" convention becomes runnable
with zero per-skill wiring.

    from haystack.tools import SkillToolset
    from haystack.skill_stores.file_system import FileSystemSkillStore
    from skill_script_tool import make_skill_script_tool

    toolset = SkillToolset(FileSystemSkillStore("skills/"))
    exec_tool = make_skill_script_tool("skills/")
    agent = Agent(chat_generator=..., tools=[toolset, exec_tool])

Guardrails: only Python scripts that physically live inside the named skill's
directory can run, arguments are passed as argv (never through a shell), and
each skill's scripts run under that skill's own venv python3 — provisioned
automatically from its requirements.txt on first use when missing.
"""

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from haystack.tools import Tool, create_tool_from_function

_TIMEOUT_S = 600
_MAX_OUTPUT = 40_000


def _interpreter(skill_dir: Path, auto_venv: bool) -> Path:
    """The skill's venv python3, provisioning it from requirements.txt if allowed."""
    venv_py = skill_dir / "venv/bin/python3"
    if venv_py.exists():
        return venv_py
    requirements = skill_dir / "requirements.txt"
    if auto_venv and requirements.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", str(skill_dir / "venv")],
            check=True, capture_output=True, timeout=_TIMEOUT_S,
        )
        subprocess.run(
            [str(venv_py), "-m", "pip", "install", "-q", "-r", str(requirements)],
            check=True, capture_output=True, timeout=_TIMEOUT_S,
        )
        return venv_py
    return Path(sys.executable)


def make_skill_script_tool(skills_dir: str | Path, auto_venv: bool = True) -> Tool:
    """
    Build a `run_skill_script` tool over the same directory that backs the skill store.

    :param skills_dir: Root directory containing one subdirectory per skill.
    :param auto_venv: Provision a skill's venv from its requirements.txt on first use.
    :returns: A Haystack `Tool` to pass to an Agent alongside the `SkillToolset`.
    """
    root = Path(skills_dir).resolve()

    def run_skill_script(
        skill: Annotated[str, "Name of the skill that bundles the script."],
        script: Annotated[str, "Script path relative to the skill root, e.g. 'scripts/sweep.py'."],
        args: Annotated[str, "Arguments for the script, space-separated (quote paths with spaces)."] = "",
    ) -> str:
        """
        Execute a Python script bundled with a skill, under the skill's own venv.

        Use this to run the helper scripts a loaded skill's instructions tell you to
        run. Only scripts inside the named skill's directory are permitted.
        """
        skill_dir = (root / skill).resolve()
        if not skill_dir.is_relative_to(root) or not (skill_dir / "SKILL.md").is_file():
            return f"REJECTED: unknown skill {skill!r}."
        script_path = (skill_dir / script).resolve()
        if not script_path.is_relative_to(skill_dir) or script_path.suffix != ".py":
            return f"REJECTED: {script!r} is not a Python script inside the {skill!r} skill."
        if not script_path.is_file():
            return f"REJECTED: {script!r} does not exist in the {skill!r} skill."

        try:
            python = _interpreter(skill_dir, auto_venv)
        except subprocess.CalledProcessError as e:
            return f"venv setup failed: {(e.stderr or b'').decode(errors='replace')[-2000:]}"

        proc = subprocess.run(
            [str(python), str(script_path), *shlex.split(args)],
            capture_output=True, text=True, timeout=_TIMEOUT_S, cwd=skill_dir,
        )
        out = proc.stdout[-_MAX_OUTPUT:]
        if proc.returncode != 0:
            return f"exit {proc.returncode}\n{proc.stderr[-4000:]}\n{out}"
        return out

    return create_tool_from_function(function=run_skill_script, name="run_skill_script")
