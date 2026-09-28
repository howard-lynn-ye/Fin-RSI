"""Declarative Skill Scaffolder & In-Place Upgrader (`fin_skills.skill_rsi.scaffold`).

Automates the creation and iterative upgrading of FinSkills so every skill upgrade is
validated against the 6-field Agent Skills specification (`<= 1024` char description,
third-person sentence opening, contrastive `TRIGGER` / `SKIP` clauses, bidirectional
1-hop `xref` graph, dual-mode executable Python module, and marketplace registration)
before touching package generation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
RESERVED_WORDS = ("anthropic", "claude")
DESC_HARD_CAP = 1024
BODY_CHAR_CAP = 24_000


@dataclass
class SkillUpgradeBlueprint:
    """Declarative blueprint for creating or upgrading a FinSkill in `plugins/<plugin>/skills/<name>/`."""

    name: str
    plugin: str
    module_name: str
    description: str
    body_markdown: str
    script_source: str
    verified_on: str = "2026-09-28"
    license: str = "MIT"
    compatibility: str = "Python 3.10+, numpy, pandas"
    xref_neighbors: tuple[str, ...] = ()
    guard_name: str | None = None
    probe_queries: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ScaffoldResult:
    """Outcome of scaffolding or upgrading a FinSkill."""

    skill_name: str
    plugin: str
    skill_md_path: str
    script_path: str
    description_chars: int
    body_chars: int
    xref_updated_neighbors: list[str]
    marketplace_updated: bool
    validation_errors: list[str]

    @property
    def passed(self) -> bool:
        return len(self.validation_errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "plugin": self.plugin,
            "passed": self.passed,
            "skill_md_path": self.skill_md_path,
            "script_path": self.script_path,
            "description_chars": self.description_chars,
            "body_chars": self.body_chars,
            "xref_updated_neighbors": list(self.xref_updated_neighbors),
            "marketplace_updated": self.marketplace_updated,
            "validation_errors": list(self.validation_errors),
        }


def validate_skill_blueprint(bp: SkillUpgradeBlueprint) -> list[str]:
    """Validate a `SkillUpgradeBlueprint` against every rule in `scripts/validate.py`."""
    errs: list[str] = []
    if not NAME_RE.match(bp.name):
        errs.append(f"name {bp.name!r} must match ^[a-z0-9]+(-[a-z0-9]+)*$")
    if len(bp.name) > 64:
        errs.append(f"name is {len(bp.name)} chars (max 64)")
    for w in RESERVED_WORDS:
        if w in bp.name:
            errs.append(f"name contains reserved word {w!r}")
    if not bp.module_name.isidentifier() or bp.module_name.startswith("_"):
        errs.append(f"module_name {bp.module_name!r} must be a public Python identifier")

    desc = " ".join(x.strip() for x in bp.description.strip().splitlines() if x.strip())
    if not desc:
        errs.append("description must be non-empty")
    elif len(desc) > DESC_HARD_CAP:
        errs.append(f"description is {len(desc)} chars (spec hard cap {DESC_HARD_CAP})")

    for key in ("license", "metadata", "name", "compatibility", "allowed-tools"):
        if re.search(rf"(?<![\w-]){key}\s*:", desc):
            errs.append(f"description contains '{key}:' which looks like a swallowed YAML key")

    clean_desc = re.sub(r"^\[[\w-]+\]\s*", "", desc)
    first = clean_desc.split()[0].strip("`*.,").lower() if clean_desc.split() else ""
    bare = bp.name.replace("lib-", "")
    starts_with_own_name = bool(first) and (first in bare or bare in first)
    if (
        clean_desc
        and not clean_desc[0].isupper()
        and not clean_desc.startswith(("A ", "An ", "The "))
        and not starts_with_own_name
    ):
        errs.append("description must start as a third-person sentence (capitalized first word)")

    if "TRIGGER" not in desc:
        errs.append("description should include an explicit 'TRIGGER - ...' clause for routing")
    if "SKIP" not in desc:
        errs.append("description should include an explicit 'SKIP for ...' clause for disambiguation")

    expected_script_ref = f"scripts/{bp.module_name}.py"
    if f"`{expected_script_ref}`" not in bp.body_markdown:
        errs.append(
            f"body_markdown must reference `{expected_script_ref}` in backticks so validate.py "
            f"tracks the executable script"
        )
    for nb in bp.xref_neighbors:
        if f"`{nb}`" not in bp.body_markdown and f"`{nb}`" not in desc:
            errs.append(
                f"xref neighbor `{nb}` must be referenced in backticks in body_markdown or description"
            )
    if not bp.script_source.strip():
        errs.append("script_source must be non-empty")

    full_text = render_skill_markdown(bp)
    if len(full_text) > BODY_CHAR_CAP:
        errs.append(f"rendered SKILL.md is {len(full_text)} chars (max {BODY_CHAR_CAP})")
    return errs


def render_skill_markdown(bp: SkillUpgradeBlueprint) -> str:
    """Render the complete `SKILL.md` file with strict 6-field Agent Skills frontmatter."""
    desc_lines = [line.strip() for line in bp.description.strip().splitlines() if line.strip()]
    indented_desc = "\n".join(f"  {line}" for line in desc_lines)
    frontmatter = (
        "---\n"
        f"name: {bp.name}\n"
        "description: >-\n"
        f"{indented_desc}\n"
        f"license: {bp.license}\n"
        f"compatibility: {bp.compatibility}\n"
        "metadata:\n"
        f"  verified_on: \"{bp.verified_on}\"\n"
        "---\n\n"
    )
    return frontmatter + bp.body_markdown.strip() + "\n"


def ensure_bidirectional_xref(
    root: Path,
    skill_name: str,
    neighbor_skills: Sequence[str],
    summary_hint: str = "Related RSI skill.",
) -> list[str]:
    """Ensure every neighbor skill's `SKILL.md` links back to `` `skill_name` `` in backticks."""
    updated: list[str] = []
    for nb in neighbor_skills:
        matches = list(root.glob(f"plugins/*/skills/{nb}/SKILL.md"))
        if not matches:
            continue
        nb_md = matches[0]
        text = nb_md.read_text(encoding="utf-8")
        if f"`{skill_name}`" in text:
            continue
        bullet = f"- `{skill_name}` — {summary_hint}\n"
        if "## Cross-References" in text or "## Where this sits" in text or "## Related" in text:
            text = text.rstrip() + "\n" + bullet
        else:
            text = text.rstrip() + f"\n\n## Related Skills\n\n{bullet}"
        nb_md.write_text(text, encoding="utf-8", newline="\n")
        updated.append(nb)
    return updated


def sync_marketplace_entry(root: Path, plugin: str, skill_name: str) -> bool:
    """Ensure `.claude-plugin/marketplace.json` lists `./skills/<skill_name>` under `plugin`."""
    mp_path = root / ".claude-plugin" / "marketplace.json"
    if not mp_path.exists():
        return False
    data = json.loads(mp_path.read_text(encoding="utf-8"))
    changed = False
    target_entry = f"./skills/{skill_name}"
    for p in data.get("plugins", []):
        if p.get("name") == plugin and isinstance(p.get("source"), str):
            skills_list = list(p.get("skills", []))
            if target_entry not in skills_list:
                skills_list.append(target_entry)
                p["skills"] = skills_list
                changed = True
            break
    if changed:
        mp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return changed


def apply_skill_blueprint(
    bp: SkillUpgradeBlueprint,
    root: Path | None = None,
) -> ScaffoldResult:
    """Validate and write a new or upgraded FinSkill (`SKILL.md` + `scripts/<mod>.py` + `xref` + marketplace)."""
    repo_root = root or Path(__file__).resolve().parent.parent.parent
    errs = validate_skill_blueprint(bp)
    skill_dir = repo_root / "plugins" / bp.plugin / "skills" / bp.name
    skill_md_path = skill_dir / "SKILL.md"
    script_path = skill_dir / "scripts" / f"{bp.module_name}.py"
    desc_one_line = " ".join(x.strip() for x in bp.description.strip().splitlines() if x.strip())
    rendered_md = render_skill_markdown(bp)

    if errs:
        return ScaffoldResult(
            skill_name=bp.name,
            plugin=bp.plugin,
            skill_md_path=str(skill_md_path),
            script_path=str(script_path),
            description_chars=len(desc_one_line),
            body_chars=len(rendered_md),
            xref_updated_neighbors=[],
            marketplace_updated=False,
            validation_errors=errs,
        )

    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    skill_md_path.write_text(rendered_md, encoding="utf-8", newline="\n")
    script_path.write_text(bp.script_source.strip() + "\n", encoding="utf-8", newline="\n")

    short_hint = re.sub(r"\s*TRIGGER\s*-.*$", "", desc_one_line).strip()
    xref_updated = ensure_bidirectional_xref(
        repo_root, bp.name, bp.xref_neighbors, summary_hint=short_hint
    )
    mp_updated = sync_marketplace_entry(repo_root, bp.plugin, bp.name)

    return ScaffoldResult(
        skill_name=bp.name,
        plugin=bp.plugin,
        skill_md_path=str(skill_md_path),
        script_path=str(script_path),
        description_chars=len(desc_one_line),
        body_chars=len(rendered_md),
        xref_updated_neighbors=xref_updated,
        marketplace_updated=mp_updated,
        validation_errors=[],
    )
