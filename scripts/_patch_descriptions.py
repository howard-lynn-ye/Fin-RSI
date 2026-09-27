# -*- coding: utf-8 -*-
"""Targeted Skill-Level RSI pass on SKILL.md descriptions, driven by eval_triggers.py & eval_blind.py.

Synchronizes the 48 evolved skill descriptions (Gen-1 Contrastive TRIGGER Amplification +
Gen-2 Orthogonal Negative-SKIP Disambiguation) and Gen-3 1-hop body cross-references (`xref`).
Idempotent and re-runnable.
Run:  python scripts/_patch_descriptions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_skill_level_rsi import (  # noqa: E402
    GEN2_DESCRIPTIONS,
    GEN3_BODY_XREFS,
    write_skill_descriptions_and_xrefs,
)


def main() -> int:
    modified = write_skill_descriptions_and_xrefs(GEN2_DESCRIPTIONS, GEN3_BODY_XREFS)
    for name in sorted(GEN2_DESCRIPTIONS):
        desc = GEN2_DESCRIPTIONS[name]
        if len(desc) > 1024:
            print(f"  !! {name} now {len(desc)} chars (max 1024)")
            return 1
        print(f"  {len(desc):>5}  {name}")
    print(f"\nverified {len(GEN2_DESCRIPTIONS)} evolved skills (modified {modified})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
