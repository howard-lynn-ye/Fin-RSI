"""First-request orientation for the opt-in v6 interface, not frozen experiments."""
import hashlib
from pathlib import Path


GUIDE = Path(__file__).with_name("trading_library_guide_v6.md")


def orientation(arm):
    if arm == "raw":
        return "", None
    if arm != "library":
        raise ValueError(f"unknown arm: {arm}")
    # Never silently omit or truncate the guide if it is missing or oversized.
    text = GUIDE.read_text(encoding="utf-8")
    if not text.strip() or len(text) > 6500:
        raise ValueError("library orientation must be nonempty and <=6500 characters")
    return text, {"source": GUIDE.name, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "chars": len(text), "scope": "initial request; not proof of comprehension"}


def document_retrieval(result):
    """Record returned pages, without mistaking a Python retrieval for model exposure."""
    if not result.get("ok"):
        return {}
    text = result["text"]
    return {"skill_name": result["name"], "reference": result.get("reference"),
            "offset": result["offset"], "next_offset": result["next_offset"],
            "returned_chars": len(text), "total_chars": result["total_chars"],
            "page_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "evidence": "retrieval only; not proof of reading or use"}
