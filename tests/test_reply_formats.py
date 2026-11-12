"""scripts/reply_formats.py: prose answers are counted apart from programs."""
import json

from scripts import reply_formats


def _events(path, replies):
    lines = [dict(kind="fit_started"),
             dict(kind="call_completed", role="writer", content="not counted: fitting")]
    for library, texts in replies.items():
        lines.append(dict(kind="episode_started", method="external_code", library=library,
                          task_id="t1", seed=11))
        lines += [dict(kind="call_completed", role="writer", content=t) for t in texts]
        lines.append(dict(kind="call_completed", role="reviewer:timing", content="no code"))
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_counts_calls_and_final_replies_per_chunk_and_arm(tmp_path, capsys):
    _events(tmp_path / "E-phi4-mini-003.jsonl.events.jsonl",
            {True: ["The answer is 4.", "```Python\n2 + 2\n```"],
             False: ["```python\n2 + 2"]})
    reply_formats.main([str(tmp_path)])
    rows = {(r["group"], r["library"]): r for r in json.loads(capsys.readouterr().out)}
    assert rows[("E-phi4-mini", True)]["calls"] == {"no_fence": 1, "fenced": 1}
    assert rows[("E-phi4-mini", True)]["final"] == {"fenced": 1}
    assert rows[("E-phi4-mini", False)]["final"] == {"unclosed_fence": 1}
