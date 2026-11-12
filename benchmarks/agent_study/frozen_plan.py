"""Build and verify complete P/E study cells independently of their observed outcomes."""
from collections import Counter


def key(cell):
    return (cell["track"], cell["model_id"], cell["method"], cell["library"],
            cell["task_id"], cell["seed"])


def validate_cells(cells):
    if not cells or len({key(c) for c in cells}) != len(cells):
        raise ValueError("study cells must be nonempty and unique")
    if any(type(c["library"]) is not bool or type(c["seed"]) is not int for c in cells):
        raise ValueError("explicit boolean condition and integer seed required")
    if any(c["track"] not in ("P", "E") for c in cells):
        raise ValueError("only the frozen P/E tracks are supported")


def chunks(cells, *, tasks_per_chunk):
    """Partition the complete plan; every paired condition and seed stays in one chunk."""
    validate_cells(cells)
    if type(tasks_per_chunk) is not int or tasks_per_chunk < 1:
        raise ValueError("positive task chunk size required")
    groups = {}
    for cell in cells:
        groups.setdefault((cell["track"], cell["model_id"]), {}).setdefault(cell["task_id"], []).append(cell)
    result = []
    for (track, model_id), tasks in sorted(groups.items()):
        ids = sorted(tasks)
        for start in range(0, len(ids), tasks_per_chunk):
            selected = ids[start:start + tasks_per_chunk]
            result.append(dict(id=f"{track}-{model_id}-{start // tasks_per_chunk:03d}",
                track=track, model_id=model_id, task_ids=selected,
                cells=[cell for task_id in selected for cell in sorted(tasks[task_id], key=key)]))
    if Counter(key(c) for chunk in result for c in chunk["cells"]) != Counter(map(key, cells)):
        raise ValueError("partition changed the complete study scope")
    return result


def validate_report(report, chunk):
    """A completed chunk must account for every frozen cell, with exact task identities."""
    report._validate()
    planned = {(c["method"], c["library"], c["task_id"], c["seed"]): c for c in chunk["cells"]}
    actual = {(r["method"], r["library"], r["task_id"], r["seed"]): r for r in report.rows}
    if len(actual) != len(report.rows) or set(actual) != set(planned):
        raise ValueError("chunk results differ from frozen study cells")
    for key_, row in actual.items():
        cell = planned[key_]
        if row["task_sha256"] != cell["task_sha256"] or row["model_id"] != cell["model"]:
            raise ValueError("result task or model differs from frozen cell")

