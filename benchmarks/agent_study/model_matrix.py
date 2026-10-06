"""Pinned study model declarations, separate from executable qualifications."""
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name('MULTISOURCE_MODEL_MATRIX.json')


def matrix(path=MANIFEST):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['models']
    for field in ('id', 'model'):
        if len({r[field].casefold() for r in rows}) != len(rows):
            raise ValueError(f'duplicate {field}; aliases must not increase the model count')
    if len(rows) < 20:
        raise ValueError('at least 20 distinct planned models required')
    for row in rows:
        if row['transport'] == 'hf':
            revision = row.get('revision', '')
            if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
                raise ValueError('open-weight revisions must be full commit hashes')
    return rows


def hf_models(rows, legacy):
    """Resolve legacy names without silently changing a declared model or revision."""
    known = {name.casefold(): (family, name, revision) for family, name, revision in legacy}
    reserved = {family for family, name, revision in legacy}
    models = []
    for row in rows:
        if row['transport'] != 'hf':
            continue
        previous = known.get(row['model'].casefold())
        if previous:
            if row['revision'] != previous[2]:
                raise ValueError(f"matrix revision differs from legacy pin: {row['model']}")
            models.append(previous)
        else:
            if row['id'] in reserved:
                raise ValueError(f"matrix id collides with legacy model: {row['id']}")
            models.append((row['id'], row['model'], row['revision']))
    if len({m[0] for m in models}) != len(models):
        raise ValueError('duplicate resolved model id')
    return tuple(models)
